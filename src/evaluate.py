"""
evaluate.py

Loads jointly trained directions.pt + controllers.pt and evaluates on
the 50 fixed fear test questions using the same fear_judge as the rest
of the codebase.

At inference time we steer ALL tokens (not just completion tokens), using
alpha_t = C_l @ [h_t; 1] learned during training.

Outputs (in --out_dir/<concept>/):
  eval_results.csv    -- per-question scores
  eval_summary.json   -- mean score + metadata

Usage:
    python evaluate.py \\
        --dataset data/fear_blood.json \\
        --out_dir outputs \\
        --model meta-llama/Llama-3.1-8B-Instruct \\  # or Qwen/Qwen2.5-7B-Instruct, etc.
        --judge_model gpt-4o-mini \\
        --max_new_tokens 100
"""
import argparse
import csv
import json
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, ".")
from utils import load_model
from fear_judge import judge_response as fear_judge_response
try:
    from persona_judge import judge_response as persona_judge_response
except ImportError:
    print("WARNING: persona_judge.py not found -- falling back to fear_judge for personas.")
    persona_judge_response = fear_judge_response

def format_chat(tokenizer, prompt_text):
    """Format a user prompt using the tokenizer's own chat template.
    Works with any model (Llama, Qwen, Mistral, etc.) as long as the
    tokenizer ships a chat template."""
    chat = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


class LayerController(torch.nn.Module):
    """
    alpha_t = W @ [h_t; e_y; 1]
    Input dim: d+2. At inference always pass e_y=1.0.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size + 2, 1, bias=False)

    def forward(self, h, e_y=1.0):
        T    = h.shape[0]
        ey   = torch.full((T, 1), e_y, device=h.device, dtype=h.dtype)
        ones = torch.ones( T,  1,      device=h.device, dtype=h.dtype)
        h_aug = torch.cat([h, ey, ones], dim=-1)
        return self.linear(h_aug).squeeze(-1)


def install_hooks(model, directions, controllers, layers):
    hooks = []

    def make_hook(layer_idx):
        def hook(module, args):
            h = args[0]       # (1, T, d)
            h_2d = h[0]       # (T, d)
            d_vec = directions[layer_idx].to(device=h.device, dtype=h.dtype)  # match h's dtype
            ctrl  = controllers[layer_idx]
            with torch.no_grad():
                alpha = ctrl(h_2d, e_y=1.0)                     # (T,) — always e_y=1 at inference
            delta = (alpha.unsqueeze(-1) * d_vec.unsqueeze(0)).to(dtype=h.dtype)  # (T, d)
            return (( h[0] + delta).unsqueeze(0),) + args[1:]
        return hook

    for l in layers:
        hook = model.model.layers[l].register_forward_pre_hook(make_hook(l))
        hooks.append(hook)
    return hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


def generate_steered(model, tokenizer, prompt_text, directions, controllers, layers,
                     max_new_tokens=100):
    formatted = format_chat(tokenizer, prompt_text)
    inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to(model.device)

    hooks = install_hooks(model, directions, controllers, layers)
    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,   # suppress Qwen config sampling flags
                top_p=1.0,
                top_k=0,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        remove_hooks(hooks)

    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",        required=True)
    p.add_argument("--out_dir",        default="outputs/models")
    p.add_argument("--model",          required=True)
    p.add_argument("--judge_model",    default="gpt-4o-mini")
    p.add_argument("--max_new_tokens", type=int, default=100)
    p.add_argument("--cache_dir",      default=None)
    args = p.parse_args()

    with open(args.dataset) as f:
        dataset = json.load(f)
    concept      = dataset.get("concept") or dataset.get("persona") or dataset.get("fear")
    concept_type = dataset.get("type", "persona" if "persona" in dataset else "fear")
    slug         = concept.lower().replace(" ", "_")
    test_examples = dataset["test"]

    # Select judge based on concept type
    if concept_type == "persona":
        judge_fn = persona_judge_response
    else:
        judge_fn = fear_judge_response

    print(f"Concept: {concept}  |  type: {concept_type}  |  judge: {concept_type}_judge")

    concept_dir = os.path.join(args.out_dir, slug)
    dir_path  = os.path.join(concept_dir, "directions.pt")
    ctrl_path = os.path.join(concept_dir, "controllers.pt")

    directions_raw = torch.load(dir_path, map_location="cpu")
    ctrl_raw       = torch.load(ctrl_path, map_location="cpu")

    # Re-normalize directions (should already be unit norm, but be safe)
    directions = {int(l): F.normalize(v, dim=0) for l, v in directions_raw.items()}
    layers = sorted(directions.keys())

    model, tokenizer = load_model(args.model, args.cache_dir)
    model.eval()
    d = model.config.hidden_size

    model_dtype = model.dtype
    controllers = {}
    for l in layers:
        # Cast saved state_dict to model dtype before loading — controllers are
        # saved as float32 from training but Qwen (and other models) run in bfloat16.
        state = {k: v.to(dtype=model_dtype) for k, v in ctrl_raw[l].items()}
        ctrl = LayerController(d).to(device=model.device, dtype=model_dtype)
        ctrl.load_state_dict(state)
        ctrl.eval()
        controllers[l] = ctrl

    rows = []
    scores = []
    print(f"Evaluating {concept} on {len(test_examples)} questions...")
    for ex in tqdm(test_examples):
        response = generate_steered(
            model, tokenizer, ex["prompt"], directions, controllers, layers, args.max_new_tokens
        )
        score, explanation = judge_fn(concept, ex["question"], response, args.judge_model)
        rows.append({
            "concept": concept,
            "question": ex["question"],
            "response": response,
            "score": score,
            "explanation": explanation,
        })
        scores.append(score)

    out_csv  = os.path.join(concept_dir, "eval_results.csv")
    out_json = os.path.join(concept_dir, "eval_summary.json")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["concept","question","response","score","explanation"])
        w.writeheader()
        w.writerows(rows)

    mean_score = sum(scores) / len(scores)
    summary = {
        "concept":    concept,
        "n_questions": len(scores),
        "mean_score":  mean_score,
        "directions_path":   dir_path,
        "controllers_path":  ctrl_path,
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nMean score: {mean_score:.4f}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()