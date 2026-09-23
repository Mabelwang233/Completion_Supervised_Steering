"""
Step 6-7: steer generation on the 50 fixed test questions using each
method's PER-LAYER directions, applied at ALL layers simultaneously with
a single global coefficient -- exactly matching the notebook's
controller.generate(layers_to_control=list(range(-1,-31,-1)), control_coef=0.6)
pattern, not single-layer steering.

All directions (RFM, diff-means, sequence-RFM) are already unit-norm at
the source (verified: MeanDifferenceToolkit normalizes explicitly,
RFM/sequence-RFM directions come from eigendecomposition/SVD which are
unit-norm by construction) -- so alpha_grid values are directly
comparable to the notebook's control_coef=0.6 convention.
"""
import argparse
import csv
import json
import pickle
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, ".")
from utils import load_model
import generation_utils
from judge import judge_response

ASSISTANT_TAG = '<|start_header_id|>assistant<|end_header_id|>'


def format_chat(tokenizer, prompt_text):
    chat = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(chat, tokenize=False) + ASSISTANT_TAG


def load_directions(baseline_path, seqrfm_path, completion_rfm_path=None, bag_of_words_path=None):
    """Returns {method: {layer: direction_tensor}}"""
    directions = {}

    with open(baseline_path, "rb") as f:
        baselines = pickle.load(f)
    for method, d in baselines.items():
        directions[method] = d["directions_per_layer"]  # already unit-norm, see MeanDifferenceToolkit

    seqrfm = torch.load(seqrfm_path)
    seqrfm_directions = {}
    for layer, d in seqrfm["directions_per_layer"].items():
        vec = d[:, 0].float()  # top component, already unit-norm (SVD)
        seqrfm_directions[layer] = vec
    directions["sequence_rfm"] = seqrfm_directions

    # optional 4th method: last-token embedding of (x, z) -- sits between
    # vanilla RFM (last token of x, already in `baselines`) and sequence-RFM
    # (all tokens of z, above). Same pickle shape as baseline_directions.pkl.
    if completion_rfm_path:
        with open(completion_rfm_path, "rb") as f:
            completion_rfm = pickle.load(f)
        directions["completion_rfm"] = completion_rfm["directions_per_layer"]

    # optional 5th method: bag-of-words sequence-RFM (k_tau === 1, position
    # ignored) -- same .pt shape as sequence_rfm_directions.pt.
    if bag_of_words_path:
        bagwords = torch.load(bag_of_words_path)
        bagwords_directions = {}
        for layer, d in bagwords["directions_per_layer"].items():
            bagwords_directions[layer] = d[:, 0].float()
        directions["bag_of_words_rfm"] = bagwords_directions

    return directions


def generate_steered(language_model, tokenizer, prompt_text, directions_per_layer,
                      alpha, max_new_tokens=100):
    formatted = format_chat(tokenizer, prompt_text)
    inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to(language_model.device)

    layers = list(directions_per_layer.keys())
    directions_for_hook = {
        layer: [directions_per_layer[layer].to(language_model.device)]
        for layer in layers
    }
    hooks = generation_utils.hook_model(language_model, directions_for_hook, layers, alpha)
    try:
        with torch.no_grad():
            output_ids = language_model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        generation_utils.clear_hooks(hooks)

    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def run_evaluation(dataset_path, baseline_path, seqrfm_path, out_dir, model_name,
                    alpha_grid, judge_model="gpt-4o-mini", max_new_tokens=100,
                    cache_dir=None, seed=0, completion_rfm_path=None, bag_of_words_path=None):
    torch.manual_seed(seed)

    with open(dataset_path) as f:
        dataset = json.load(f)
    test_prompts = [ex["prompt"] for ex in dataset["test"]]
    test_sources = [ex.get("source", "statement") for ex in dataset["test"]]

    language_model, tokenizer = load_model(model_name, cache_dir=cache_dir)
    language_model.eval()

    directions = load_directions(baseline_path, seqrfm_path, completion_rfm_path, bag_of_words_path)
    for method, per_layer in directions.items():
        print(f"{method}: directions at {len(per_layer)} layers")
    print(f"alpha grid: {alpha_grid}")

    import os
    os.makedirs(out_dir, exist_ok=True)
    rows_path = os.path.join(out_dir, "per_example_results.csv")
    summary_path = os.path.join(out_dir, "summary.csv")

    all_rows = []
    with open(rows_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "alpha", "prompt_idx", "source", "prompt", "response", "score", "explanation"])

        for method, per_layer_directions in directions.items():
            for alpha in alpha_grid:
                print(f"\n=== method={method}  alpha={alpha} ===")
                for i, prompt in enumerate(tqdm(test_prompts)):
                    response = generate_steered(language_model, tokenizer, prompt,
                                                 per_layer_directions, alpha, max_new_tokens)
                    score, explanation = judge_response(prompt, response, judge_model)
                    row = [method, alpha, i, test_sources[i], prompt, response, score, explanation]
                    writer.writerow(row)
                    all_rows.append({"method": method, "alpha": alpha, "source": test_sources[i], "score": score})

    from collections import defaultdict
    agg = defaultdict(list)
    agg_by_source = defaultdict(list)
    for r in all_rows:
        agg[(r["method"], r["alpha"])].append(r["score"])
        agg_by_source[(r["method"], r["alpha"], r["source"])].append(r["score"])

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "alpha", "source", "mean_score", "n"])
        for (method, alpha), scores in sorted(agg.items()):
            writer.writerow([method, alpha, "all", sum(scores) / len(scores), len(scores)])
        for (method, alpha, source), scores in sorted(agg_by_source.items()):
            writer.writerow([method, alpha, source, sum(scores) / len(scores), len(scores)])

    print(f"\nWrote {rows_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", default="../data/shakespeare_dataset.json")
    p.add_argument("--baseline_path", default="../outputs/baseline_directions.pkl")
    p.add_argument("--seqrfm_path", default="../outputs/sequence_rfm_directions.pt")
    p.add_argument("--out_dir", default="../outputs")
    p.add_argument("--model", required=True)
    p.add_argument("--alpha_grid", type=float, nargs="+", required=True)
    p.add_argument("--judge_model", default="gpt-4o-mini")
    p.add_argument("--eval_n_tokens", type=int, default=100)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--completion_rfm_path", default=None,
                    help="optional: pickle from completion_rfm.py, adds a 4th method to the sweep")
    p.add_argument("--bag_of_words_path", default=None,
                    help="optional: .pt from train_sequence_rfm.py --bag_of_words, adds a 5th method")
    args = p.parse_args()

    run_evaluation(args.dataset_path, args.baseline_path, args.seqrfm_path, args.out_dir,
                    args.model, args.alpha_grid, args.judge_model,
                    args.eval_n_tokens, args.cache_dir, args.seed,
                    completion_rfm_path=args.completion_rfm_path,
                    bag_of_words_path=args.bag_of_words_path)
