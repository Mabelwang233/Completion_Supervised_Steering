"""
Persona steer+evaluate using a TRAINED CONTROLLER instead of a
grid-searched global alpha -- the controller-based analog of
persona_steer_and_evaluate.py. No alpha_grid: the controller decides
per-token, per-layer steering strength on its own (that's the entire
point of Section 6).

Reuses persona_judge.judge_response exactly as persona_steer_and_evaluate.py
does, and the fixed 50-question test set from the persona dataset json
(dataset["test"]), so results are directly comparable to your existing
alpha-grid sweep's summary.csv (same personas, same questions, same
judge).
"""
import argparse
import csv
import json
import os

import torch
from tqdm import tqdm

from utils import load_model
from persona_judge import judge_response
from controller import SequenceRFMController, load_directions_for_k, seqrfm_method_label
from train_controller import get_assistant_tag


def run_evaluation(dataset_path, controller_path, out_dir, model_name,
                    judge_model="gpt-4o-mini", max_new_tokens=100, cache_dir=None, seed=0):
    torch.manual_seed(seed)

    with open(dataset_path) as f:
        dataset = json.load(f)
    persona = dataset["persona"]
    test_examples = dataset["test"]

    ckpt = torch.load(controller_path)
    k = ckpt["k"]
    # Read from the checkpoint if train_controller.py already recorded it
    # (newer checkpoints), otherwise derive it from the directions file
    # itself -- either way, the directions file is the ground truth for
    # which RFM variant actually produced D_ell.
    method = ckpt.get("seqrfm_method") or seqrfm_method_label(ckpt["seqrfm_path"])
    method_label = f"{method}_controller"

    language_model, tokenizer = load_model(model_name, cache_dir=cache_dir)
    language_model.eval()
    d_model = language_model.config.hidden_size

    directions_per_layer = load_directions_for_k(ckpt["seqrfm_path"], k=k)
    controller = SequenceRFMController(directions_per_layer, d_model,
                                        dtype=language_model.dtype).to(language_model.device)
    controller.load_state_dict(ckpt["controller_state_dict"])
    controller.eval()

    print(f"Persona: {persona}  |  method={method_label}  k={k}  layers={len(controller.layers)}")

    os.makedirs(out_dir, exist_ok=True)
    rows_path = os.path.join(out_dir, "per_example_results.csv")
    summary_path = os.path.join(out_dir, "summary.csv")

    scores = []
    with open(rows_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["persona", "method", "k", "question", "response", "score", "explanation"])

        for ex in tqdm(test_examples):
            chat = [{"role": "user", "content": ex["prompt"]}]
            formatted = tokenizer.apply_chat_template(chat, tokenize=False) + \
                get_assistant_tag(tokenizer)
            response = controller.controlled_generate(
                language_model, tokenizer, formatted, max_new_tokens=max_new_tokens)
            score, explanation = judge_response(persona, ex["question"], response, judge_model)
            writer.writerow([persona, method_label, k, ex["question"], response, score, explanation])
            scores.append(score)

    mean_score = sum(scores) / len(scores) if scores else float("nan")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["persona", "method", "k", "mean_score", "n"])
        writer.writerow([persona, method_label, k, mean_score, len(scores)])

    print(f"\nWrote {rows_path}")
    print(f"Wrote {summary_path}  (mean_score={mean_score:.3f}, n={len(scores)})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--controller_path", required=True,
                    help="output of train_controller.py's --out_path")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--judge_model", default="gpt-4o-mini")
    p.add_argument("--eval_n_tokens", type=int, default=100)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run_evaluation(args.dataset_path, args.controller_path, args.out_dir, args.model,
                    args.judge_model, args.eval_n_tokens, args.cache_dir, args.seed)