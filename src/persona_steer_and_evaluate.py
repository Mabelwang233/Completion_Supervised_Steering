"""
Persona-concept version of steer_and_evaluate.py -- same steering logic
(all layers, unit-norm directions, single global alpha) as the fear
version, but judged with persona_judge.judge_response(persona, question,
response) instead, since the judge needs to know WHICH persona is the
target, not just the question.

Test set is the fixed list of 50 generic persona-eliciting questions --
no held-out statements, matching the Fig 4A/4B design (statements are
only ever used for training).
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
from persona_judge import judge_response


LLAMA_ASSISTANT_TAG = '<|start_header_id|>assistant<|end_header_id|>'
QWEN_ASSISTANT_TAG  = '<|im_start|>assistant\n'


def get_assistant_tag(tokenizer):
    """Return the correct assistant-turn tag for this model family."""
    vocab = tokenizer.get_vocab()
    if '<|start_header_id|>' in vocab:
        return LLAMA_ASSISTANT_TAG
    if '<|im_start|>' in vocab:
        return QWEN_ASSISTANT_TAG
    return ''



def format_chat(tokenizer, prompt_text):
    chat = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(chat, tokenize=False) + get_assistant_tag(tokenizer)


def load_directions(baseline_path=None, seqrfm_path=None, completion_rfm_path=None, bag_of_words_path=None,
                     completion_diff_means_path=None, bag_of_words_diff_means_path=None,
                     seqrfm_method_name="sequence_rfm"):
    directions = {}
    if baseline_path:
        with open(baseline_path, "rb") as f:
            baselines = pickle.load(f)
        for method, d in baselines.items():
            directions[method] = d["directions_per_layer"]

    if seqrfm_path:
        seqrfm = torch.load(seqrfm_path)
        seqrfm_directions = {}
        for layer, d in seqrfm["directions_per_layer"].items():
            vec = d[:, 0]
            seqrfm_directions[layer] = vec
        # seqrfm_method_name lets a variant (e.g. linear_late temporal
        # weighting) be labeled distinctly in the output CSV instead of
        # colliding with the name "sequence_rfm" used by the uniform-weight
        # run.
        directions[seqrfm_method_name] = seqrfm_directions

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
            bagwords_directions[layer] = d[:, 0]
        directions["bag_of_words_rfm"] = bagwords_directions

    # optional 6th method: last-token embedding of (x, z), diff-means
    # instead of RFM -- same pickle shape as completion_rfm's output.
    if completion_diff_means_path:
        with open(completion_diff_means_path, "rb") as f:
            completion_dm = pickle.load(f)
        directions["completion_diff_means"] = completion_dm["directions_per_layer"]

    # optional 7th method: mean-pooled anchor activations, diff-means
    # instead of RFM -- the closest analog to Persona Vectors' actual
    # recipe. Same .pt shape as bag_of_words_rfm's output.
    if bag_of_words_diff_means_path:
        bagwords_dm = torch.load(bag_of_words_diff_means_path)
        bagwords_dm_directions = {}
        for layer, d in bagwords_dm["directions_per_layer"].items():
            bagwords_dm_directions[layer] = d[:, 0]
        directions["bag_of_words_diff_means"] = bagwords_dm_directions

    if not directions:
        raise ValueError("load_directions: no method paths provided -- nothing to evaluate. "
                          "Pass at least one of baseline_path / seqrfm_path / completion_rfm_path / bag_of_words_path.")

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
                    cache_dir=None, seed=0, completion_rfm_path=None, bag_of_words_path=None,
                    completion_diff_means_path=None, bag_of_words_diff_means_path=None,
                    seqrfm_method_name="sequence_rfm"):
    torch.manual_seed(seed)

    with open(dataset_path) as f:
        dataset = json.load(f)
    persona = dataset["persona"]
    test_examples = dataset["test"]  # 50 fixed questions, no held-out statements

    language_model, tokenizer = load_model(model_name, cache_dir=cache_dir)
    language_model.eval()

    directions = load_directions(baseline_path, seqrfm_path, completion_rfm_path, bag_of_words_path,
                                  completion_diff_means_path, bag_of_words_diff_means_path,
                                  seqrfm_method_name=seqrfm_method_name)
    print(f"Persona: {persona}")
    for method, per_layer in directions.items():
        print(f"  {method}: directions at {len(per_layer)} layers")
    print(f"  alpha grid: {alpha_grid}")

    import os
    os.makedirs(out_dir, exist_ok=True)
    rows_path = os.path.join(out_dir, "per_example_results.csv")
    summary_path = os.path.join(out_dir, "summary.csv")

    all_rows = []
    with open(rows_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["persona", "method", "alpha", "question", "response", "score", "explanation"])

        for method, per_layer_directions in directions.items():
            for alpha in alpha_grid:
                print(f"\n=== persona={persona}  method={method}  alpha={alpha} ===")
                for ex in tqdm(test_examples):
                    response = generate_steered(language_model, tokenizer, ex["prompt"],
                                                 per_layer_directions, alpha, max_new_tokens)
                    score, explanation = judge_response(persona, ex["question"], response, judge_model)
                    row = [persona, method, alpha, ex["question"], response, score, explanation]
                    writer.writerow(row)
                    all_rows.append({"method": method, "alpha": alpha, "score": score})

    from collections import defaultdict
    agg = defaultdict(list)
    for r in all_rows:
        agg[(r["method"], r["alpha"])].append(r["score"])

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["persona", "method", "alpha", "mean_score", "n"])
        for (method, alpha), scores in sorted(agg.items()):
            writer.writerow([persona, method, alpha, sum(scores) / len(scores), len(scores)])

    print(f"\nWrote {rows_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--baseline_path", default=None,
                    help="optional: pickle from baselines.py, adds mean_difference + rfm to the sweep. "
                         "Omit to evaluate ONLY the methods you explicitly pass paths for.")
    p.add_argument("--seqrfm_path", default=None,
                    help="optional: .pt from train_sequence_rfm.py, adds a sequence-RFM variant to the sweep")
    p.add_argument("--seqrfm_method_name", default="sequence_rfm",
                    help="label used in the output CSV for --seqrfm_path's method -- override this "
                         "(e.g. 'sequence_rfm_linear_late') when evaluating a variant, so it doesn't "
                         "collide with an existing 'sequence_rfm' run's results")
    p.add_argument("--out_dir", required=True)
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
    p.add_argument("--completion_diff_means_path", default=None,
                    help="optional: pickle from completion_diff_means.py, adds a 6th method to the sweep")
    p.add_argument("--bag_of_words_diff_means_path", default=None,
                    help="optional: .pt from bag_of_words_diff_means.py, adds a 7th method to the sweep")
    args = p.parse_args()

    run_evaluation(args.dataset_path, args.baseline_path, args.seqrfm_path, args.out_dir,
                    args.model, args.alpha_grid, args.judge_model,
                    args.eval_n_tokens, args.cache_dir, args.seed,
                    completion_rfm_path=args.completion_rfm_path,
                    bag_of_words_path=args.bag_of_words_path,
                    completion_diff_means_path=args.completion_diff_means_path,
                    bag_of_words_diff_means_path=args.bag_of_words_diff_means_path,
                    seqrfm_method_name=args.seqrfm_method_name)
