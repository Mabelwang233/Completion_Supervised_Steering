"""
Domain-agnostic version of eval_one_config.py: works for both Personas
and Fears, since the two domains' generate_steered / judge_response
functions have the same signature shape -- only the module to import
from and the dataset's top-level concept key ("persona" vs "fear")
differ. See DOMAIN_CONFIG below.
"""
import csv
import importlib
import json
import os
from collections import defaultdict

import torch
from tqdm import tqdm

DOMAIN_CONFIG = {
    "persona": {
        "steer_module": "persona_steer_and_evaluate",
        "judge_module": "persona_judge",
        "dataset_key": "persona",
    },
    "fear": {
        "steer_module": "fear_steer_and_evaluate",
        "judge_module": "fear_judge",
        "dataset_key": "fear",
    },
}


def get_domain_fns(domain):
    if domain not in DOMAIN_CONFIG:
        raise ValueError(f"unknown domain: {domain!r} -- must be one of {list(DOMAIN_CONFIG)}")
    cfg = DOMAIN_CONFIG[domain]
    steer_mod = importlib.import_module(cfg["steer_module"])
    judge_mod = importlib.import_module(cfg["judge_module"])
    return steer_mod.generate_steered, judge_mod.judge_response, cfg["dataset_key"]


def evaluate_one_config(language_model, tokenizer, dataset_path, eval_subset_path,
                         method_label, directions_per_layer, alpha_grid, out_dir,
                         domain, judge_model="gpt-4o-mini", max_new_tokens=100, seed=0,
                         append=True):
    torch.manual_seed(seed)

    generate_steered, judge_response, dataset_key = get_domain_fns(domain)

    with open(dataset_path) as f:
        dataset = json.load(f)
    concept = dataset[dataset_key]
    test_examples = dataset["test"]

    with open(eval_subset_path) as f:
        subset = json.load(f)
    eval_examples = [test_examples[i] for i in subset["indices"]]
    assert len(eval_examples) == subset["n"], \
        "eval_subset indices don't match dataset test set length -- was this subset built from a different dataset_with_completions.json?"

    os.makedirs(out_dir, exist_ok=True)
    rows_path = os.path.join(out_dir, "per_example_results.csv")
    write_header = append and not os.path.exists(rows_path)
    mode = "a" if append else "w"

    rows_for_summary = []
    with open(rows_path, mode, newline="") as f:
        writer = csv.writer(f)
        if write_header or not append:
            # "concept" column holds the persona name or fear name depending on
            # domain -- kept generic (rather than "persona"/"fear") so a single
            # per_example_results.csv schema works for both domains.
            writer.writerow(["domain", "concept", "method", "alpha", "question", "response", "score", "explanation"])

        for alpha in alpha_grid:
            print(f"\n=== domain={domain}  concept={concept}  method={method_label}  alpha={alpha}  (n={len(eval_examples)}) ===")
            for ex in tqdm(eval_examples):
                response = generate_steered(language_model, tokenizer, ex["prompt"],
                                             directions_per_layer, alpha, max_new_tokens)
                score, explanation = judge_response(concept, ex["question"], response, judge_model)
                writer.writerow([domain, concept, method_label, alpha, ex["question"], response, score, explanation])
                rows_for_summary.append({"alpha": alpha, "score": score})

    agg = defaultdict(list)
    for r in rows_for_summary:
        agg[r["alpha"]].append(r["score"])
    best_alpha, best_mean = max(
        ((a, sum(s) / len(s)) for a, s in agg.items()), key=lambda kv: kv[1]
    )
    print(f"  best alpha for {method_label}: {best_alpha} (mean score {best_mean:.3f})")

    return {
        "domain": domain,
        "concept": concept,
        "method": method_label,
        "per_alpha": {a: sum(s) / len(s) for a, s in agg.items()},
        "best_alpha": best_alpha,
        "best_mean_score": best_mean,
        "n_eval": len(eval_examples),
    }