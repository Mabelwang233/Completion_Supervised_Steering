"""
Completion diff-means: same idea as completion_rfm.py (last-token
embedding of the full (x, z) sequence), but extracted via mean-difference
instead of RFM/AGOP -- i.e. control_method="mean_difference" instead of
"rfm". Otherwise identical: same input string construction, same
NeuralController machinery, same output shape.

    completion RFM         : last-token embedding of (x, z), RFM/AGOP
    completion diff-means  : last-token embedding of (x, z), diff-means  <- this file

Why add this: the closest published analog to our completion-aware
methods (Persona Vectors, Chen et al. 2025) uses diff-means, not RFM.
This gives us a direct, same-data comparison against that convention,
rather than relying on a different paper's numbers on different models.
"""
import argparse
import json
import pickle
import sys

import torch

sys.path.insert(0, ".")
from utils import load_model
from neural_controllers import NeuralController


def train_completion_diff_means(dataset_with_completions_path, out_path, model_name,
                                 batch_size=8, cache_dir=None, seed=0):
    torch.manual_seed(seed)

    with open(dataset_with_completions_path) as f:
        dataset = json.load(f)

    train_examples = dataset["train"]
    full_texts = [ex["formatted_prompt"] + ex["completion"] for ex in train_examples]
    train_labels = [ex["label"] for ex in train_examples]

    language_model, tokenizer = load_model(model_name, cache_dir=cache_dir)

    print(f"Training completion-diff-means on {len(full_texts)} (x,z) pairs, all layers...")
    controller = NeuralController(
        language_model, tokenizer,
        control_method="mean_difference",
        batch_size=batch_size,
    )
    controller.compute_directions(full_texts, train_labels)

    per_layer_direction = {
        layer: comps[0].detach().cpu()
        for layer, comps in controller.directions.items()
    }
    result = {
        "directions_per_layer": per_layer_direction,
        "layers": controller.hidden_layers,
        "sign": controller.signs,
    }

    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_with_completions_path", required=True)
    p.add_argument("--out_path", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    train_completion_diff_means(args.dataset_with_completions_path, args.out_path, args.model,
                                 args.batch_size, args.cache_dir, args.seed)
