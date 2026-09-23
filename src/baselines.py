"""
Baselines: RFM and diff-means, via the existing NeuralController /
Toolkit machinery -- NOT reimplemented. Trained at ALL layers (the
NeuralController default, matching the notebook's own default of
list(range(-1, -num_hidden_layers, -1))), not a single layer, so
generation-time steering can apply per-layer directions at every layer
exactly like controller.generate(layers_to_control=range(-1,-31,-1)).

These use only the last hidden state of the PROMPT (no completions),
exactly matching the original single-vector method -- intentional, this
is the correct point of comparison for sequence-RFM.
"""
import argparse
import json
import pickle
import sys

import torch

sys.path.insert(0, ".")
from utils import load_model
from neural_controllers import NeuralController


def train_baselines(dataset_path, out_path, model_name, rfm_iters=8,
                     n_components=1, batch_size=8, cache_dir=None, seed=0):
    torch.manual_seed(seed)

    with open(dataset_path) as f:
        dataset = json.load(f)

    train_prompts = [ex["prompt"] for ex in dataset["train"]]
    train_labels = [ex["label"] for ex in dataset["train"]]

    language_model, tokenizer = load_model(model_name, cache_dir=cache_dir)

    results = {}
    for method in ["rfm", "mean_difference"]:
        print(f"\n=== training baseline: {method} (all layers) ===")
        controller = NeuralController(
            language_model, tokenizer,
            control_method=method,
            rfm_iters=rfm_iters,
            n_components=n_components,
            batch_size=batch_size,
        )
        # hidden_layers=None -> controller's default, i.e. all layers,
        # matching the notebook.
        controller.compute_directions(train_prompts, train_labels)
        # top component per layer, matching what generation_utils.hook_model expects
        per_layer_direction = {
            layer: comps[0].detach().cpu()
            for layer, comps in controller.directions.items()
        }
        results[method] = {"directions_per_layer": per_layer_direction,
                            "layers": controller.hidden_layers,
                            "sign": controller.signs}

    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", default="../data/shakespeare_dataset.json")
    p.add_argument("--out_path", default="../outputs/baseline_directions.pkl")
    p.add_argument("--model", required=True)
    p.add_argument("--rfm_iters", type=int, default=8)
    p.add_argument("--n_components", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    train_baselines(args.dataset_path, args.out_path, args.model,
                     args.rfm_iters, args.n_components, args.batch_size,
                     args.cache_dir, args.seed)
