"""
Step 3: extract activation trajectories P^ell_i from (prompt, completion)
pairs via a single teacher-forced forward pass, at ALL requested layers
-- matching the neural_controllers notebook's convention of training a
separate direction per layer and steering all of them simultaneously
(layers_to_control=list(range(-1,-31,-1)), not a single layer.

Extracting all layers costs nothing extra per example: output_hidden_states=True
already returns every layer from one forward pass, we just keep more of
what's already computed instead of discarding it. The real cost
multiplier from going multi-layer is downstream, in train_sequence_rfm.py
-- see that file's docstring.

Anchor-position subsampling (unchanged from the single-layer version):
each completion's hidden states are subsampled to n_anchors positions
per layer before being stored, to keep the O(n^2 * T^2) sequence-kernel
sum tractable.
"""
import argparse
import json
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, ".")
from utils import load_model


def subsample_anchors(hidden_states, n_anchors):
    """hidden_states: (T, d) for the completion span only.
    Returns (h_sub: (n_anchors_actual, d), tau: (n_anchors_actual,))."""
    T = hidden_states.shape[0]
    if T <= n_anchors:
        idx = torch.arange(T)
    else:
        idx = torch.linspace(0, T - 1, n_anchors).round().long().unique()
    h_sub = hidden_states[idx]
    tau = idx.float() / max(T - 1, 1)
    return h_sub, tau


def resolve_layers(layers_arg, num_layers):
    """None/'all' -> every layer (notebook default: range(-1, -num_layers, -1)).
    Otherwise a list of explicit layer indices."""
    if layers_arg is None or layers_arg == ["all"]:
        return list(range(-1, -num_layers, -1))
    return [int(l) for l in layers_arg]


def extract_trajectories(dataset_path, out_path, model_name, layers=None,
                          n_anchors=25, cache_dir=None, seed=0):
    torch.manual_seed(seed)

    with open(dataset_path) as f:
        dataset = json.load(f)

    language_model, tokenizer = load_model(model_name, cache_dir=cache_dir)
    language_model.eval()
    num_layers = len(language_model.model.layers)
    layers = resolve_layers(layers, num_layers)
    print(f"Extracting trajectories at {len(layers)} layers: {layers}")

    train_examples = dataset["train"]
    print(f"Extracting trajectories for {len(train_examples)} training examples...")

    # per-layer accumulation: {layer: [ {h, tau}, ... ]}
    trajectories_per_layer = {l: [] for l in layers}
    labels = []

    with torch.no_grad():
        for ex in tqdm(train_examples):
            full_text = ex["formatted_prompt"] + ex["completion"]
            prompt_ids = tokenizer(ex["formatted_prompt"], return_tensors="pt",
                                    add_special_tokens=False).input_ids
            full_ids = tokenizer(full_text, return_tensors="pt",
                                  add_special_tokens=False).to(language_model.device)

            outputs = language_model(**full_ids, output_hidden_states=True)
            prompt_len = prompt_ids.shape[1]

            skip_example = False
            per_layer_result = {}
            for layer in layers:
                idx = layer if layer >= 0 else num_layers + layer + 1
                layer_hidden = outputs.hidden_states[idx][0]  # (seq_len, d)
                completion_hidden = layer_hidden[prompt_len:].detach().cpu()

                if completion_hidden.shape[0] == 0:
                    skip_example = True
                    break

                h_sub, tau = subsample_anchors(completion_hidden, n_anchors)
                per_layer_result[layer] = {"h": h_sub, "tau": tau}

            if skip_example:
                continue

            for layer in layers:
                trajectories_per_layer[layer].append(per_layer_result[layer])
            labels.append(1 if ex["label"] == 1 else -1)

    torch.save({"trajectories_per_layer": trajectories_per_layer,
                "labels": torch.tensor(labels), "layers": layers}, out_path)
    n_kept = len(labels)
    avg_anchors = sum(t["h"].shape[0] for t in trajectories_per_layer[layers[0]]) / max(n_kept, 1)
    print(f"Wrote {out_path}  ({n_kept} trajectories x {len(layers)} layers, "
          f"avg {avg_anchors:.1f} anchors/trajectory)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", default="../data/shakespeare_dataset_with_completions.json")
    p.add_argument("--out_path", default="../data/trajectories.pt")
    p.add_argument("--model", required=True)
    p.add_argument("--layers", nargs="+", default=None,
                    help="explicit layer indices, or omit for all layers (notebook default)")
    p.add_argument("--n_anchors", type=int, default=25)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    extract_trajectories(args.dataset_path, args.out_path, args.model, args.layers,
                          args.n_anchors, args.cache_dir, seed=args.seed)
