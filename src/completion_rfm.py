"""
Completion RFM: sits strictly between vanilla RFM and sequence-RFM.

    vanilla RFM     : last-token embedding of x               (baselines.py)
    completion RFM  : last-token embedding of (x, z)           <- this file
    sequence RFM    : all token embeddings of (x, z_<t)        (train_sequence_rfm.py)

Reuses the SAME completions already generated for sequence-RFM
(dataset_with_completions.json from generate_completions.py) -- no new
generation happens here, this just pools/uses them a different way: one
last-token vector per example instead of the full trajectory.

Trained via the existing NeuralController/RFMToolkit machinery, exactly
like baselines.py's "rfm" branch -- the only difference is what string
gets tokenized (formatted_prompt + completion, instead of formatted_prompt
alone), which is what makes the last-token activation reflect (x, z)
rather than just x.
"""
import argparse
import json
import pickle
import sys

import torch

sys.path.insert(0, ".")
from utils import load_model
from neural_controllers import NeuralController


def train_completion_rfm(dataset_with_completions_path, out_path, model_name,
                          rfm_iters=8, n_components=1, batch_size=8,
                          cache_dir=None, seed=0):
    torch.manual_seed(seed)

    with open(dataset_with_completions_path) as f:
        dataset = json.load(f)

    train_examples = dataset["train"]
    # formatted_prompt already includes the chat-template + assistant tag
    # (added by generate_completions.py); appending completion gives the
    # full (x, z) string whose LAST token we take as the feature.
    full_texts = [ex["formatted_prompt"] + ex["completion"] for ex in train_examples]
    train_labels = [ex["label"] for ex in train_examples]

    language_model, tokenizer = load_model(model_name, cache_dir=cache_dir)

    print(f"Training completion-RFM on {len(full_texts)} (x,z) pairs, all layers...")
    controller = NeuralController(
        language_model, tokenizer,
        control_method="rfm",
        rfm_iters=rfm_iters,
        n_components=n_components,
        batch_size=batch_size,
    )
    # get_hidden_states defaults to rep_token=-1, all_positions=False --
    # i.e. the LAST token of whatever string is passed in. Passing
    # full_texts (x+z) instead of prompts alone is the entire difference
    # from vanilla RFM; no other code path changes.
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
    p.add_argument("--dataset_with_completions_path", required=True,
                    help="the dataset_with_completions.json already produced for this concept")
    p.add_argument("--out_path", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--rfm_iters", type=int, default=8)
    p.add_argument("--n_components", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    train_completion_rfm(args.dataset_with_completions_path, args.out_path, args.model,
                          args.rfm_iters, args.n_components, args.batch_size,
                          args.cache_dir, args.seed)
