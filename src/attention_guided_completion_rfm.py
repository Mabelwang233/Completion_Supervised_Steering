"""
Attention-guided completion-RFM: per-layer, per-example z-window
truncation driven by tau*_{p,l} from attention_guided_tau.py, instead
of a single global fraction/token-count shared across all layers.

Requires attention_guided_tau.py to have already been run for this
concept (produces tau_star.json: per-example tau*_{p,l} for label=1,
plus a per-layer median for label=0 to borrow -- see that file's
docstring for why label=0 has no tau* of its own).

COST NOTE: NeuralController.compute_directions trains ALL layers from
ONE forward pass when given a single text per example -- fine when
every layer shares the same truncation, but incompatible with a
genuinely different truncation per layer, since the texts themselves
differ by layer here. So this calls compute_directions once PER LAYER,
each with that layer's own tau*-truncated texts -- L forward passes
total instead of 1. (Note: direction_utils.get_hidden_states always
extracts every layer's hidden state in a single forward pass regardless
of what's requested -- there's no way to ask it for just one layer and
save compute that way. The L-forward-pass cost here is coming entirely
from needing L different truncated-text inputs, not from anything
about layer restriction.) Restrict --layers to a middle band (default
below) rather than requesting all ~31 layers, or this gets expensive.
"""
import argparse
import json
import pickle
import sys

import torch

sys.path.insert(0, ".")
from utils import load_model
from neural_controllers import NeuralController
from completion_rfm_windowed import compute_window_cutoff

# Pragmatic middle band in NeuralController's own negative-index convention
# (-1 = last layer, extract_trajectories.py's resolve_layers: forward_idx =
# num_layers + layer + 1). NOT a literal translation of the paper's
# forward-numbered "blocks 5-20" (their Fig 2C/4B) -- the two indexing
# conventions don't map 1:1 and getting that exact isn't necessary for a
# first pass. Override via --layers for a different/wider band.
ATTENTION_LAYERS_DEFAULT = list(range(-8, -25, -1))  # -8 .. -24, 17 layers


def load_tau_star(tau_star_path):
    with open(tau_star_path) as f:
        return json.load(f)


def build_texts_for_layer(dataset_with_completions_path, tokenizer, tau_star_data, layer):
    with open(dataset_with_completions_path) as f:
        dataset = json.load(f)
    train_examples = dataset["train"]

    # JSON round-trips dict keys to strings -- match that on lookup.
    per_example_tau = {r["statement"]: r["tau_star_per_layer"][str(layer)]
                        for r in tau_star_data["per_example"]}
    median_tau = tau_star_data["median_per_layer"][str(layer)]

    texts, labels = [], []
    n_label1_own, n_label0_borrowed, n_dropped = 0, 0, 0
    for ex in train_examples:
        formatted_prompt = ex["formatted_prompt"]
        completion = ex["completion"]
        full_text = formatted_prompt + completion

        prompt_ids = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).input_ids
        full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids
        prompt_len = prompt_ids.shape[1]
        full_len = full_ids.shape[1]

        if ex["label"] == 1 and ex.get("statement") in per_example_tau:
            tau = per_example_tau[ex["statement"]]
            n_label1_own += 1
        else:
            tau = median_tau  # label=0, or a label=1 example that was skipped during tau* scoring
            n_label0_borrowed += 1

        cutoff = compute_window_cutoff(prompt_len, full_len, "anchor_frac", window_frac=tau)
        if cutoff <= 0:
            n_dropped += 1
            continue

        completion_ids_trunc = full_ids[0, prompt_len:prompt_len + cutoff]
        truncated_completion = tokenizer.decode(completion_ids_trunc, skip_special_tokens=True)
        texts.append(formatted_prompt + truncated_completion)
        labels.append(ex["label"])

    print(f"  layer {layer}: {len(texts)} examples kept "
          f"({n_label1_own} label=1 using own tau*, {n_label0_borrowed} using borrowed median tau*={median_tau:.3f}"
          f"{f', {n_dropped} dropped' if n_dropped else ''})")
    return texts, labels


def train_attention_guided_completion_rfm(dataset_with_completions_path, tau_star_path, out_path,
                                           model_name, layers=None, rfm_iters=8, n_components=1,
                                           batch_size=8, cache_dir="/data/mabel/cache/MCL_rebuttal/Data_Models/hf_cache",
                                           seed=0, language_model=None, tokenizer=None):
    torch.manual_seed(seed)
    if layers is None:
        layers = ATTENTION_LAYERS_DEFAULT

    if language_model is None or tokenizer is None:
        language_model, tokenizer = load_model(model_name, cache_dir=cache_dir)

    tau_star_data = load_tau_star(tau_star_path)

    per_layer_direction = {}
    for layer in layers:
        texts, labels = build_texts_for_layer(dataset_with_completions_path, tokenizer, tau_star_data, layer)

        controller = NeuralController(
            language_model, tokenizer,
            control_method="rfm",
            rfm_iters=rfm_iters,
            n_components=n_components,
            batch_size=batch_size,
        )
        # direction_utils.get_hidden_states unconditionally extracts EVERY layer in one forward
        # pass (range(-1, -num_layers, -1)), regardless of what hidden_layers is passed -- that
        # argument only pre-allocates dict keys, it doesn't restrict computation. So requesting a
        # short list bought no compute savings and caused a KeyError the moment the loop hit a
        # layer we hadn't pre-allocated. Just let it default to the full range (matches
        # NeuralController's own __init__ default) and pull out only the layer we want afterward.
        controller.compute_directions(texts, labels)
        per_layer_direction[layer] = controller.directions[layer][0].detach().cpu()

    result = {
        "directions_per_layer": per_layer_direction,
        "layers": layers,
        "method": "attention_guided_completion_rfm",
        "tau_star_path": tau_star_path,
    }
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_with_completions_path", required=True)
    p.add_argument("--tau_star_path", required=True)
    p.add_argument("--out_path", required=True)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--layers", nargs="+", type=int, default=None)
    p.add_argument("--rfm_iters", type=int, default=8)
    p.add_argument("--n_components", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--cache_dir", default="/data/mabel/cache/MCL_rebuttal/Data_Models/hf_cache")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    train_attention_guided_completion_rfm(
        args.dataset_with_completions_path, args.tau_star_path, args.out_path, args.model,
        args.layers, args.rfm_iters, args.n_components, args.batch_size, args.cache_dir, args.seed)
