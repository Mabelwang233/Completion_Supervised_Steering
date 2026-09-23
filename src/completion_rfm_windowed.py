"""
Generalizes completion_rfm.py to support truncating z to a window before
taking the last-token (x, z_window) embedding -- for the response-length
ablation (how much of the response does completion-RFM actually need?).

Window spec (pick one):
    --window_type fixed        --window_tokens N     (e.g. 15, 30, 60)
    --window_type anchor_frac  --window_frac F        (e.g. 0.10, 0.25, 0.50, 0.75)

T (completion length in tokens) and the truncation index are computed
EXACTLY the way extract_trajectories.py's subsample_anchors() computes
them for the anchor grid:

    prompt_len = len(tokenizer(formatted_prompt, add_special_tokens=False))
    full_len   = len(tokenizer(formatted_prompt + completion, add_special_tokens=False))
    T          = full_len - prompt_len
    tau        = idx / (T - 1)                      (extract_trajectories.py)

so --window_frac F truncates at the same token position that would carry
tau ~ F in Sequence-RFM's own anchor grid -- the two sweeps are directly
comparable.

NOT handled here (reuse existing artifacts instead of calling this):
  - window_tokens=0 or window_frac=0.0  -> identical to vanilla RFM
    (x only). Reuse baseline_directions.pkl's "rfm" entry.
  - window_type=fixed with window_tokens >= every completion's length,
    or window_frac=1.0                  -> identical to the original,
    unwindowed completion_rfm.py. Reuse completion_rfm_directions.pkl
    if you already trained it.

Truncated text is reconstructed via tokenizer.decode() on the sliced
input_ids (not tokenizer.tokenize/convert_tokens_to_string), so it uses
the SAME ids as the T/tau computation above. NeuralController re-
tokenizes the reconstructed string internally, which can differ from
the original ids by at most a token or so right at the truncation
boundary (normal BPE encode-decode-reencode looseness) -- this is the
same kind of seam extract_trajectories.py already tolerates by
tokenizing prompt and full text separately.
"""
import argparse
import json
import pickle
import sys

import torch

sys.path.insert(0, ".")
from utils import load_model
from neural_controllers import NeuralController


def compute_window_cutoff(prompt_len, full_ids_len, window_type, window_tokens=None, window_frac=None):
    """Returns the number of completion tokens to keep, in [0, T]."""
    T = full_ids_len - prompt_len
    if T <= 0:
        return 0
    if window_type == "fixed":
        return max(0, min(window_tokens, T))
    elif window_type == "anchor_frac":
        # matches extract_trajectories.subsample_anchors: tau = idx / (T - 1)
        idx = round(window_frac * (T - 1))
        idx = max(0, min(idx, T - 1))
        return idx + 1  # keep tokens [0 .. idx], i.e. inclusive of the anchor token
    else:
        raise ValueError(f"unknown window_type: {window_type}")


def build_windowed_texts(dataset_with_completions_path, tokenizer, window_type,
                          window_tokens=None, window_frac=None):
    with open(dataset_with_completions_path) as f:
        dataset = json.load(f)
    train_examples = dataset["train"]

    windowed_texts, labels, kept_lengths = [], [], []
    n_dropped = 0
    for ex in train_examples:
        formatted_prompt = ex["formatted_prompt"]
        completion = ex["completion"]
        full_text = formatted_prompt + completion

        prompt_ids = tokenizer(formatted_prompt, return_tensors="pt",
                                add_special_tokens=False).input_ids
        full_ids = tokenizer(full_text, return_tensors="pt",
                              add_special_tokens=False).input_ids
        prompt_len = prompt_ids.shape[1]
        full_len = full_ids.shape[1]

        cutoff = compute_window_cutoff(prompt_len, full_len, window_type, window_tokens, window_frac)
        if cutoff <= 0:
            n_dropped += 1
            continue

        completion_ids_trunc = full_ids[0, prompt_len:prompt_len + cutoff]
        truncated_completion = tokenizer.decode(completion_ids_trunc, skip_special_tokens=True)

        windowed_texts.append(formatted_prompt + truncated_completion)
        labels.append(ex["label"])
        kept_lengths.append(cutoff)

    if n_dropped:
        print(f"  WARNING: dropped {n_dropped}/{len(train_examples)} examples with a 0-token window "
              f"-- check for unexpectedly short completions.")
    avg_len = sum(kept_lengths) / max(len(kept_lengths), 1)
    window_desc = f"tokens={window_tokens}" if window_type == "fixed" else f"frac={window_frac}"
    print(f"  window={window_type} {window_desc}: {len(windowed_texts)} examples kept, "
          f"avg z length {avg_len:.1f} tokens")
    return windowed_texts, labels


def train_completion_rfm_windowed(dataset_with_completions_path, out_path, model_name,
                                   window_type, window_tokens=None, window_frac=None,
                                   rfm_iters=8, n_components=1, batch_size=8,
                                   cache_dir="/data/mabel/cache/MCL_rebuttal/Data_Models/hf_cache", seed=0,
                                   language_model=None, tokenizer=None):
    """
    language_model/tokenizer: pass an already-loaded model to reuse it
    across multiple window configs (strongly recommended -- see
    window_ablation_personas.py). If omitted, this function loads its
    own copy via utils.load_model (fp32 by default for raw HF model
    ids -- ~32GB for an 8B model) and does NOT free it afterward, so
    repeated standalone calls will accumulate GPU memory. Only rely on
    this fallback for one-off/CLI use, not inside a loop.
    """
    torch.manual_seed(seed)

    if language_model is None or tokenizer is None:
        language_model, tokenizer = load_model(model_name, cache_dir=cache_dir)

    full_texts, train_labels = build_windowed_texts(
        dataset_with_completions_path, tokenizer, window_type, window_tokens, window_frac)

    print(f"Training windowed completion-RFM on {len(full_texts)} (x, z_window) pairs, all layers...")
    controller = NeuralController(
        language_model, tokenizer,
        control_method="rfm",
        rfm_iters=rfm_iters,
        n_components=n_components,
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
        "window_type": window_type,
        "window_tokens": window_tokens,
        "window_frac": window_frac,
    }

    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_with_completions_path", required=True)
    p.add_argument("--out_path", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--window_type", required=True, choices=["fixed", "anchor_frac"])
    p.add_argument("--window_tokens", type=int, default=None,
                    help="required if --window_type fixed")
    p.add_argument("--window_frac", type=float, default=None,
                    help="required if --window_type anchor_frac, in [0, 1]")
    p.add_argument("--rfm_iters", type=int, default=8)
    p.add_argument("--n_components", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--cache_dir", default="/data/mabel/cache/MCL_rebuttal/Data_Models/hf_cache")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.window_type == "fixed" and args.window_tokens is None:
        p.error("--window_tokens is required when --window_type fixed")
    if args.window_type == "anchor_frac" and args.window_frac is None:
        p.error("--window_frac is required when --window_type anchor_frac")

    train_completion_rfm_windowed(
        args.dataset_with_completions_path, args.out_path, args.model,
        args.window_type, args.window_tokens, args.window_frac,
        args.rfm_iters, args.n_components, args.batch_size,
        args.cache_dir, args.seed)