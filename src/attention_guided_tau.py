"""
Attention-guided window selection, adapted from Davarmanesh et al.
"Efficient and accurate steering of LLMs through attention-guided
feature learning" (arXiv:2602.00333) Eq. 3, extended from their fixed
4-candidate-token setting to your completion tokens z.

WHAT THIS COMPUTES, PER LABEL=1 (Pc) TRAINING EXAMPLE, PER LAYER:

    tau*_{p,l} = argmax_i [ sum_{j in prefix span} A^(l)_{i,j}(x_p, z_p) ] / (T_p - 1)

i.e. the normalized position (matching extract_trajectories.py's tau
convention exactly) of the completion token that pays the most
attention back to the concept-injecting prefix span in the prompt
(e.g. "Personify Sun Tzu." for personas), at that layer.

LABEL=0 (P0) EXAMPLES ARE NOT SCORED HERE. There is no prefix span in
a label=0 prompt (persona_data_prep.py's class_0 statements never get
the CLASS_1_PREFIX_TEMPLATE prepended), so "attention to prefix" is
undefined for them -- this mirrors the paper's own method, where P0
never gets a soft label either (Eq. 4 hard-sets y_p=0 for p in P0).
Per-layer MEDIANS over the label=1 tau*_{p,l} values are computed here
so a downstream script can apply the borrowed value to label=0
examples (see window_ablation.py's anchor_frac path, which already
takes a per-layer window_frac).

DOMAIN CONFIG: both "persona" and "fear" are filled in below, each
confirmed against that domain's own data-prep script's
CLASS_1_PREFIX_TEMPLATE -- persona from persona_data_prep.py
("Personify {persona}."), fear from fear_data_prep.py ("Personify
someone who is terrified of {fear}.").

ATTENTION AGGREGATION ACROSS HEADS: the paper's Eq. 3 attention matrix
A^(l) is written head-agnostically ("we have also omitted the notion
of attention heads... for simplicity" -- Preliminaries). In practice
this means averaging attention across heads at each layer before
scoring, which is what this script does (output_attentions gives
per-head matrices; we mean over the head dimension). This is a
reasonable reading of the paper but not something stated as an
explicit implementation detail there -- flagging as an assumption.
"""
import argparse
import json
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, ".")
from utils import load_model

DOMAIN_PREFIX_CONFIG = {
    "persona": {
        "concept_key": "persona",
        "prefix_template": "Personify {concept}.",
    },
    "fear": {
        "concept_key": "fear",
        "prefix_template": "Personify someone who is terrified of {concept}.",
    },
}


def get_prefix_text(domain, dataset):
    cfg = DOMAIN_PREFIX_CONFIG.get(domain)
    if cfg is None:
        raise NotImplementedError(
            f"domain={domain!r} has no confirmed prefix template in DOMAIN_PREFIX_CONFIG -- "
            f"upload the data-prep script for this domain and fill it in before running."
        )
    concept = dataset[cfg["concept_key"]]
    return cfg["prefix_template"].format(concept=concept)


def compute_tau_star_for_example(language_model, tokenizer, formatted_prompt, completion,
                                  prefix_text, layers):
    """
    Returns {layer: tau_star} for ONE label=1 example, across the given layers.

    prompt_len / full_len / T follow extract_trajectories.py's exact
    convention (separately-tokenized prompt length, jointly-tokenized
    full length) so tau here is directly comparable to Sequence-RFM's
    own anchor tau grid and to completion_rfm_windowed.py's anchor_frac windows.
    """
    full_text = formatted_prompt + completion

    prompt_ids = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(language_model.device)
    prompt_len = prompt_ids.shape[1]
    T = full_ids.input_ids.shape[1] - prompt_len
    if T <= 1:
        return None  # degenerate completion, nothing to score

    # Locate the prefix span WITHIN the prompt (not the full x+z sequence).
    # formatted_prompt already wraps the raw prompt in the chat template;
    # prefix_text is the raw, un-wrapped concept-injecting phrase
    # (e.g. "Personify Sun Tzu."), which must appear verbatim at the start
    # of the raw prompt per persona_data_prep.py's construction. We locate
    # it by tokenizing the prefix alone and taking its length -- same
    # "tokenize a substring separately" convention already used for
    # prompt_len elsewhere in this codebase, with the same small caveat
    # about BPE merge effects right at the boundary.
    prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids
    prefix_len = prefix_ids.shape[1]
    if prefix_len >= prompt_len:
        return None  # prefix span computation went wrong for this example -- skip rather than guess

    with torch.no_grad():
        outputs = language_model(**full_ids, output_hidden_states=False, output_attentions=True)

    if outputs.attentions is None:
        raise RuntimeError(
            "output_attentions=True returned None -- the model's attention backend (SDPA/Flash "
            "Attention) doesn't materialize attention weights. Call "
            "language_model.set_attn_implementation('eager') before running this, and switch back "
            "afterward if you want faster generation for the rest of the pipeline."
        )

    n_attn_layers = len(outputs.attentions)  # NOTE: no embedding-layer entry, unlike hidden_states
    tau_star_per_layer = {}
    for layer in layers:
        # attentions has n_attn_layers entries (one per decoder layer, indexed from the front),
        # NOT n_attn_layers+1 like hidden_states (which also includes the embedding layer at index
        # 0) -- so this offset is deliberately different from extract_trajectories.py's resolve_layers.
        idx = layer if layer >= 0 else n_attn_layers + layer
        # attentions[idx]: (1, n_heads, T_full, T_full) -- mean over heads
        attn = outputs.attentions[idx][0].mean(dim=0)  # (T_full, T_full)

        # rows = completion token positions (queries), cols = prefix span (keys)
        completion_rows = attn[prompt_len:, :]  # (T, T_full)
        prefix_scores = completion_rows[:, :prefix_len].sum(dim=1)  # (T,) -- sum over prefix key positions

        i_star = int(torch.argmax(prefix_scores).item())
        tau_star = i_star / max(T - 1, 1)
        tau_star_per_layer[layer] = tau_star

    return tau_star_per_layer


def compute_tau_star_dataset(dataset_with_completions_path, domain, model_name, out_path,
                              layers=None, cache_dir="/data/mabel/cache/MCL_rebuttal/Data_Models/hf_cache",
                              language_model=None, tokenizer=None):
    with open(dataset_with_completions_path) as f:
        dataset = json.load(f)
    prefix_text = get_prefix_text(domain, dataset)
    print(f"Prefix text for this concept: {prefix_text!r}")

    if language_model is None:
        language_model, tokenizer = load_model(model_name, cache_dir=cache_dir)
    language_model.eval()

    if layers is None:
        num_layers = len(language_model.model.layers)
        layers = list(range(-1, -num_layers, -1))  # matches extract_trajectories.py's default

    # output_attentions=True silently returns None under SDPA/Flash Attention backends -- eager is
    # required to actually materialize attention weights. Switch for this scoring pass only, and
    # restore whatever was there before (so training/generation elsewhere keep using the faster
    # backend). set_attn_implementation was added in transformers 4.48 -- if it's missing, fail
    # loudly rather than silently scoring garbage.
    original_attn_impl = getattr(language_model.config, "_attn_implementation", None)
    if not hasattr(language_model, "set_attn_implementation"):
        raise RuntimeError(
            "language_model.set_attn_implementation is not available (needs transformers>=4.48). "
            "Reload the model with attn_implementation='eager' explicitly before calling this."
        )
    language_model.set_attn_implementation("eager")

    label_1_examples = [ex for ex in dataset["train"] if ex["label"] == 1]
    print(f"Scoring {len(label_1_examples)} label=1 examples across {len(layers)} layers...")

    per_example_results = []
    n_skipped = 0
    try:
        for ex in tqdm(label_1_examples):
            result = compute_tau_star_for_example(
                language_model, tokenizer, ex["formatted_prompt"], ex["completion"], prefix_text, layers)
            if result is None:
                n_skipped += 1
                continue
            per_example_results.append({"statement": ex.get("statement"), "tau_star_per_layer": result})
    finally:
        # restore original attention backend regardless of success/failure above
        if original_attn_impl:
            language_model.set_attn_implementation(original_attn_impl)

    if n_skipped:
        print(f"  skipped {n_skipped}/{len(label_1_examples)} examples (degenerate completion or prefix-span issue)")

    # per-layer median across label=1 examples, for label=0 to borrow downstream
    median_per_layer = {}
    for layer in layers:
        vals = sorted(r["tau_star_per_layer"][layer] for r in per_example_results)
        n = len(vals)
        median_per_layer[layer] = vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) / 2

    out = {
        "domain": domain,
        "concept": dataset.get(DOMAIN_PREFIX_CONFIG[domain]["concept_key"]),
        "prefix_text": prefix_text,
        "layers": layers,
        "per_example": per_example_results,  # label=1 only, each example's own tau*_{p,l}
        "median_per_layer": median_per_layer,  # for label=0 to borrow
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    print(f"Median tau* per layer (sample): { {k: round(v, 3) for k, v in list(median_per_layer.items())[:5]} } ...")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_with_completions_path", required=True)
    p.add_argument("--domain", required=True, choices=list(DOMAIN_PREFIX_CONFIG.keys()))
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--out_path", required=True)
    p.add_argument("--layers", nargs="+", default=None, type=int)
    p.add_argument("--cache_dir", default="/data/mabel/cache/MCL_rebuttal/Data_Models/hf_cache")
    args = p.parse_args()

    compute_tau_star_dataset(args.dataset_with_completions_path, args.domain, args.model,
                              args.out_path, args.layers, args.cache_dir)
