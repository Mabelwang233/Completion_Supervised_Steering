"""
CLAS (Contextual Linear Activation Steering), Hsu et al. 2026.

Genuinely different from every other method in this project: all of
mean_difference, rfm, completion_rfm, completion_diff_means,
bag_of_words_rfm, bag_of_words_diff_means, sequence_rfm are CLOSED-FORM
probes (kernel ridge, diff-means, AGOP) -- none of them backprop through
the LLM. CLAS does: it freezes the whole model and trains a small
per-layer "sensing vector" c_ell via gradient descent on next-token
prediction loss, to make the steering coefficient CONTEXT-DEPENDENT
instead of a single global alpha:

    h~_l,t = h_l,t + (c_l . [h_l,t; 1]) d_l

d_l is NOT trained here -- it's the existing prompt-only RFM direction
(same as our "rfm" baseline), extracted exactly like LAS. Only c_l is
learned, via a differentiable version of our steering hook.

Steer dataset (for training c_l): the paper pairs a NON-task prompt with
a TASK-exhibiting completion -- i.e. teaching the sensing vector to fire
even on an unprompted input, since that's the whole point of steering.
Mapped onto our data: x = a label-0 (plain) prompt, z = a label-1
(concept-exhibiting) completion, cross-paired by index.

DATASET SIZE / SPLIT (CLAS-aligned with completion_rfm + controller):
build_steer_pairs now uses ALL available cross-paired examples (capped
by whichever label has fewer), split val_frac/1-val_frac instead of a
fixed n_train=n_val=10 -- so if your dataset_with_completions.json has
~200 of each label (matching the completion-RFM+controller pool), CLAS
trains on the SAME total amount of data, for a fair comparison, rather
than the paper's original tiny 10/10 scale. See train_clas's docstring
for the training-loop restructuring this required.
"""
import json
import pickle
import random
import sys

import torch

sys.path.insert(0, ".")
from utils import load_model


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


def build_steer_pairs(dataset_with_completions_path, val_frac=0.2, seed=0):
    """Cross-pairs (neutral_prompt, concept_completion) for all available
    training examples.

    Supports two dataset formats:

    1. New format (neutral_prompt / concept_completion fields): uses ALL
       examples directly -- every example has both fields stored under the
       same statement, so there is no mismatch. Gives 400 pairs for 400
       statements.

    2. Old format (label=0/1, same statements): label=0 has neutral prompts,
       label=1 has concept completions. Sorted by statement before pairing
       to guarantee alignment regardless of shuffle order in the dataset.
       Gives min(len(label0), len(label1)) pairs.

    In both cases: x = neutral prompt (no concept cue),
                   z = concept-exhibiting completion.
    """
    with open(dataset_with_completions_path) as f:
        dataset = json.load(f)
    train_examples = dataset.get("train", [])

    # Detect format by checking first example's keys
    if train_examples and "neutral_prompt" in train_examples[0]:
        # New format: every example has neutral_prompt + concept_completion
        pairs = []
        for ex in train_examples:
            prompt     = ex.get("neutral_prompt", "")
            completion = ex.get("concept_completion", "")
            if prompt.strip() and completion.strip():
                pairs.append({"prompt": prompt, "completion": completion})
        print(f"  [build_steer_pairs] new format: {len(pairs)} pairs "
              f"(neutral_prompt -> concept_completion).")
    else:
        label0 = sorted(
            [ex for ex in train_examples if ex["label"] == 0],
            key=lambda x: x["statement"]
        )
        label1 = sorted(
            [ex for ex in train_examples if ex["label"] == 1],
            key=lambda x: x["statement"]
        )
        n_pairs = min(len(label0), len(label1))
        if n_pairs == 0:
            raise ValueError(f"Need at least 1 example of each label, "
                             f"got {len(label0)} label-0 and {len(label1)} label-1.")

        # Check whether label-0 and label-1 share the same statements
        mismatches = sum(1 for a, b in zip(label0, label1)
                         if a["statement"] != b["statement"])

        if mismatches == 0:
            # Old format: same statements across both labels -- pair by index
            pairs = [{"prompt": label0[i]["prompt"], "completion": label1[i]["completion"]}
                     for i in range(n_pairs)]
            print(f"  [build_steer_pairs] old format: {len(pairs)} pairs "
                  f"(label-0 neutral prompt -> label-1 concept completion, sorted by statement).")
        else:
            # Split-statement format (e.g. topophile): label-0 and label-1 use
            # different statements, so cross-pairing by index is impossible.
            # Mirror train_controller.py: take label-1 examples only, reconstruct
            # the unprimed prompt from ex["statement"] (stripping the concept
            # prefix), and pair with ex["completion"] (generated from the primed
            # prompt). x = plain statement prompt, z = concept-exhibiting completion.
            pairs = []
            for ex in label1:
                stmt = ex.get("statement", "")
                completion = ex.get("completion", "")
                if not stmt.strip() or not completion.strip():
                    continue
                unprimed = (f"What are your thoughts on the following statement?\n"
                            f"Statement: {stmt}")
                pairs.append({"prompt": unprimed, "completion": completion})
            print(f"  [build_steer_pairs] split-statement format: {len(pairs)} pairs "
                  f"(unprimed statement prompt -> label-1 concept completion, "
                  f"mirroring train_controller.py).")

    if not pairs:
        raise ValueError("No valid pairs found. Check that completions are generated.")

    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_frac)))
    val_pairs   = shuffled[:n_val]
    train_pairs = shuffled[n_val:]
    return train_pairs, val_pairs


def register_clas_train_hooks(language_model, d_directions, c_weight, c_bias):
    """Differentiable hooks -- c_weight/c_bias are looked up FRESH from the
    outer dict on every forward call (not captured as frozen default args),
    so autograd sees the CURRENT parameter values at each training step.

    IMPORTANT: unlike generation_utils.hook_model (only ever used inside
    .generate(), where KV-caching means a decoder layer's output is always
    a tuple), this hook also runs during a PLAIN forward pass
    (compute_example_loss below) -- in that mode a decoder layer can
    return a bare tensor instead of a tuple. Checking isinstance() BEFORE
    indexing (not after) is required, or output[0] silently indexes into
    the batch dimension instead of extracting hidden_states, corrupting
    the tensor rank and causing a shape-mismatch error several layers
    downstream (inside attention/RoPE, not here -- misleading traceback).
    """
    hooks = {}
    for layer in d_directions:
        d_vec = d_directions[layer]

        def block_hook(module, input, output, layer=layer, d_vec=d_vec):
            is_tuple = isinstance(output, tuple)
            hidden_states = output[0] if is_tuple else output

            c_vec = torch.cat([c_weight[layer], c_bias[layer]]).to(hidden_states.dtype)
            ones = torch.ones(*hidden_states.shape[:-1], 1, device=hidden_states.device, dtype=hidden_states.dtype)
            h_aug = torch.cat([hidden_states, ones], dim=-1)
            alpha = torch.einsum('...i,i->...', h_aug, c_vec)  # per-token, context-dependent coefficient
            new_hidden = hidden_states + alpha.unsqueeze(-1) * d_vec.to(dtype=hidden_states.dtype, device=hidden_states.device)

            if is_tuple:
                return (new_hidden,) + output[1:]
            return new_hidden

        block = language_model.model.layers[layer]
        hooks[layer] = block.register_forward_hook(block_hook)
    return hooks


def clear_hooks(hooks):
    for h in hooks.values():
        h.remove()


def compute_example_loss(language_model, tokenizer, prompt_text, completion_text):
    """Next-token prediction loss, computed ONLY on completion tokens
    (prompt positions masked with -100). HF's labels= handles the
    shift-by-one internally."""
    formatted_prompt = format_chat(tokenizer, prompt_text)
    full_text = formatted_prompt + completion_text

    prompt_ids = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids.to(language_model.device)
    prompt_len = prompt_ids.shape[1]

    labels = full_ids.clone()
    labels[:, :prompt_len] = -100

    outputs = language_model(input_ids=full_ids, labels=labels, use_cache=False)
    return outputs.loss


def train_clas(dataset_with_completions_path, baseline_path, out_path, model_name,
                val_frac=0.2, lr_weight=3e-3, lr_bias=1e-1, epochs=8, accum_steps=10,
                patience=2, cache_dir=None, seed=0):
    """
    CLAS-aligned schedule, generalized beyond the paper's own tiny
    (n=10) scale: AdamW, effective batch size `accum_steps` (batch 1 x
    accum_steps grad-accumulation steps -- FIXED regardless of dataset
    size, unlike the previous version where "one step" silently meant
    "accumulate over the entire train set", which only matched the
    paper's spec because n_train was hardcoded to exactly 10).

    `epochs` (default 8) is epochs, not a literal copy of the paper's
    "8 update steps" -- see train_controller.py's module docstring for
    the same reasoning: the paper's "8 update steps" only equals "8
    epochs" because their train set size matched their effective batch
    size (10 ~= 10). For a larger train set this script now generalizes
    to, one epoch = ceil(n_train / accum_steps) update steps.

    Early stopping: training stops once val loss hasn't improved for
    `patience` consecutive epochs; the best-val-loss checkpoint is kept
    regardless of when/whether it triggers -- same convention as
    train_controller.py's train_one_controller.
    """
    torch.manual_seed(seed)

    train_pairs, val_pairs = build_steer_pairs(dataset_with_completions_path, val_frac, seed)
    print(f"Steer dataset: {len(train_pairs)} train / {len(val_pairs)} val pairs "
          f"(x = plain prompt, z = concept-exhibiting completion, cross-paired).")

    with open(baseline_path, "rb") as f:
        baselines = pickle.load(f)
    # Keep directions in their stored dtype (float32 from NeuralController's CPU math).
    # The hooks cast to model dtype (.to(hidden_states.dtype)) at apply time, so no
    # pre-cast is needed here -- and forcing .float() would break if we ever saved
    # directions in bfloat16.
    d_directions_cpu = {layer: vec for layer, vec in baselines["rfm"]["directions_per_layer"].items()}
    layers = list(d_directions_cpu.keys())
    d = next(iter(d_directions_cpu.values())).shape[0]
    print(f"d_l = existing vanilla RFM baseline direction, {len(layers)} layers, d={d}.")

    language_model, tokenizer = load_model(model_name, cache_dir=cache_dir)
    language_model.eval()
    for p in language_model.parameters():
        p.requires_grad_(False)  # freeze the whole LLM -- only c_l gets gradients

    device = language_model.device
    d_directions = {layer: vec.to(device) for layer, vec in d_directions_cpu.items()}

    c_weight = {layer: torch.zeros(d, device=device, requires_grad=True) for layer in layers}
    c_bias = {layer: torch.zeros(1, device=device, requires_grad=True) for layer in layers}

    optimizer = torch.optim.AdamW([
        {"params": list(c_weight.values()), "lr": lr_weight},
        {"params": list(c_bias.values()), "lr": lr_bias},
    ])

    hooks = register_clas_train_hooks(language_model, d_directions, c_weight, c_bias)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = None
    epochs_since_improvement = 0
    total_update_steps = 0

    def _val_loss():
        with torch.no_grad():
            return sum(
                compute_example_loss(language_model, tokenizer, ex["prompt"], ex["completion"]).item()
                for ex in val_pairs
            ) / len(val_pairs)

    try:
        for epoch in range(epochs):
            epoch_train_losses = []
            optimizer.zero_grad()
            accum_count = 0

            for ex in train_pairs:
                loss = compute_example_loss(language_model, tokenizer, ex["prompt"], ex["completion"])
                (loss / accum_steps).backward()
                accum_count += 1
                epoch_train_losses.append(loss.item())

                if accum_count == accum_steps:
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_count = 0
                    total_update_steps += 1

            if accum_count > 0:  # flush a final partial accumulation window
                optimizer.step()
                optimizer.zero_grad()
                total_update_steps += 1

            train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            val_loss = _val_loss()
            is_best = val_loss < best_val_loss
            print(f"  epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"update_steps={total_update_steps}{'  <- best so far' if is_best else ''}")

            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_state = {
                    layer: (c_weight[layer].detach().clone(), c_bias[layer].detach().clone())
                    for layer in layers
                }
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= patience:
                    print(f"\nEarly stopping: val_loss hasn't improved for {patience} "
                          f"consecutive epochs (best was epoch {best_epoch}).")
                    break
    finally:
        clear_hooks(hooks)

    print(f"Best val_loss={best_val_loss:.4f} at epoch {best_epoch} "
          f"({total_update_steps} total update steps)")

    result = {
        "d_directions_per_layer": {layer: d_directions_cpu[layer].cpu() for layer in layers},
        "c_weight_per_layer": {layer: best_state[layer][0].cpu() for layer in layers},
        "c_bias_per_layer": {layer: best_state[layer][1].cpu() for layer in layers},
        "layers": layers,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }
    torch.save(result, out_path)
    print(f"Wrote {out_path}")


def generate_with_clas(language_model, tokenizer, prompt_text, d_directions, c_weight, c_bias,
                        max_new_tokens=100):
    """Inference-time application of the LEARNED (fixed) sensing vectors --
    same hook math as training, but no gradient tracking needed."""
    formatted = format_chat(tokenizer, prompt_text)
    inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to(language_model.device)

    hooks = {}
    for layer in d_directions:
        c_vec = torch.cat([c_weight[layer], c_bias[layer]]).to(language_model.device)
        d_vec = d_directions[layer].to(language_model.device)

        def block_hook(module, input, output, c_vec=c_vec, d_vec=d_vec):
            is_tuple = isinstance(output, tuple)
            hidden_states = output[0] if is_tuple else output
            ones = torch.ones(*hidden_states.shape[:-1], 1, device=hidden_states.device, dtype=hidden_states.dtype)
            h_aug = torch.cat([hidden_states, ones], dim=-1)
            alpha = torch.einsum('...i,i->...', h_aug, c_vec.to(hidden_states.dtype))
            new_hidden = hidden_states + alpha.unsqueeze(-1) * d_vec.to(hidden_states.dtype)
            if is_tuple:
                return (new_hidden,) + output[1:]
            return new_hidden

        block = language_model.model.layers[layer]
        hooks[layer] = block.register_forward_hook(block_hook)

    try:
        with torch.no_grad():
            output_ids = language_model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        clear_hooks(hooks)

    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)