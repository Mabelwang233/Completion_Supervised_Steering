"""
Trains a SequenceRFMController (controller.py) for one persona concept
and one fixed k, via the pdf's Section 6 log-likelihood objective:

    min_C  sum_i sum_t  -log p_theta(z_{i,t} | x_i, z_{i,<t}, y_i; D, C)

replacing your current grid-searched global alpha entirely -- there is
no alpha_grid argument anywhere in this file. D_ell (frozen) comes from
a sequence_rfm_directions.pt already trained for this k via
train_sequence_rfm.py.

*** CORRECTED INPUT/TARGET PAIRING (see the "bad results" discussion) ***
Teacher forcing here uses the UNPRIMED question (no "Personify X."
prefix -- reconstructed from ex["statement"] via the same
QUESTION_TEMPLATE persona_data_prep.py/fear_data_prep.py use for
label==0 examples) as the INPUT, paired with ex["completion"] (which
was generated from the PRIMED prompt) as the TARGET. An earlier version
of this file used the PRIMED prompt (ex["formatted_prompt"]) as the
input -- but the unsteered model, given that same primed prompt,
already assigns near-maximal likelihood to its own greedy completion,
so that objective had almost no gradient signal pushing the controller
to learn any nontrivial steering (confirmed: a real checkpoint's
history showed train_loss already at 0.26 by epoch 1, then falling
while val_loss got WORSE -- overfitting on a task that was already
solved at init). Using the unprimed prompt as input forces the
controller to actually bridge the primed/unprimed gap via steering,
matching both (a) what happens at eval time (steering is applied to
the raw, unprimed test question) and (b) the PSR paper's own training
setup (unsteered x paired with the response elicited by a steered x').

TRAINING SCHEDULE (CLAS-aligned): AdamW, effective batch size 10
(--batch_size 1 with --accum_steps 10 gradient-accumulation steps),
lr=3e-3 for the main controller weights, a SEPARATE lr=--bias_lr
(default 1e-1, matching the paper's Llama setting) for any parameter
whose name contains "bias" -- see train_one_controller's docstring for
the caveat on how that split is detected.

--epochs default is 8, but read this as EPOCHS, not a literal copy of
the paper's "8 update steps": CLAS's own steer sets are ~10 examples,
matching their effective batch size, so one of THEIR update steps is
already a full pass over their training set -- their "8 update steps"
IS 8 epochs, for their dataset size. For a training set of n examples,
one epoch = ceil(n / 10) update steps; e.g. 160 train examples (200
total, 20% val) -> 16 steps/epoch -> 128 total updates over 8 epochs.
Treating "8" as a literal global step cap would train on far less data
than CLAS actually did whenever your training set is larger than 10.

Early stopping is now real (not just best-checkpoint-keeping): training
stops once val loss hasn't improved for --patience consecutive epochs
(default 2), in addition to always keeping the best-val-loss checkpoint
regardless of when training stops.

TRAINING DATA / SCHEMA ASSUMPTION -- please check this against your
actual dataset_with_completions.json (from generate_completions.py):
this script expects a JSON with a top-level "train" list of examples,
each having at least

    {"prompt": "<the PRIMED input prompt actually sent to the model>",
     "statement": "<the raw statement, no persona/fear prefix>",
     "completion": "<z_i, the generated completion text>",
     "label": 0 or 1}

and trains the controller on label==1 examples only (the positive,
persona/fear-eliciting completions) -- matching both (a) the
"sufficient to train on positive examples only" convention from the
PSR paper, and (b) the fact that D_ell/M_ell were themselves fit
contrasting label-1 vs label-0 trajectories, so label-1 completions are
what a successful intervention should reproduce. If your actual field
names differ, `_extract_examples` and `build_teacher_forced_batch`
below are where to change them.
"""
import argparse
import copy
import json
import os
import random

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import load_model
from controller import SequenceRFMController, load_directions_for_k, seqrfm_method_label

# Llama-3 uses a special assistant-header token; Qwen uses <|im_start|>assistant.
# We derive the correct tag at runtime from the tokenizer rather than hard-coding
# one family, so the same code works for both models.
LLAMA_ASSISTANT_TAG = '<|start_header_id|>assistant<|end_header_id|>'
QWEN_ASSISTANT_TAG  = '<|im_start|>assistant\n'


def get_assistant_tag(tokenizer):
    """Return the model-appropriate assistant tag based on the tokenizer vocabulary."""
    vocab = tokenizer.get_vocab()
    if '<|start_header_id|>' in vocab:
        return LLAMA_ASSISTANT_TAG
    # Qwen2 / Qwen2.5 use <|im_start|>
    if '<|im_start|>' in vocab:
        return QWEN_ASSISTANT_TAG
    # Fallback: apply_chat_template handles everything; no extra tag needed.
    return ''


# Must match persona_data_prep.py / fear_data_prep.py's QUESTION_TEMPLATE
# exactly -- this is what reconstructs the UNPRIMED version of a label==1
# training example's question (i.e. what that same statement's prompt
# would have looked like as a label==0 example, with no persona/fear
# prefix). If you ever change QUESTION_TEMPLATE in either data-prep
# file, update it here too.
QUESTION_TEMPLATE = "What are your thoughts on the following statement?\nStatement: {statement}"

_warned_missing_statement = False


def format_chat(tokenizer, prompt_text):
    chat = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(chat, tokenize=False) + get_assistant_tag(tokenizer)


def _extract_examples(dataset_with_completions_path, label_filter=1):
    with open(dataset_with_completions_path) as f:
        data = json.load(f)
    examples = [ex for ex in data["train"] if ex.get("label") == label_filter]
    if not examples:
        raise ValueError(
            f"No label=={label_filter} training examples found in "
            f"{dataset_with_completions_path}. Check the schema assumption "
            f"in this file's docstring against your actual completions file."
        )
    return examples


def train_val_split(examples, val_frac=0.2, seed=0):
    """Deterministic shuffle + split. val_frac=0.2 -> ~80/20 train/val,
    e.g. 200 label==1 examples -> ~160 train / ~40 val."""
    rng = random.Random(seed)
    shuffled = examples[:]  # don't mutate caller's list
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_frac)))
    val_examples = shuffled[:n_val]
    train_examples = shuffled[n_val:]
    return train_examples, val_examples


def build_teacher_forced_batch(tokenizer, ex, device, train_prompt_suffix=None):
    """Tokenizes [unprimed_question; completion] as one sequence and
    returns (input_ids, loss_mask) where loss_mask is 1 exactly on
    completion-token positions (what the pdf's sum over t targets), 0
    on the prompt part.

    INPUT is the UNPRIMED question (reconstructed from ex["statement"]
    via QUESTION_TEMPLATE -- no "Personify X." prefix), NOT
    ex["formatted_prompt"]/ex["prompt"] (which are the PRIMED versions
    that generated ex["completion"] in the first place). See the module
    docstring for why -- using the primed prompt as input gave the
    controller almost nothing to learn, since the unsteered model
    already nearly reproduces its own greedy completion when given the
    same primed prompt it was generated from.

    train_prompt_suffix: pass the SAME suffix (if any) you used in
    generate_completions.py's --train_prompt_suffix, so the unprimed
    input matches the primed one in every respect except the persona/
    fear prefix itself. Leave None for the long_subsampled_anchor
    strategy (no suffix used).

    Falls back to ex["formatted_prompt"]/ex["prompt"] (the OLD, buggy
    primed-input behavior) with a one-time warning if ex["statement"]
    is missing -- e.g. an older dataset_with_completions.json that
    predates this fix.
    """
    global _warned_missing_statement
    if "statement" in ex:
        question_text = QUESTION_TEMPLATE.format(statement=ex["statement"])
        if train_prompt_suffix:
            question_text = f"{question_text} {train_prompt_suffix}"
        formatted_prompt = format_chat(tokenizer, question_text)
    else:
        if not _warned_missing_statement:
            print("WARNING: ex['statement'] missing -- falling back to the PRIMED "
                  "prompt as controller-training input (the pre-fix, low-signal "
                  "behavior). Regenerate dataset_with_completions.json with a "
                  "generate_completions.py that preserves 'statement' to fix this.")
            _warned_missing_statement = True
        formatted_prompt = ex.get("formatted_prompt") or format_chat(tokenizer, ex["prompt"])

    prompt_ids = tokenizer(formatted_prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(ex["completion"], add_special_tokens=False)["input_ids"]

    input_ids = torch.tensor([prompt_ids + completion_ids], device=device)
    loss_mask = torch.zeros(input_ids.shape[1], dtype=torch.bool, device=device)
    loss_mask[len(prompt_ids):] = True
    return input_ids, loss_mask


def _example_loss(language_model, controller, tokenizer, ex, device, train_prompt_suffix=None):
    """Runs one teacher-forced forward pass with the controller's hooks
    attached and returns the masked completion-token NLL as a scalar
    tensor (still attached to the autograd graph -- caller decides
    whether to backward() or just read .item()). Returns None if the
    example has no completion tokens to supervise."""
    input_ids, loss_mask = build_teacher_forced_batch(tokenizer, ex, device, train_prompt_suffix)
    if loss_mask.sum() == 0:
        return None

    controller.attach(language_model)
    try:
        out = language_model(input_ids=input_ids)
    finally:
        controller.detach()

    logits = out.logits
    shift_logits = logits[:, :-1, :]
    shift_targets = input_ids[:, 1:]
    shift_mask = loss_mask[1:]
    if shift_mask.sum() == 0:
        return None

    token_losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_targets.reshape(-1),
        reduction="none",
    ).reshape(shift_targets.shape)
    return (token_losses * shift_mask.unsqueeze(0)).sum() / shift_mask.sum()


@torch.no_grad()
def _val_loss(language_model, controller, tokenizer, val_examples, device, train_prompt_suffix=None):
    controller.eval()
    losses = []
    for ex in val_examples:
        loss = _example_loss(language_model, controller, tokenizer, ex, device, train_prompt_suffix)
        if loss is not None:
            losses.append(loss.item())
    controller.train()
    return sum(losses) / max(len(losses), 1)


def train_one_controller(language_model, tokenizer, train_examples, val_examples,
                          directions_per_layer, d_model, epochs=8, lr=3e-3,
                          bias_lr=1e-1, accum_steps=10, weight_decay=1e-4, grad_clip=1.0,
                          patience=2, device="cuda", log_every=20,
                          train_prompt_suffix=None):
    """
    CLAS-aligned schedule: AdamW, effective batch size `accum_steps`
    (one example forward/backward at a time, gradients accumulated over
    `accum_steps` examples before each optimizer.step() -- i.e. batch
    size 1 x accum_steps grad-accumulation steps, matching the paper's
    "batch size 1 with 10 gradient accumulation steps" exactly when
    accum_steps=10). Per-example losses are averaged (not summed) over
    the accumulation window, so the effective loss scale -- and hence
    what `lr` means -- doesn't depend on accum_steps.

    bias_lr: a SEPARATE learning rate for each layer's coefficient-bias
    term (LayerController.linear.bias in controller.py -- the "+1" in
    [h;1] from the pdf's Section 6 formula, implemented as a standard
    nn.Linear bias), matching the paper's separate "coefficient bias"
    learning rate. Confirmed against controller.py's actual structure
    (each per-layer LayerController is nn.Linear(d_model, k, bias=True));
    the frozen D_ell subspaces are buffers, not parameters, so they're
    never in either optimizer group.

    patience: stop once val loss hasn't improved for this many
    consecutive epochs. The best-val-loss checkpoint is always kept
    regardless of when/whether early stopping triggers.
    """
    controller = SequenceRFMController(directions_per_layer, d_model,
                                        dtype=language_model.dtype).to(device)
    controller.train()

    # Freeze the base model entirely -- only C_ell's parameters are trained.
    for p in language_model.parameters():
        p.requires_grad_(False)
    language_model.eval()  # keep dropout/etc off; we still need grad to FLOW through it

    # Confirmed against controller.py: each layer's LayerController is
    # nn.Linear(d_model, k, bias=True) -- .linear.bias IS the CLAS
    # "coefficient bias" term exactly (the "+1" in [h;1] implemented the
    # standard nn.Linear way), and the frozen D_ell subspaces are
    # buffers, not parameters, so they're already excluded from both
    # groups below automatically.
    main_params, bias_params = [], []
    for layer_controller in controller.controllers.values():
        main_params.append(layer_controller.linear.weight)
        bias_params.append(layer_controller.linear.bias)
    assert len(main_params) == len(bias_params) == len(controller.layers), (
        f"expected one weight+bias pair per layer ({len(controller.layers)} layers), "
        f"got {len(main_params)} weights / {len(bias_params)} biases -- "
        f"controller.py's LayerController structure may have changed."
    )
    print(f"  [bias_lr] {len(bias_params)} per-layer coefficient-bias terms at lr={bias_lr}, "
          f"{len(main_params)} per-layer weight matrices at lr={lr}")
    optimizer = torch.optim.AdamW([
        {"params": main_params, "lr": lr},
        {"params": bias_params, "lr": bias_lr},
    ], weight_decay=weight_decay)

    history = []
    best_val_loss = float("inf")
    best_state = None
    best_epoch = None
    epochs_since_improvement = 0
    total_update_steps = 0

    for epoch in range(epochs):
        epoch_losses = []
        optimizer.zero_grad()
        accum_count = 0

        def _flush():
            nonlocal accum_count, total_update_steps
            torch.nn.utils.clip_grad_norm_(controller.trainable_parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            accum_count = 0
            total_update_steps += 1

        for i, ex in enumerate(tqdm(train_examples, desc=f"epoch {epoch+1}/{epochs}")):
            loss = _example_loss(language_model, controller, tokenizer, ex,
                                  language_model.device, train_prompt_suffix)
            if loss is None:
                continue  # empty completion, nothing to supervise

            (loss / accum_steps).backward()
            accum_count += 1
            epoch_losses.append(loss.item())

            if accum_count == accum_steps:
                _flush()

            if (i + 1) % log_every == 0:
                recent = sum(epoch_losses[-log_every:]) / len(epoch_losses[-log_every:])
                print(f"  epoch {epoch+1} step {i+1}/{len(train_examples)}  "
                      f"train_loss={recent:.4f}  update_steps={total_update_steps}")

        if accum_count > 0:  # flush a final partial accumulation window
            _flush()

        train_mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        val_mean_loss = _val_loss(language_model, controller, tokenizer, val_examples,
                                   device, train_prompt_suffix)
        is_best = val_mean_loss < best_val_loss
        print(f"epoch {epoch+1}/{epochs}  train_loss={train_mean_loss:.4f}  "
              f"val_loss={val_mean_loss:.4f}  update_steps={total_update_steps}"
              f"{'  <- best so far' if is_best else ''}")
        history.append({"epoch": epoch + 1, "train_loss": train_mean_loss,
                         "val_loss": val_mean_loss, "update_steps": total_update_steps})

        if is_best:
            best_val_loss = val_mean_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(controller.state_dict())
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"\nEarly stopping: val_loss hasn't improved for {patience} "
                      f"consecutive epochs (best was epoch {best_epoch}).")
                break

    if best_state is not None:
        controller.load_state_dict(best_state)
        print(f"\nLoaded best checkpoint: epoch {best_epoch}, val_loss={best_val_loss:.4f} "
              f"(final epoch {history[-1]['epoch']} val_loss was {history[-1]['val_loss']:.4f}, "
              f"{total_update_steps} total update steps)")
    else:
        print("\nWARNING: no valid val loss computed on any epoch -- "
              "keeping final-epoch weights as-is.")

    return controller, history, best_epoch, best_val_loss


def main(args):
    torch.manual_seed(args.seed)

    directions_per_layer = load_directions_for_k(args.seqrfm_path, k=args.k)
    print(f"Loaded D_ell for {len(directions_per_layer)} layers, k={args.k}, "
          f"from {args.seqrfm_path}")

    examples = _extract_examples(args.dataset_with_completions_path, label_filter=1)
    train_examples, val_examples = train_val_split(examples, val_frac=args.val_frac, seed=args.seed)
    print(f"{len(examples)} label==1 completions from {args.dataset_with_completions_path} "
          f"-> {len(train_examples)} train / {len(val_examples)} val "
          f"(val_frac={args.val_frac})")

    language_model, tokenizer = load_model(args.model, cache_dir=args.cache_dir)
    d_model = language_model.config.hidden_size

    controller, history, best_epoch, best_val_loss = train_one_controller(
        language_model, tokenizer, train_examples, val_examples, directions_per_layer, d_model,
        epochs=args.epochs, lr=args.lr, bias_lr=args.bias_lr, accum_steps=args.accum_steps,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip, patience=args.patience,
        device=language_model.device, log_every=args.log_every,
        train_prompt_suffix=args.train_prompt_suffix,
    )

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torch.save({
        "controller_state_dict": controller.state_dict(),
        "layers": controller.layers,
        "k": controller.k,
        "d_model": d_model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "val_frac": args.val_frac,
        "train_prompt_suffix": args.train_prompt_suffix,
        "seqrfm_path": args.seqrfm_path,
        "seqrfm_method": seqrfm_method_label(args.seqrfm_path),
        "dataset_with_completions_path": args.dataset_with_completions_path,
    }, args.out_path)
    print(f"\nWrote {args.out_path}  (best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seqrfm_path", required=True,
                    help="sequence_rfm_directions.pt trained with --k matching --k below")
    p.add_argument("--dataset_with_completions_path", required=True,
                    help="dataset_with_completions.json for this persona (see schema note in docstring)")
    p.add_argument("--k", type=int, required=True, choices=[1, 3, 5])
    p.add_argument("--out_path", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--epochs", type=int, default=8,
                    help="read as EPOCHS, not the paper's literal step count -- see module "
                         "docstring for why (CLAS's '8 update steps' ~= 8 epochs for THEIR "
                         "dataset size, since effective batch size ~= their dataset size)")
    p.add_argument("--val_frac", type=float, default=0.2,
                    help="fraction of label==1 examples held out for val-loss-based "
                         "best-checkpoint selection and early stopping")
    p.add_argument("--train_prompt_suffix", default=None,
                    help='same suffix (if any) passed to generate_completions.py -- '
                         'e.g. "Answer in 15 words or less." for short_instructed_cut. '
                         'Leave unset for long_subsampled_anchor.')
    p.add_argument("--lr", type=float, default=3e-3, help="main controller weights, CLAS-aligned")
    p.add_argument("--bias_lr", type=float, default=1e-1,
                    help="separate lr for each layer's coefficient-bias term, CLAS's Llama "
                         "setting (0.5 for Qwen per the paper -- change if you switch model family)")
    p.add_argument("--accum_steps", type=int, default=10,
                    help="gradient-accumulation steps at batch_size=1 -- effective batch size, "
                         "CLAS-aligned default of 10")
    p.add_argument("--patience", type=int, default=2,
                    help="stop after this many consecutive epochs with no val_loss improvement")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args)