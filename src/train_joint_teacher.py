"""
train_joint_teacher.py

Teacher-completion variant of train_joint_new.py.
Identical logic; output directory gets a _teacher suffix so results
never collide with self-completion runs.

Jointly trains a unit-norm steering direction D_l (k=1) and a per-layer
linear controller C_l, using e_y (label embedding) as input to C_l.

Training pairs (all use neutral_prompt as input -- no concept cue):
  e_y=1: (neutral_prompt, concept_completion) → steer ON
    D and C_l must inject the concept to make concept_completion likely.
  e_y=0: (neutral_prompt, neutral_completion) → steer ON, but C_l sees e_y=0
    C_l learns to output alpha≈0 when e_y=0, so D is effectively silenced.
    Both pair types produce gradient for C_l. Only e_y=1 pairs produce
    meaningful gradient for D (since e_y=0 pairs teach C_l to zero out alpha).

Controller input: [h_{l,t}; e_y; 1]  (d + 1 + 1 = d+2 dimensions)
  e_y is a scalar: 1.0 for concept pairs, 0.0 for neutral pairs.
  At inference, always pass e_y=1.

D_l parameterized as v_l / ||v_l|| (unit-sphere). Initialized from
diff-means if --init_from_diff_means is provided, else random.

Outputs (--out_dir/<slug>/):
  directions.pt      -- {layer: unit-norm d-vector}
  controllers.pt     -- {layer: C_l state_dict}
  training_log.json  -- per-epoch and per-step losses

Usage:
    python src/train_joint.py \\
        --dataset outputs/datasets/blood.json \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --out_dir outputs/models \\
        --epochs 8 --lr 3e-3 --dir_lr 1e-4 \\
        --accum_steps 10 --log_every 10 \\
        --val_frac 0.15 --patience 2
"""
import argparse
import json
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.insert(0, ".")
from utils import load_model

# Assistant tag is derived from the tokenizer's own chat template via
# add_generation_prompt=True -- works for Llama, Qwen, and any other model.


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

QUESTION_TEMPLATE = "What are your thoughts on the following statement?\nStatement: {statement}"

def build_pairs(dataset):
    """
    Handles two dataset formats:

    FORMAT A — dataset_with_completions.json (label=0/1):
        Works for both:
          - 400x2: same 400 statements for label=0 and label=1 (matched)
          - 200/200: different 200 statements for each label (unmatched)

        Training pairs:
          e_y=1: (neutral_prompt from statement, concept completion from label=1[i])
          e_y=0: (neutral_prompt from statement, neutral completion from label=0[i])

        Neutral prompt is reconstructed from ex["statement"] via QUESTION_TEMPLATE,
        NOT from ex["prompt"] for label=1 -- same reason as train_controller.py:
        the unsteered model already reproduces label=1 completion well when given
        the concept-prefix prompt, leaving D and C nothing to learn.

    FORMAT B — build_dataset.py + generate_completions.py output:
        flat "train" list with neutral_prompt/concept_completion/neutral_completion fields.
    """
    train = dataset["train"]

    if "label" in train[0]:
        # FORMAT A: label=0/1 dataset_with_completions.json
        label0 = [ex for ex in train if ex["label"] == 0]
        label1 = [ex for ex in train if ex["label"] == 1]

        missing_c = [i for i, ex in enumerate(label1) if not ex.get("completion", "").strip()]
        missing_n = [i for i, ex in enumerate(label0) if not ex.get("completion", "").strip()]
        if missing_c or missing_n:
            raise ValueError(
                f"Missing completions: {len(missing_c)} concept (label=1), "
                f"{len(missing_n)} neutral (label=0). "
                f"Run generate_completions.py first."
            )

        n = min(len(label0), len(label1))
        pairs = []
        for i in range(n):
            # Reconstruct neutral prompt from statement for both groups
            neutral_prompt_1 = QUESTION_TEMPLATE.format(statement=label1[i]["statement"])
            neutral_prompt_0 = QUESTION_TEMPLATE.format(statement=label0[i]["statement"])
            # e_y=1: neutral prompt + concept completion
            pairs.append({
                "prompt":     neutral_prompt_1,
                "completion": label1[i]["completion"],
                "e_y":        1.0,
            })
            # e_y=0: neutral prompt + neutral completion
            pairs.append({
                "prompt":     neutral_prompt_0,
                "completion": label0[i]["completion"],
                "e_y":        0.0,
            })
        print(f"  [build_pairs] FORMAT A: {len(pairs)} pairs "
              f"({n} label=1, {n} label=0, "
              f"{'matched' if len(set(ex['statement'] for ex in label1)) == len(set(ex['statement'] for ex in label0)) else 'unmatched'} statements).")

    else:
        # FORMAT B: fear dataset with concept_prompt/neutral_prompt
        missing_c = [i for i, ex in enumerate(train)
                     if not ex.get("concept_completion", "").strip()]
        missing_n = [i for i, ex in enumerate(train)
                     if not ex.get("neutral_completion", "").strip()]
        if missing_c or missing_n:
            raise ValueError(
                f"Missing completions: {len(missing_c)} concept, "
                f"{len(missing_n)} neutral. "
                f"Run generate_completions.py first."
            )
        pairs = []
        for ex in train:
            pairs.append({
                "prompt":     ex["neutral_prompt"],
                "completion": ex["concept_completion"],
                "e_y":        1.0,
            })
            pairs.append({
                "prompt":     ex["neutral_prompt"],
                "completion": ex["neutral_completion"],
                "e_y":        0.0,
            })
        print(f"  [build_pairs] FORMAT B: {len(pairs)} pairs from {len(train)} statements.")

    return pairs


class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        return p["prompt"], p["completion"], p["e_y"]


# ---------------------------------------------------------------------------
# Controller (one per layer) — takes [h; e_y; 1] as input
# ---------------------------------------------------------------------------

class LayerController(nn.Module):
    """
    alpha_t = W @ [h_t; e_y; 1]
    Input dim: d + 2  (hidden + label scalar + bias term)
    At inference: always pass e_y=1.0
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size + 2, 1, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, h, e_y):
        # h may be BFloat16 from the model; cast to float32 for the linear layer
        h = h.float()
        T    = h.shape[0]
        ey   = torch.full((T, 1), e_y, device=h.device, dtype=torch.float32)
        ones = torch.ones( T,  1,      device=h.device, dtype=torch.float32)
        h_aug = torch.cat([h, ey, ones], dim=-1)   # (T, d+2)
        return self.linear(h_aug).squeeze(-1)       # (T,)


# ---------------------------------------------------------------------------
# Hook machinery
# ---------------------------------------------------------------------------

class SteeringState:
    def __init__(self):
        self.directions  = {}   # layer -> unit-norm (d,)
        self.controllers = {}   # layer -> LayerController
        self.e_y         = 1.0  # current label scalar, set per example


def make_hooks(model, state, layers):
    hooks = []

    def make_pre_hook(layer_idx):
        def hook(module, args):
            h     = args[0]   # (1, T, d) -- BFloat16
            h_2d  = h[0]
            d_vec = state.directions[layer_idx]          # float32
            alpha = state.controllers[layer_idx](h_2d.detach(), state.e_y)  # float32 (T,)
            delta = alpha.unsqueeze(-1) * d_vec.unsqueeze(0)                # float32 (T, d)
            # cast delta back to h's dtype before adding
            return ((h[0] + delta.to(h.dtype)).unsqueeze(0),) + args[1:]
        return hook

    for layer_idx in layers:
        h = model.model.layers[layer_idx].register_forward_pre_hook(
            make_pre_hook(layer_idx)
        )
        hooks.append(h)
    return hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def format_chat(tokenizer, prompt_text):
    """Model-agnostic: uses the tokenizer's own chat template for both Llama and Qwen."""
    chat = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def compute_nll(model, tokenizer, prompt_text, completion_text, device):
    full_text = format_chat(tokenizer, prompt_text) + completion_text
    enc       = tokenizer(full_text, return_tensors="pt",
                          add_special_tokens=False).to(device)
    input_ids = enc["input_ids"]

    prompt_enc = tokenizer(
        format_chat(tokenizer, prompt_text),
        return_tensors="pt", add_special_tokens=False,
    ).to(device)
    prompt_len = prompt_enc["input_ids"].shape[1]

    labels = input_ids.clone()
    labels[0, :prompt_len] = -100
    return model(input_ids=input_ids, labels=labels).loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args):
    torch.manual_seed(args.seed)

    with open(args.dataset) as f:
        dataset = json.load(f)

    concept  = dataset.get("concept") or dataset.get("persona") or dataset.get("fear")

    all_pairs = build_pairs(dataset)
    random.shuffle(all_pairs)

    n_val       = max(1, int(len(all_pairs) * args.val_frac))
    val_pairs   = all_pairs[-n_val:]
    train_pairs = all_pairs[:-n_val]

    n1 = sum(1 for p in all_pairs if p["e_y"] == 1.0)
    n0 = sum(1 for p in all_pairs if p["e_y"] == 0.0)
    print(f"Concept : {concept}")
    print(f"Pairs   : {len(all_pairs)} total  "
          f"({len(train_pairs)} train, {len(val_pairs)} val)")
    print(f"  e_y=1 (class_1 neutral prompt + concept completion): {n1}")
    print(f"  e_y=0 (class_0 neutral prompt + neutral completion): {n0}")

    train_ds = PairDataset(train_pairs)
    val_ds   = PairDataset(val_pairs)

    model, tokenizer = load_model(args.model, args.cache_dir)
    model.train()
    device   = next(model.parameters()).device
    d        = model.config.hidden_size
    n_layers = model.config.num_hidden_layers

    layers = [int(l) for l in args.layers] if args.layers else list(range(n_layers))
    print(f"Steering layers: {layers}")

    # --- Initialize directions ---
    raw_directions = {}
    if args.init_from_diff_means and os.path.exists(args.init_from_diff_means):
        import pickle
        with open(args.init_from_diff_means, "rb") as f:
            baselines = pickle.load(f)
        dm = baselines.get("mean_difference", {}).get("directions_per_layer", {})
        dm = {int(k) % n_layers: v for k, v in dm.items()}
        for layer in layers:
            v = dm[layer].float().clone().to(device) if layer in dm \
                else F.normalize(torch.randn(d, device=device), dim=0)
            raw_directions[layer] = nn.Parameter(v)
        print("Initialized D from diff-means directions.")
    else:
        for layer in layers:
            raw_directions[layer] = nn.Parameter(
                F.normalize(torch.randn(d, device=device), dim=0).float()
            )
        print("Initialized D randomly.")

    # --- Initialize controllers (d+2 input: h + e_y + bias) ---
    controllers = {layer: LayerController(d).to(device) for layer in layers}

    # --- Steering state ---
    state = SteeringState()
    state.directions  = {l: F.normalize(raw_directions[l], dim=0) for l in layers}
    state.controllers = controllers

    # --- Optimizer ---
    dir_params  = list(raw_directions.values())
    ctrl_params = [p for c in controllers.values() for p in c.parameters()]
    optimizer = torch.optim.AdamW([
        {"params": dir_params,  "lr": args.dir_lr, "weight_decay": 0.0},
        {"params": ctrl_params, "lr": args.lr,      "weight_decay": 1e-4},
    ])

    # Write directly into args.out_dir (caller is responsible for namespacing)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    log = {
        "train_loss_per_epoch": [],
        "val_loss_per_epoch":   [],
        "train_loss_per_step":  [],
        "val_loss_per_step":    [],
        "best_epoch": 0,
    }
    best_val_loss    = float("inf")
    patience_counter = 0
    best_dir_state   = None
    best_ctrl_state  = None
    global_step      = 0

    hooks = make_hooks(model, state, layers)

    for epoch in range(args.epochs):
        indices = list(range(len(train_ds)))
        random.shuffle(indices)

        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        epoch_loss   = 0.0
        step_count   = 0

        for i, idx in enumerate(tqdm(indices, desc=f"Epoch {epoch+1}/{args.epochs} train")):
            prompt, completion, e_y = train_ds[idx]

            for l in layers:
                state.directions[l] = F.normalize(raw_directions[l], dim=0)
            state.e_y = float(e_y)   # set label scalar for this example

            loss         = compute_nll(model, tokenizer, prompt, completion, device)
            loss         = loss / args.accum_steps
            loss.backward()

            example_loss  = loss.item() * args.accum_steps
            running_loss += example_loss
            epoch_loss   += example_loss

            if (i + 1) % args.accum_steps == 0 or (i + 1) == len(indices):
                torch.nn.utils.clip_grad_norm_(dir_params + ctrl_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                step_count  += 1
                global_step += 1

                n_in_window  = min(args.accum_steps,
                                   i + 1 - (step_count - 1) * args.accum_steps)
                step_loss    = running_loss / n_in_window
                running_loss = 0.0

                log["train_loss_per_step"].append({
                    "epoch": epoch + 1, "step": step_count,
                    "global_step": global_step, "loss": round(step_loss, 6),
                })

                if step_count % args.log_every == 0:
                    print(f"  [epoch {epoch+1} step {step_count} "
                          f"| global {global_step}] train_loss={step_loss:.4f}")

        avg_train_loss = epoch_loss / len(indices)
        log["train_loss_per_epoch"].append(round(avg_train_loss, 6))

        # --- Validation ---
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for prompt, completion, e_y in tqdm(val_ds, desc=f"Epoch {epoch+1} val"):
                for l in layers:
                    state.directions[l] = F.normalize(raw_directions[l], dim=0)
                state.e_y = float(e_y)
                val_loss_total += compute_nll(
                    model, tokenizer, prompt, completion, device
                ).item()

        avg_val_loss = val_loss_total / len(val_ds)
        log["val_loss_per_epoch"].append(round(avg_val_loss, 6))
        log["val_loss_per_step"].append({
            "epoch": epoch + 1, "global_step": global_step,
            "loss":  round(avg_val_loss, 6),
        })

        print(f"  Epoch {epoch+1}: train_loss={avg_train_loss:.4f}  "
              f"val_loss={avg_val_loss:.4f}")

        # --- Early stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss    = avg_val_loss
            patience_counter = 0
            log["best_epoch"] = epoch + 1
            best_dir_state  = {
                l: F.normalize(raw_directions[l], dim=0).detach().cpu().clone()
                for l in layers
            }
            best_ctrl_state = {
                l: {k: v.detach().cpu().clone()
                    for k, v in controllers[l].state_dict().items()}
                for l in layers
            }
            print(f"  -> New best val_loss={best_val_loss:.4f}, saving checkpoint.")
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print("Early stopping.")
                break

    remove_hooks(hooks)

    directions_path  = os.path.join(out_dir, "directions.pt")
    controllers_path = os.path.join(out_dir, "controllers.pt")
    log_path         = os.path.join(out_dir, "training_log.json")

    torch.save(best_dir_state, directions_path)
    torch.save({l: best_ctrl_state[l] for l in layers}, controllers_path)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nSaved directions  -> {directions_path}")
    print(f"Saved controllers -> {controllers_path}")
    print(f"Saved log         -> {log_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",              required=True)
    p.add_argument("--model",                required=True)
    p.add_argument("--out_dir",              default="outputs/models")
    p.add_argument("--layers",               nargs="+", type=int, default=None)
    p.add_argument("--epochs",               type=int,   default=8)
    p.add_argument("--lr",                   type=float, default=3e-3)
    p.add_argument("--dir_lr",               type=float, default=1e-4)
    p.add_argument("--accum_steps",          type=int,   default=10)
    p.add_argument("--log_every",            type=int,   default=10)
    p.add_argument("--val_frac",             type=float, default=0.15)
    p.add_argument("--patience",             type=int,   default=2)
    p.add_argument("--init_from_diff_means", default=None)
    p.add_argument("--cache_dir",            default=None)
    p.add_argument("--seed",                 type=int,   default=42)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()