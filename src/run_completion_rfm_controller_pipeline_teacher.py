"""
Completion-RFM + controller pipeline, k=1 only — TEACHER variant.

Drop-in replacement for run_completion_rfm_controller_pipeline.py that
reads teacher-generated completions (via --teacher_datasets_dir) by
default. Results go to a separate _teacher strategy dir so they never
collide with self-completion runs.

Same three-step shape as run_persona_controller_pipeline.py /
run_fear_controller_pipeline.py, but the frozen direction D_ell comes
from completion_rfm.py (last-token embedding of (x, z)) instead of
bag-of-words-RFM or sequence-RFM -- and only k=1, since completion_rfm.py
only ever extracts a single (n_components=1) direction per layer, so
there's no k=3/5 to sweep here.

  1. Train completion-RFM directions (completion_rfm.py) if not already
     present, reusing the SAME dataset_with_completions.json as the
     other methods (--old_output_dir). Convert its .pkl output to the
     .pt container format load_directions_for_k expects
     (completion_rfm_to_pt.py) -- see that file's own warning: this
     conversion is unverified against train_controller.py's actual
     expectations, sanity-check before trusting a real run.
  2. Train a CLAS-style controller C_ell on top of the frozen k=1
     direction (train_controller.py).
  3. Steer + evaluate with the trained controller (no alpha grid).

*** NOT YET CLAS-ALIGNED: this driver calls train_controller.py's
existing train_one_controller(...) with whatever hyperparameters you
pass via --epochs/--lr/etc. It does NOT yet implement AdamW + effective
batch size 10 (batch 1 x 10 grad-accum steps) + max 8 update steps +
separate coefficient-bias LR from the CLAS paper -- those require
changes inside train_controller.py itself, which hasn't been shared.
Until that's done, --epochs/--lr here just pass through to whatever
train_controller.py currently does. ***

REQUIRED existing artifacts per concept, from your prior
{fear,persona}_experiment.py run (--old_output_dir should be the same
--output_dir you passed there):

    <old_output_dir>/{fear,persona}_<safe_name>/<strategy>/dataset_with_completions.json

completion_rfm.py doesn't need trajectories.pt (it reuses
dataset_with_completions.json directly, last-token embedding of the
full formatted_prompt+completion string).
"""
import argparse
import csv
import json
import os

import completion_rfm
import completion_rfm_to_pt
import train_controller

CONCEPT_CONFIG = {
    "fear": {"manifest_key": "fear", "dir_prefix": "fear"},
    "persona": {"manifest_key": "persona", "dir_prefix": "persona"},
}


def safe_name(name):
    return name.lower().replace(" ", "_")


def run_one(concept_name, dataset_path, old_strategy_dir, new_strategy_dir, args,
            teacher_completions_path=None):
    # Use teacher dataset if provided (ablation), otherwise fall back to self-completions
    if teacher_completions_path is not None:
        completions_path = teacher_completions_path
        print(f"  Using TEACHER completions: {completions_path}")
    else:
        completions_path = os.path.join(old_strategy_dir, "dataset_with_completions.json")
    if not os.path.exists(completions_path):
        print(f"  SKIP {concept_name}: missing dataset_with_completions.json under {completions_path}")
        return None

    os.makedirs(new_strategy_dir, exist_ok=True)
    pkl_path = os.path.join(new_strategy_dir, "completion_rfm_directions.pkl")
    pt_path = os.path.join(new_strategy_dir, "completion_rfm_directions.pt")
    controller_path = os.path.join(new_strategy_dir, "controller.pt")
    eval_dir = os.path.join(new_strategy_dir, "eval")

    print(f"\n--- {concept_name}  method=completion_rfm  k=1 ---")

    print("[1/3] train completion-RFM (x,z last token)")
    if not os.path.exists(pkl_path):
        completion_rfm.train_completion_rfm(
            completions_path, pkl_path, args.model,
            rfm_iters=args.rfm_iters, n_components=1, batch_size=args.rfm_batch_size,
            cache_dir=args.cache_dir, seed=args.seed,
        )
    else:
        print(f"  {pkl_path} exists, skipping.")

    if not os.path.exists(pt_path):
        completion_rfm_to_pt.convert(pkl_path, pt_path, k=1)
    else:
        print(f"  {pt_path} exists, skipping.")

    print("[2/3] train controller (k=1)")
    if not os.path.exists(controller_path):
        directions_per_layer = train_controller.load_directions_for_k(pt_path, k=1)
        language_model, tokenizer = train_controller.load_model(args.model, cache_dir=args.cache_dir)
        d_model = language_model.config.hidden_size
        examples = train_controller._extract_examples(completions_path, label_filter=1)
        train_examples, val_examples = train_controller.train_val_split(
            examples, val_frac=args.val_frac, seed=args.seed)
        controller, history, best_epoch, best_val_loss = train_controller.train_one_controller(
            language_model, tokenizer, train_examples, val_examples, directions_per_layer, d_model,
            epochs=args.epochs, lr=args.lr, bias_lr=args.bias_lr, accum_steps=args.accum_steps,
            weight_decay=args.weight_decay, grad_clip=args.grad_clip, patience=args.patience,
            device=language_model.device, log_every=args.log_every,
            train_prompt_suffix=args.train_prompt_suffix,
        )
        import torch
        os.makedirs(os.path.dirname(controller_path), exist_ok=True)
        torch.save({
            "controller_state_dict": controller.state_dict(), "layers": controller.layers,
            "k": controller.k, "d_model": d_model, "history": history,
            "best_epoch": best_epoch, "best_val_loss": best_val_loss, "val_frac": args.val_frac,
            "seqrfm_path": pt_path, "seqrfm_method": "completion_rfm",
            "dataset_with_completions_path": completions_path,
            "model_dtype": str(language_model.dtype),
        }, controller_path)
        print(f"  wrote {controller_path}  (best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f})")
    else:
        print(f"  {controller_path} exists, skipping.")

    print("[3/3] steer + evaluate (k=1)")
    steer_and_evaluate = args.steer_and_evaluate_module
    steer_and_evaluate.run_evaluation(
        dataset_path, controller_path, eval_dir, args.model,
        judge_model=args.judge_model, max_new_tokens=args.eval_gen_tokens,
        cache_dir=args.cache_dir, seed=args.seed,
    )
    return os.path.join(eval_dir, "summary.csv")


def main(args):
    cfg = CONCEPT_CONFIG[args.concept_type]

    if args.concept_type == "fear":
        import fear_controller_steer_and_evaluate as steer_and_evaluate_module
    else:
        import controller_steer_and_evaluate as steer_and_evaluate_module
    args.steer_and_evaluate_module = steer_and_evaluate_module

    with open(args.manifest_path) as f:
        manifest = json.load(f)

    all_summaries = []
    for entry in manifest:
        concept_name = entry[cfg["manifest_key"]]
        dataset_path = entry["path"]
        dir_name = f"{cfg['dir_prefix']}_{safe_name(concept_name)}"
        old_strategy_dir = os.path.join(args.old_output_dir, dir_name, args.completion_strategy)
        new_strategy_dir = os.path.join(args.new_output_dir, dir_name, args.completion_strategy)

        # Resolve teacher dataset path if --teacher_datasets_dir is provided
        teacher_completions_path = None
        if args.teacher_datasets_dir:
            slug = safe_name(concept_name)
            model_tag = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "_")
            teacher_completions_path = os.path.join(
                args.teacher_datasets_dir,
                f"{cfg['dir_prefix']}s",  # e.g. "personas" or "fears"
                model_tag,
                f"{slug}_teacher_{model_tag}.json",
            )
            if not os.path.exists(teacher_completions_path):
                print(f"  WARNING: teacher dataset not found at {teacher_completions_path}, "
                      f"falling back to self-completions.")
                teacher_completions_path = None

        summary_path = run_one(concept_name, dataset_path, old_strategy_dir, new_strategy_dir,
                               args, teacher_completions_path=teacher_completions_path)
        if summary_path:
            all_summaries.append(summary_path)

    combined_path = os.path.join(
        args.new_output_dir, f"combined_summary_controller_{args.concept_type}_completion_rfm_teacher.csv")
    os.makedirs(args.new_output_dir, exist_ok=True)
    with open(combined_path, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow([cfg["manifest_key"], "method", "k", "mean_score", "n"])
        for sp in all_summaries:
            with open(sp) as in_f:
                reader = csv.reader(in_f)
                next(reader)
                for row in reader:
                    writer.writerow(row)

    print(f"\nAll concepts done (completion_rfm, k=1). Combined summary: {combined_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--concept_type", required=True, choices=["fear", "persona"])
    p.add_argument("--manifest_path", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--old_output_dir", required=True,
                    help="root of your EXISTING {fear,persona}_experiment.py output, containing "
                         "{fear,persona}_<name>/<strategy>/dataset_with_completions.json")
    p.add_argument("--new_output_dir", required=True,
                    help="fresh directory for this experiment -- never overlaps with --old_output_dir")
    p.add_argument("--completion_strategy", default="teacher_long_subsampled_anchor",
                    help="strategy tag used for output subdirectory naming; defaults to "
                         "teacher_long_subsampled_anchor to separate from self-completion runs")
    p.add_argument("--cache_dir", default=None)
    # completion-RFM hyperparameters (same defaults as completion_rfm.py)
    p.add_argument("--rfm_iters", type=int, default=8)
    p.add_argument("--rfm_batch_size", type=int, default=8)
    # controller training hyperparameters -- NOT YET CLAS-aligned, see module docstring
    p.add_argument("--epochs", type=int, default=8,
                    help="read as EPOCHS, not the paper's literal step count -- see "
                         "train_controller.py's module docstring for why")
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--train_prompt_suffix", default=None)
    p.add_argument("--lr", type=float, default=3e-3, help="main controller weights, CLAS-aligned")
    p.add_argument("--bias_lr", type=float, default=1e-1,
                    help="separate lr for any 'bias'-named controller parameter, CLAS's Llama "
                         "setting (0.5 for Qwen per the paper)")
    p.add_argument("--accum_steps", type=int, default=10,
                    help="gradient-accumulation steps at batch_size=1 -- effective batch size")
    p.add_argument("--patience", type=int, default=2,
                    help="stop after this many consecutive epochs with no val_loss improvement")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=20)
    # eval
    p.add_argument("--eval_gen_tokens", type=int, default=100)
    p.add_argument("--judge_model", default="gpt-4o-mini")
    p.add_argument("--teacher_datasets_dir", default=None,
                    help="optional: root of teacher-formatted datasets from format_teacher_datasets.py "
                         "(teacher_ablation/datasets). When set, uses teacher completions instead of "
                         "self-completions for direction training and controller training.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    main(args)
