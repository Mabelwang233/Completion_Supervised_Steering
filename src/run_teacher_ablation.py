"""
run_teacher_ablation.py

Stage 3: runs all completion-based direction methods on teacher datasets
for one target model and one category (persona or fear).

Methods run per concept:
  1. completion_diff_means  -> completion_diff_means.py
  2. completion_rfm         -> completion_rfm.py
  3. contextual completion rfm -> run_completion_rfm_controller_pipeline_teacher.py
                                   + train_controller.py
  4. joint_training            -> train_joint_teacher.py
  5. clas                      -> clas.py (train_and_evaluate_clas.py pattern)

Evaluates all methods via persona_steer_and_evaluate / fear_steer_and_evaluate.
Writes results to:
  <out_dir>/<model_tag>/<category>_<slug>/teacher_long_subsampled_anchor/
    completion_rfm_directions.pkl
    completion_diff_means_directions.pkl
    per_example_results.csv
    summary.csv

Usage:
    python run_teacher_ablation.py \
        --ablation_manifest  teacher_ablation/ablation_manifest.json \
        --teacher_datasets_dir teacher_ablation/datasets \
        --category           persona \
        --model              meta-llama/Llama-3.1-8B-Instruct \
        --model_tag          llama \
        --out_dir            teacher_ablation/outputs \
        --alpha_grid         0.3 0.4 0.5 0.6 0.7 0.8 \
        --rfm_iters          8 \
        --cache_dir          /path/to/hf_cache \
        --judge_model        gpt-4o-mini
"""
import argparse
import json
import os
import sys

sys.path.insert(0, ".")
import gc
import completion_diff_means
import completion_rfm
import extract_trajectories


STRATEGY_TAG = "teacher_long_subsampled_anchor"


def run_one_concept(concept_name, concept_key, dataset_path, teacher_dataset_path,
                    out_dir, args):
    slug = concept_name.lower().replace(" ", "_").replace(".", "").replace(",", "")
    concept_dir   = os.path.join(out_dir, f"{concept_key}_{slug}")
    strategy_dir  = os.path.join(concept_dir, STRATEGY_TAG)
    os.makedirs(strategy_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"{concept_key.upper()}: {concept_name}  |  source: teacher  |  model: {args.model_tag}")
    print(f"{'='*60}")

    # Paths for direction artifacts
    completion_rfm_path  = os.path.join(strategy_dir, "completion_rfm_directions.pkl")
    completion_dm_path   = os.path.join(strategy_dir, "completion_diff_means_directions.pkl")
    trajectories_path    = os.path.join(strategy_dir, "trajectories.pt")

    # ------------------------------------------------------------------
    # Step 1: completion diff-means
    # ------------------------------------------------------------------
    print(f"\n[1/4] completion diff-means (teacher completions)")
    if not os.path.exists(completion_dm_path):
        completion_diff_means.train_completion_diff_means(
            teacher_dataset_path, completion_dm_path, args.model,
            cache_dir=args.cache_dir, seed=args.seed,
        )
    else:
        print(f"  {completion_dm_path} exists, skipping.")

    # ------------------------------------------------------------------
    # Step 2: completion RFM
    # ------------------------------------------------------------------
    print(f"\n[2/4] completion RFM (teacher completions)")
    if not os.path.exists(completion_rfm_path):
        completion_rfm.train_completion_rfm(
            teacher_dataset_path, completion_rfm_path, args.model,
            rfm_iters=args.rfm_iters, cache_dir=args.cache_dir, seed=args.seed,
        )
    else:
        print(f"  {completion_rfm_path} exists, skipping.")

    # ------------------------------------------------------------------
    # Step 3: contextual completion RFM (controller)
    # Uses run_completion_rfm_controller_pipeline.py's run_one() directly
    # with the teacher dataset path.
    # ------------------------------------------------------------------
    ctrl_rfm_path = os.path.join(strategy_dir, "completion_rfm_controller.pt")
    print(f"\n[3/4] contextual completion RFM controller (teacher completions)")
    if not os.path.exists(ctrl_rfm_path):
        try:
            import completion_rfm_to_pt
            import train_controller as train_controller_mod
            from run_completion_rfm_controller_pipeline_teacher import run_one as run_ctrl_one

            # Convert pkl -> pt format that train_controller expects
            import completion_rfm_to_pt
            pt_path = os.path.join(strategy_dir, "completion_rfm_directions.pt")
            if not os.path.exists(pt_path):
                completion_rfm_to_pt.convert(completion_rfm_path, pt_path, k=1)

            directions_per_layer = train_controller_mod.load_directions_for_k(pt_path, k=1)
            from utils import load_model as _load_model
            lm, tokenizer = _load_model(args.model, cache_dir=args.cache_dir)
            d_model = lm.config.hidden_size

            examples = train_controller_mod._extract_examples(teacher_dataset_path, label_filter=1)
            train_ex, val_ex = train_controller_mod.train_val_split(
                examples, val_frac=args.clas_val_frac, seed=args.seed)

            controller, history, best_epoch, best_val_loss = train_controller_mod.train_one_controller(
                lm, tokenizer, train_ex, val_ex, directions_per_layer, d_model,
                epochs=args.clas_epochs, lr=args.clas_lr_weight,
                bias_lr=args.clas_lr_bias, accum_steps=args.clas_accum_steps,
                patience=args.clas_patience, device=lm.device,
            )
            import torch
            torch.save({
                "controller_state_dict": controller.state_dict(),
                "layers": controller.layers, "k": controller.k,
                "d_model": d_model, "history": history,
                "best_epoch": best_epoch, "best_val_loss": best_val_loss,
                "seqrfm_path": pt_path, "seqrfm_method": "completion_rfm_teacher",
                "dataset_with_completions_path": teacher_dataset_path,
                "model_dtype": str(lm.dtype),
            }, ctrl_rfm_path)
            print(f"  Wrote {ctrl_rfm_path} (best_epoch={best_epoch})")
            del lm  # free GPU memory before evaluation step
        except ImportError as e:
            print(f"  SKIPPED: could not import required module: {e}")
    else:
        print(f"  {ctrl_rfm_path} exists, skipping.")

    # Evaluate contextual completion RFM controller
    ctrl_eval_dir = os.path.join(strategy_dir, "completion_rfm_controller_eval")
    ctrl_summary  = os.path.join(ctrl_eval_dir, "summary.csv")
    if os.path.exists(ctrl_rfm_path) and not os.path.exists(ctrl_summary):
        print(f"  Evaluating completion_rfm_controller...")
        try:
            if concept_key == "persona":
                import controller_steer_and_evaluate as ctrl_eval_mod
            else:
                import fear_controller_steer_and_evaluate as ctrl_eval_mod
            ctrl_eval_mod.run_evaluation(
                dataset_path, ctrl_rfm_path, ctrl_eval_dir, args.model,
                judge_model=args.judge_model, max_new_tokens=args.eval_gen_tokens,
                cache_dir=args.cache_dir, seed=args.seed,
            )
        except ImportError as e:
            print(f"  Controller eval SKIPPED: {e}")
        finally:
            # Controller eval loads a model internally — flush after it finishes
            gc.collect()
            import torch as _t; _t.cuda.empty_cache()
    elif os.path.exists(ctrl_summary):
        print(f"  {ctrl_summary} exists, skipping eval.")

    # ------------------------------------------------------------------
    # Step 4: joint training (train_joint_teacher.py)
    # Jointly trains direction D_l + controller C_l on teacher completions.
    # Outputs land in <out_dir>/<concept_slug>_teacher/ (separate from
    # self-completion joint training which uses <concept_slug>/).
    # ------------------------------------------------------------------
    # Explicitly free all GPU memory before launching joint training subprocess.
    # Steps 1-3 each load a model; if any is still resident, the subprocess
    # will OOM trying to load another full model on the same GPU.
    import torch as _torch
    gc.collect()
    _torch.cuda.empty_cache()
    print("  GPU memory freed before joint training.")

    if args.run_joint:
        print(f"\n[4/5] joint training (teacher completions)")
        joint_out_dir = os.path.join(concept_dir, f"joint_{STRATEGY_TAG}")
        os.makedirs(joint_out_dir, exist_ok=True)
        joint_directions = os.path.join(joint_out_dir, "directions.pt")
        joint_controllers = os.path.join(joint_out_dir, "controllers.pt")
        if not os.path.exists(joint_directions):
            import subprocess, os as _os
            # Pass expandable_segments to reduce fragmentation OOM
            env = _os.environ.copy()
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            cmd = [
                "python", "train_joint_teacher.py",
                "--dataset",     teacher_dataset_path,
                "--model",       args.model,
                "--out_dir",     joint_out_dir,
                "--epochs",      str(args.joint_epochs),
                "--lr",          str(args.joint_lr),
                "--dir_lr",      str(args.joint_dir_lr),
                "--accum_steps", str(args.joint_accum_steps),
                "--val_frac",    str(args.joint_val_frac),
                "--patience",    str(args.joint_patience),
                "--seed",        str(args.seed),
            ]
            if args.cache_dir:
                cmd += ["--cache_dir", args.cache_dir]
            try:
                subprocess.run(cmd, check=True, env=env)
            except subprocess.CalledProcessError as e:
                print(f"  WARNING: joint training failed for {concept_name} "
                      f"(exit code {e.returncode}). Skipping joint eval. "
                      f"If OOM, try reducing --joint_accum_steps or freeing GPU memory.")
        else:
            print(f"  {joint_directions} exists, skipping.")

        # Evaluate joint training
        joint_summary = os.path.join(joint_out_dir, "summary.csv")
        if os.path.exists(joint_directions) and not os.path.exists(joint_summary):
            print(f"  Evaluating joint training...")
            try:
                import torch, csv as _csv
                import torch.nn.functional as F
                from utils import load_model as _load_model
                from train_joint_teacher import LayerController

                if concept_key == "persona":
                    from persona_judge import judge_response as _judge
                else:
                    from fear_judge import judge_response as _judge

                dirs_raw  = torch.load(joint_directions, map_location="cpu")
                ctrls_raw = torch.load(joint_controllers, map_location="cpu")
                directions_j = {int(l): F.normalize(v, dim=0) for l, v in dirs_raw.items()}
                layers_j = sorted(directions_j.keys())

                lm_j, tok_j = _load_model(args.model, cache_dir=args.cache_dir)
                lm_j.eval()
                d_j = lm_j.config.hidden_size
                model_dtype = lm_j.dtype

                controllers_j = {}
                for l in layers_j:
                    # Controllers are saved as float32 from training (LayerController
                    # is initialized without dtype= so it defaults to float32).
                    # Keep them in float32 — we cast h_2d to float32 in the hook
                    # and cast the result back to model dtype afterward.
                    ctrl = LayerController(d_j)  # float32
                    ctrl.load_state_dict(ctrls_raw[l])
                    ctrl.to(device=lm_j.device)
                    ctrl.eval()
                    controllers_j[l] = ctrl

                # Hook-based steered generation (mirrors evaluate.py from joint steering repo)
                def _install_hooks(model, directions, controllers, layers):
                    hooks = []
                    def _make_hook(lidx):
                        def hook(module, args_h):
                            h = args_h[0]           # (1, T, d) — model's dtype (e.g. bfloat16)
                            h_2d = h[0]             # (T, d)
                            orig_dtype = h.dtype    # preserve for final cast
                            # Controller is float32 — cast h_2d to float32 for its forward pass
                            h_2d_f32 = h_2d.to(dtype=torch.float32)
                            d_vec = directions[lidx].to(device=h.device, dtype=torch.float32)
                            with torch.no_grad():
                                alpha = controllers[lidx](h_2d_f32, e_y=1.0)  # (T,) float32
                            delta = alpha.unsqueeze(-1) * d_vec.unsqueeze(0)   # (T, d) float32
                            # Cast delta and result back to original model dtype (bfloat16 for Qwen)
                            new_h = (h[0].to(torch.float32) + delta).to(dtype=orig_dtype).unsqueeze(0)
                            return (new_h,) + args_h[1:]
                        return hook
                    for l in layers:
                        hooks.append(model.model.layers[l].register_forward_pre_hook(_make_hook(l)))
                    return hooks

                with open(dataset_path) as _f:
                    ds_j = json.load(_f)
                test_j   = ds_j["test"]
                target_j = ds_j.get(concept_key, ds_j.get("concept", concept_name))

                joint_rows = os.path.join(joint_out_dir, "per_example_results.csv")
                scores_j = []
                with open(joint_rows, "w", newline="") as _f:
                    _w = _csv.writer(_f)
                    _w.writerow([concept_key, "method", "alpha", "question", "response", "score", "explanation"])
                    from tqdm import tqdm as _tqdm
                    for ex in _tqdm(test_j, desc="Joint eval"):
                        formatted = tok_j.apply_chat_template(
                            [{"role": "user", "content": ex["prompt"]}],
                            tokenize=False, add_generation_prompt=True
                        )
                        inputs = tok_j(formatted, return_tensors="pt", add_special_tokens=False).to(lm_j.device)
                        hooks = _install_hooks(lm_j, directions_j, controllers_j, layers_j)
                        try:
                            with torch.no_grad():
                                out_ids = lm_j.generate(
                                    **inputs, max_new_tokens=args.eval_gen_tokens,
                                    do_sample=False,
                                    temperature=1.0,  # suppress Qwen config sampling flags
                                    top_p=1.0,
                                    top_k=0,
                                    pad_token_id=tok_j.pad_token_id,
                                )
                        finally:
                            for hk in hooks: hk.remove()
                        gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
                        resp = tok_j.decode(gen_ids, skip_special_tokens=True)
                        sc, expl = _judge(target_j, ex.get("question", ex["prompt"]), resp, args.judge_model)
                        _w.writerow([target_j, "joint_teacher", 0.0, ex.get("question", ""), resp, sc, expl])
                        scores_j.append(sc)

                mean_j = sum(scores_j) / len(scores_j) if scores_j else 0.0
                with open(joint_summary, "w", newline="") as _f:
                    _w = _csv.writer(_f)
                    _w.writerow([concept_key, "method", "alpha", "mean_score", "n"])
                    _w.writerow([target_j, "joint_teacher", 0.0, mean_j, len(scores_j)])
                print(f"  Joint teacher mean_score: {mean_j:.3f}")
                del lm_j
            except (ImportError, FileNotFoundError) as e:
                print(f"  Joint eval SKIPPED: {e}")
        elif os.path.exists(joint_summary):
            print(f"  {joint_summary} exists, skipping eval.")

    # ------------------------------------------------------------------
    # Step 5: evaluate completion_diff_means + completion_rfm (alpha sweep)
    # ------------------------------------------------------------------
    print(f"\n[5/5] steer + evaluate (alpha sweep)")
    step5_summary = os.path.join(strategy_dir, "summary.csv")
    if os.path.exists(step5_summary):
        print(f"  {step5_summary} exists, skipping eval.")
    else:
        if concept_key == "persona":
            import persona_steer_and_evaluate as evaluator
        else:
            import fear_steer_and_evaluate as evaluator

        # Write to temp dir then append to avoid overwriting
        import csv, shutil
        tmp_dir = os.path.join(strategy_dir, "_tmp_eval")
        os.makedirs(tmp_dir, exist_ok=True)

        eval_kwargs = dict(
            dataset_path=dataset_path,
            baseline_path=None,
            seqrfm_path=None,
            out_dir=tmp_dir,
            model_name=args.model,
            alpha_grid=args.alpha_grid,
            judge_model=args.judge_model,
            max_new_tokens=args.eval_gen_tokens,
            cache_dir=args.cache_dir,
            seed=args.seed,
            completion_rfm_path=completion_rfm_path,
            completion_diff_means_path=completion_dm_path,
            bag_of_words_path=None,
            bag_of_words_diff_means_path=None,
        )
        evaluator.run_evaluation(**eval_kwargs)

        for filename in ("summary.csv", "per_example_results.csv"):
            src  = os.path.join(tmp_dir, filename)
            dest = os.path.join(strategy_dir, filename)
            if not os.path.exists(src):
                continue
            with open(src, newline="") as in_f:
                reader = csv.reader(in_f)
                header = next(reader)
                rows   = list(reader)
            if os.path.exists(dest):
                with open(dest, "a", newline="") as out_f:
                    csv.writer(out_f).writerows(rows)
            else:
                with open(dest, "w", newline="") as out_f:
                    w = csv.writer(out_f)
                    w.writerow(header)
                    w.writerows(rows)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # CLAS (optional — uses teacher completions as controller targets)
    # ------------------------------------------------------------------
    # Free GPU memory again before CLAS loads its model
    gc.collect()
    _torch.cuda.empty_cache()

    if args.run_clas:
        print(f"\n[CLAS] train + evaluate (teacher completions as controller targets)")
        clas_out_dir = os.path.join(concept_dir, f"clas_{STRATEGY_TAG}")
        os.makedirs(clas_out_dir, exist_ok=True)
        import clas as clas_module
        import torch
        import csv as csv_mod
        from utils import load_model

        clas_path = os.path.join(clas_out_dir, "clas_directions.pt")
        # baseline_directions.pkl comes from the MAIN experiment output (self-completions),
        # not the teacher ablation dir. Pass --baseline_dir to point at it.
        baseline_path = os.path.join(
            args.baseline_dir, f"{concept_key}_{slug}", "baseline_directions.pkl"
        ) if args.baseline_dir else None
        if not baseline_path or not os.path.exists(baseline_path):
            print(f"  WARNING: baseline_directions.pkl not found"
                  f"{f' at {baseline_path}' if baseline_path else ' (--baseline_dir not set)'}. "
                  f"CLAS needs the prompt-RFM direction -- skipping CLAS for {concept_name}.")
            baseline_path = None
        if baseline_path:
            if not os.path.exists(clas_path):
                clas_module.train_clas(
                    teacher_dataset_path, baseline_path, clas_path, args.model,
                    val_frac=args.clas_val_frac, lr_weight=args.clas_lr_weight,
                    lr_bias=args.clas_lr_bias, epochs=args.clas_epochs,
                    accum_steps=args.clas_accum_steps, patience=args.clas_patience,
                    cache_dir=args.cache_dir, seed=args.seed,
                )
            else:
                print(f"  {clas_path} exists, skipping training.")

            # Evaluate
            clas_summary_path = os.path.join(clas_out_dir, "summary.csv")
            if os.path.exists(clas_summary_path):
                print(f"  {clas_summary_path} exists, skipping CLAS eval.")
            else:
                saved = torch.load(clas_path)
                d_dirs   = saved["d_directions_per_layer"]
                c_weight = saved["c_weight_per_layer"]
                c_bias   = saved["c_bias_per_layer"]
                with open(dataset_path) as f:
                    ds = json.load(f)
                test_examples = ds["test"]

                if concept_key == "persona":
                    from persona_judge import judge_response
                else:
                    from fear_judge import judge_response

                target = ds.get(concept_key, ds.get("concept", concept_name))
                lm, tokenizer = load_model(args.model, cache_dir=args.cache_dir)
                lm.eval()

                rows_path    = os.path.join(clas_out_dir, "per_example_results.csv")
                summary_path = os.path.join(clas_out_dir, "summary.csv")
                scores = []
                with open(rows_path, "w", newline="") as f:
                    writer = csv_mod.writer(f)
                    writer.writerow([concept_key, "method", "alpha", "question",
                                      "response", "score", "explanation"])
                    from tqdm import tqdm
                    for ex in tqdm(test_examples, desc="CLAS eval"):
                        response = clas_module.generate_with_clas(
                            lm, tokenizer, ex["prompt"], d_dirs, c_weight, c_bias,
                            args.eval_gen_tokens
                        )
                        score, explanation = judge_response(target, ex.get("question", ex["prompt"]),
                                                            response, args.judge_model)
                        writer.writerow([target, "clas_teacher", 0.0,
                                         ex.get("question", ""), response, score, explanation])
                        scores.append(score)
                mean_score = sum(scores) / len(scores) if scores else 0.0
                with open(summary_path, "w", newline="") as f:
                    writer = csv_mod.writer(f)
                    writer.writerow([concept_key, "method", "alpha", "mean_score", "n"])
                    writer.writerow([target, "clas_teacher", 0.0, mean_score, len(scores)])
                print(f"  CLAS (teacher) mean_score: {mean_score:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation_manifest",              required=True)
    p.add_argument("--teacher_datasets_dir",           required=True)
    p.add_argument("--category",                       required=True,
                   choices=["persona", "fear"])
    p.add_argument("--model",                          required=True)
    p.add_argument("--model_tag",                      required=True,
                   help="short tag matching format_teacher_datasets.py output (e.g. 'llama', 'qwen')")
    p.add_argument("--out_dir",                        required=True)
    p.add_argument("--alpha_grid",                     type=float, nargs="+",
                   default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    p.add_argument("--rfm_iters",                      type=int, default=8)
    p.add_argument("--eval_gen_tokens",                type=int, default=100)
    p.add_argument("--judge_model",                    default="gpt-4o-mini")
    p.add_argument("--cache_dir",                      default=None)
    p.add_argument("--seed",                           type=int, default=42)
    # completion_rfm_controller is now integrated directly via
    # run_completion_rfm_controller_pipeline.py + train_controller.py
    p.add_argument("--run_joint",                      action="store_true",
                   help="run joint training (train_joint_teacher.py)")
    p.add_argument("--joint_epochs",                   type=int,   default=8)
    p.add_argument("--joint_lr",                       type=float, default=3e-3)
    p.add_argument("--joint_dir_lr",                   type=float, default=1e-4)
    p.add_argument("--joint_accum_steps",              type=int,   default=10)
    p.add_argument("--joint_val_frac",                 type=float, default=0.15)
    p.add_argument("--joint_patience",                 type=int,   default=2)
    p.add_argument("--baseline_dir",                   default=None,
                   help="root of main experiment outputs containing "
                        "<concept_key>_<slug>/baseline_directions.pkl "
                        "(needed for CLAS, which reuses the prompt-RFM direction)")
    p.add_argument("--run_clas",                       action="store_true")
    p.add_argument("--clas_val_frac",                  type=float, default=0.2)
    p.add_argument("--clas_lr_weight",                 type=float, default=3e-3)
    p.add_argument("--clas_lr_bias",                   type=float, default=1e-1)
    p.add_argument("--clas_epochs",                    type=int,   default=8)
    p.add_argument("--clas_accum_steps",               type=int,   default=10)
    p.add_argument("--clas_patience",                  type=int,   default=2)
    args = p.parse_args()

    with open(args.ablation_manifest) as f:
        ablation_manifest = json.load(f)

    category_key   = "personas" if args.category == "persona" else "fears"
    concept_key    = args.category  # "persona" or "fear"
    category_dir   = os.path.join(args.teacher_datasets_dir,
                                  category_key, args.model_tag)

    for entry in ablation_manifest[category_key]:
        concept_name   = entry[concept_key]
        dataset_path   = entry["path"]
        slug = concept_name.lower().replace(" ", "_").replace(".", "").replace(",", "")
        teacher_dataset_path = os.path.join(
            category_dir, f"{slug}_teacher_{args.model_tag}.json"
        )
        if not os.path.exists(teacher_dataset_path):
            print(f"SKIPPING {concept_name}: teacher dataset not found at "
                  f"{teacher_dataset_path}")
            continue

        out_dir = os.path.join(args.out_dir, args.model_tag)
        run_one_concept(concept_name, concept_key, dataset_path,
                        teacher_dataset_path, out_dir, args)

    # Combined summary — collects all methods across all per-concept subdirs
    import csv
    combined_path = os.path.join(
        args.out_dir, args.model_tag,
        f"combined_summary_teacher_{args.category}.csv"
    )
    with open(combined_path, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow([concept_key, "method", "alpha", "mean_score", "n"])
        for entry in ablation_manifest[category_key]:
            concept_name = entry[concept_key]
            slug = concept_name.lower().replace(" ", "_").replace(".", "").replace(",", "")
            concept_dir_c = os.path.join(args.out_dir, args.model_tag, f"{concept_key}_{slug}")
            summary_paths = [
                # completion_rfm + completion_diff_means
                os.path.join(concept_dir_c, STRATEGY_TAG, "summary.csv"),
                # contextual completion RFM controller
                os.path.join(concept_dir_c, STRATEGY_TAG, "completion_rfm_controller_eval", "summary.csv"),
                # CLAS
                os.path.join(concept_dir_c, f"clas_{STRATEGY_TAG}", "summary.csv"),
                # joint training
                os.path.join(concept_dir_c, f"joint_{STRATEGY_TAG}", "summary.csv"),
            ]
            for summary in summary_paths:
                if not os.path.exists(summary):
                    continue
                with open(summary) as in_f:
                    reader = csv.reader(in_f)
                    next(reader)
                    for row in reader:
                        writer.writerow(row)
    print(f"\nCombined summary -> {combined_path}")


if __name__ == "__main__":
    main()