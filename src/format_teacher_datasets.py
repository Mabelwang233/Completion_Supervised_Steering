"""
format_teacher_datasets.py

Stage 2 of the teacher-completion ablation.
Takes the canonical teacher completions (from generate_teacher_completions.py)
and builds target-model-specific dataset files for Llama and Qwen.

No API calls are made here. Only the target tokenizer is loaded (no GPU).

Produces one dataset file per (category, concept, target_model):
    <out_dir>/<category>/<concept_slug>_<model_tag>.json

Each dataset mirrors the schema of dataset_with_completions.json:
    {
      "concept": ...,
      "type": ...,
      "train": [
          {
            "prompt": ...,
            "label": ...,
            "statement": ...,
            "completion": "<teacher text>",
            "formatted_prompt": "<target model chat template>",
          },
          ...
      ],
      "test": [...]
    }

Usage:
    python format_teacher_datasets.py \
        --ablation_manifest  teacher_ablation/ablation_manifest.json \
        --canonical_completions teacher_ablation/canonical_teacher_completions.json \
        --out_dir            teacher_ablation/datasets \
        --models             meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct \
        --model_tags         llama qwen \
        --cache_dir          /path/to/hf_cache
"""
import argparse
import json
import os

from transformers import AutoTokenizer


def make_model_tag(model_name):
    """Derive a short filesystem-safe tag from a model name."""
    return model_name.split("/")[-1].lower().replace("-", "_").replace(".", "_")


def format_chat(tokenizer, prompt_text):
    """Model-agnostic chat formatting using the tokenizer's own template."""
    chat = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def build_dataset_for_model(dataset_path, canonical, tokenizer, model_name):
    with open(dataset_path) as f:
        dataset = json.load(f)

    n_missing = 0
    n_empty = 0
    new_train = []
    for ex in dataset["train"]:
        prompt_text = ex["prompt"]
        record = canonical.get(prompt_text, {})
        completion = record.get("completion", "")
        if prompt_text not in canonical:
            n_missing += 1
        if not completion.strip():
            n_empty += 1
        new_ex = dict(ex)
        new_ex["completion"] = completion
        new_ex["formatted_prompt"] = format_chat(tokenizer, prompt_text)
        new_ex["completion_source"] = "teacher"
        new_ex["teacher_model"] = record.get("teacher_model", "unknown")
        new_train.append(new_ex)

    out_dataset = dict(dataset)
    out_dataset["train"] = new_train
    out_dataset["completion_source"] = "teacher"
    out_dataset["target_model"] = model_name

    return out_dataset, n_missing, n_empty


def validate_dataset(dataset, concept_name):
    label1 = [ex for ex in dataset["train"] if ex["label"] == 1]
    label0 = [ex for ex in dataset["train"] if ex["label"] == 0]
    empty1 = [ex for ex in label1 if not ex.get("completion", "").strip()]
    empty0 = [ex for ex in label0 if not ex.get("completion", "").strip()]
    ok = True
    if len(label1) != 200:
        print(f"  WARN [{concept_name}]: expected 200 label-1, got {len(label1)}")
        ok = False
    if len(label0) != 200:
        print(f"  WARN [{concept_name}]: expected 200 label-0, got {len(label0)}")
        ok = False
    if empty1:
        print(f"  WARN [{concept_name}]: {len(empty1)} empty label-1 completions")
        ok = False
    if empty0:
        print(f"  WARN [{concept_name}]: {len(empty0)} empty label-0 completions")
        ok = False
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation_manifest",      required=True)
    p.add_argument("--canonical_completions",  required=True)
    p.add_argument("--out_dir",                required=True)
    p.add_argument("--models",                 nargs="+",
                   default=["meta-llama/Llama-3.1-8B-Instruct",
                             "Qwen/Qwen2.5-7B-Instruct"])
    p.add_argument("--model_tags",             nargs="+", default=None,
                   help="short tags for filenames; defaults to auto-derived from model names")
    p.add_argument("--cache_dir",              default=None)
    args = p.parse_args()

    model_tags = args.model_tags or [make_model_tag(m) for m in args.models]
    assert len(model_tags) == len(args.models), \
        "--model_tags must have the same length as --models"

    with open(args.ablation_manifest) as f:
        ablation_manifest = json.load(f)
    with open(args.canonical_completions) as f:
        canonical = json.load(f)

    print(f"Loaded {len(canonical)} canonical completions.")

    # Load tokenizers once per model
    tokenizers = {}
    for model_name in args.models:
        print(f"Loading tokenizer: {model_name}")
        tokenizers[model_name] = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=args.cache_dir
        )

    all_ok = True
    for category, key in [("personas", "persona"), ("fears", "fear")]:
        for entry in ablation_manifest[category]:
            concept_name = entry[key]
            dataset_path = entry["path"]
            slug = concept_name.lower().replace(" ", "_").replace(".", "").replace(",", "")

            for model_name, model_tag in zip(args.models, model_tags):
                tokenizer = tokenizers[model_name]
                out_dataset, n_missing, n_empty = build_dataset_for_model(
                    dataset_path, canonical, tokenizer, model_name
                )

                ok = validate_dataset(out_dataset, f"{concept_name}/{model_tag}")
                if not ok:
                    all_ok = False

                out_dir = os.path.join(args.out_dir, category, model_tag)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{slug}_teacher_{model_tag}.json")
                with open(out_path, "w") as f:
                    json.dump(out_dataset, f, indent=2)

                status = "OK" if (n_missing == 0 and n_empty == 0) else f"WARN(miss={n_missing},empty={n_empty})"
                print(f"  [{status}] {concept_name} / {model_tag} -> {out_path}")

    if all_ok:
        print("\nAll datasets validated successfully.")
    else:
        print("\nWARNING: some datasets have missing or empty completions.")


if __name__ == "__main__":
    main()
