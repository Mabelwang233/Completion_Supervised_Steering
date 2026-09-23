"""
generate_teacher_completions.py

Stage 1 of the teacher-completion ablation:
  - Generates canonical teacher completions once per unique prompt
  - Caches results persistently (keyed by teacher model + exact input)
  - Supports dry-run mode, resumability, exponential backoff
  - Writes a raw canonical JSON (concept-agnostic, model-agnostic)

Stage 2 (format_teacher_datasets.py) then builds Llama- and Qwen-formatted
datasets from this canonical file without any further API calls.

Usage:
    # Dry run:
    python generate_teacher_completions.py --dry_run ...

    # Full run:
    python generate_teacher_completions.py \
        --ablation_manifest  teacher_ablation/ablation_manifest.json \
        --out_path           teacher_ablation/canonical_teacher_completions.json \
        --cache_path         teacher_ablation/teacher_cache.json \
        --teacher_model      o3-mini \
        --api_max_tokens     1000 \
        --reasoning_effort   low \
        --max_retries        4 \
        --seed               42

Output: teacher_ablation/canonical_teacher_completions.json
    {
      "<exact_prompt_text>": {
          "completion": "...",
          "teacher_model": "o3-mini",
          "reasoning_effort": "low",
          "api_max_tokens": 1000,
          "n_retries": 0,
          "timestamp": "..."
      },
      ...
    }
"""
import argparse
import json
import os
import time
import datetime

from openai import OpenAI, BadRequestError, RateLimitError
from tqdm import tqdm


def get_client():
    return OpenAI()


def call_teacher(client, prompt_text, teacher_model, api_max_tokens,
                 reasoning_effort, max_retries):
    budget = api_max_tokens
    for attempt in range(max_retries + 1):
        try:
            kwargs = dict(
                model=teacher_model,
                messages=[{"role": "user", "content": prompt_text}],
                max_completion_tokens=budget,
            )
            if "o3" in teacher_model or "o1" in teacher_model:
                kwargs["reasoning_effort"] = reasoning_effort
            else:
                kwargs["temperature"] = 0.0
            completion = client.chat.completions.create(**kwargs)
            text = completion.choices[0].message.content or ""
            if text.strip():
                return text.strip(), attempt
            budget = budget * 2
        except BadRequestError as e:
            if "output limit" in str(e).lower() and attempt < max_retries:
                budget = budget * 2
                time.sleep(2 ** attempt)
                continue
            raise
        except RateLimitError:
            wait = 2 ** (attempt + 2)
            print(f"  Rate limit hit, waiting {wait}s...")
            time.sleep(wait)
    return "", max_retries


def collect_unique_prompts(ablation_manifest):
    """Collect all unique prompt strings across all selected concepts."""
    unique = {}  # prompt_text -> list of (category, concept_name, label, idx)
    for category in ("personas", "fears"):
        key = "persona" if category == "personas" else "fear"
        for entry in ablation_manifest[category]:
            concept_name = entry[key]
            dataset_path = entry["path"]
            with open(dataset_path) as f:
                dataset = json.load(f)
            for i, ex in enumerate(dataset["train"]):
                prompt_text = ex["prompt"]
                if prompt_text not in unique:
                    unique[prompt_text] = []
                unique[prompt_text].append((category, concept_name, ex["label"], i))
    return unique


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation_manifest", required=True)
    p.add_argument("--out_path",          required=True,
                   help="canonical teacher completions JSON")
    p.add_argument("--cache_path",        required=True,
                   help="persistent cache (same format as out_path, written after every completion)")
    p.add_argument("--teacher_model",     default="o3-mini")
    p.add_argument("--api_max_tokens",    type=int, default=1000)
    p.add_argument("--reasoning_effort",  default="low",
                   choices=["low", "medium", "high"])
    p.add_argument("--train_prompt_suffix", default=None,
                   help="appended to every prompt before sending to teacher "
                        "(e.g. 'Answer in 100 words or less.')")
    p.add_argument("--max_retries",       type=int, default=4)
    p.add_argument("--dry_run",           action="store_true",
                   help="report stats without making any API calls")
    p.add_argument("--seed",              type=int, default=42)
    args = p.parse_args()

    with open(args.ablation_manifest) as f:
        ablation_manifest = json.load(f)

    print("Collecting unique prompts across all concepts...")
    unique_prompts = collect_unique_prompts(ablation_manifest)
    print(f"  Total training rows : "
          f"{sum(len(v) for v in unique_prompts.values())}")
    print(f"  Unique prompts      : {len(unique_prompts)}")

    # Load persistent cache
    cache = {}
    if os.path.exists(args.cache_path):
        with open(args.cache_path) as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache)} cached completions from {args.cache_path}")

    # Build teacher inputs (optionally append suffix)
    def teacher_input(prompt_text):
        if args.train_prompt_suffix:
            return f"{prompt_text} {args.train_prompt_suffix}"
        return prompt_text

    cache_key = lambda pt: f"{args.teacher_model}|{args.reasoning_effort}|{teacher_input(pt)}"

    todo = [pt for pt in unique_prompts if cache_key(pt) not in cache]
    cached = len(unique_prompts) - len(todo)

    print(f"\n  Already cached      : {cached}")
    print(f"  API calls needed    : {len(todo)}")

    if args.dry_run:
        print("\nDry run — no API calls made.")
        return

    if not todo:
        print("All prompts already cached.")
    else:
        client = get_client()
        os.makedirs(os.path.dirname(args.cache_path) or ".", exist_ok=True)

        n_empty = 0
        n_retried = 0
        for prompt_text in tqdm(todo, desc="Generating teacher completions"):
            inp = teacher_input(prompt_text)
            completion, n_attempts = call_teacher(
                client, inp, args.teacher_model, args.api_max_tokens,
                args.reasoning_effort, args.max_retries
            )
            if n_attempts > 0:
                n_retried += 1
            if not completion:
                n_empty += 1
                print(f"  WARNING: empty completion for: {prompt_text[:80]}...")

            cache[cache_key(prompt_text)] = {
                "completion": completion,
                "teacher_model": args.teacher_model,
                "reasoning_effort": args.reasoning_effort,
                "api_max_tokens": args.api_max_tokens,
                "train_prompt_suffix": args.train_prompt_suffix,
                "n_retries": n_attempts,
                "timestamp": datetime.datetime.utcnow().isoformat(),
            }
            # Checkpoint after every completion
            with open(args.cache_path, "w") as f:
                json.dump(cache, f, indent=2)

        print(f"\nGeneration complete.")
        print(f"  Retried            : {n_retried}/{len(todo)}")
        print(f"  Empty completions  : {n_empty}/{len(todo)}")

    # Build canonical output (prompt_text -> completion record)
    canonical = {}
    for prompt_text in unique_prompts:
        key = cache_key(prompt_text)
        if key in cache:
            canonical[prompt_text] = cache[key]
        else:
            canonical[prompt_text] = {"completion": "", "error": "not_generated"}

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(canonical, f, indent=2)
    print(f"\nWrote canonical completions -> {args.out_path}")

    # Validation
    n_empty_final = sum(1 for v in canonical.values() if not v["completion"].strip())
    print(f"Validation: {len(canonical)} total unique prompts, "
          f"{n_empty_final} empty completions.")
    if n_empty_final > 0:
        print("  WARNING: raise --api_max_tokens or check the teacher model.")


if __name__ == "__main__":
    main()
