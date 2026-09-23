"""
Teacher-model completions: generates z_i for training prompts using a
stronger external model (default: o3-mini) via the OpenAI API, instead
of the target model's own generations. Ablation companion to
generate_completions.py -- same output schema, everything downstream
(extract_trajectories, baselines, sequence_rfm, etc.) is unchanged.

Applied to BOTH label-0 and label-1 prompts -- not just label-1 -- so
the learned direction can't partly be detecting "which model wrote
this" instead of the actual concept. Both classes share the same
completion source, only the (x,y) label differs.

o3-mini specifics (reasoning model, genuinely different from a normal
chat model -- confirmed via search, not assumed):
  - uses `max_completion_tokens`, NOT `max_tokens` (the old param
    errors out entirely on o-series models).
  - does NOT support `temperature` (omitted entirely below, not set to 0
    -- some reports show it erroring if passed).
  - `max_completion_tokens` caps HIDDEN REASONING tokens and visible
    output tokens TOGETHER, and reasoning alone commonly consumes
    hundreds of tokens even for simple prompts. If the budget runs out
    mid-reasoning, the API does NOT return a short/empty answer -- it
    raises a hard 400 BadRequestError ("Could not finish the message
    because max_tokens or model output limit was reached"). This means
    a SMALL budget (e.g. matching a 15-token target completion length)
    is not just occasionally insufficient -- it fails on nearly every
    call.

  Because of this, `api_max_completion_tokens` (the actual request
  budget, generous, covers reasoning+output) is DELIBERATELY decoupled
  from `max_new_tokens` / the length instruction (which controls the
  intended VISIBLE answer length, via the prompt -- same mechanism the
  local-model version uses). Don't try to use a small budget to force a
  short visible answer out of a reasoning model; it will just fail the
  request. Let the instruction do that job instead.

formatted_prompt is still built with the TARGET model's own tokenizer
and chat template (NOT the teacher's) -- extract_trajectories.py
teacher-forces (formatted_prompt + completion) through the TARGET model
to get ITS activations, so the prompt encoding must match what the
target model would see, regardless of which model wrote the completion.
Only the target model's tokenizer is loaded here, not its weights --
this step needs no GPU.
"""
import argparse
import json

from openai import OpenAI, BadRequestError
from tqdm import tqdm
from transformers import AutoTokenizer

ASSISTANT_TAG = '<|start_header_id|>assistant<|end_header_id|>'

_client = None


def get_client():
    global _client
    if _client is None:
        _client = OpenAI()  # reads OPENAI_API_KEY from env
    return _client


def call_teacher(prompt_text, teacher_model, api_max_completion_tokens, reasoning_effort="low", max_retries=4):
    """Returns (completion_text, n_retries_used). completion_text is ""
    if every attempt (including retries) still failed."""
    client = get_client()
    budget = api_max_completion_tokens
    for attempt in range(max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=teacher_model,
                messages=[{"role": "user", "content": prompt_text}],
                max_completion_tokens=budget,
                reasoning_effort=reasoning_effort,
            )
        except BadRequestError as e:
            if "output limit" in str(e).lower() and attempt < max_retries:
                budget = budget * 2  # reasoning ate the whole budget -- give more room, retry
                continue
            raise  # a different 400 error -- don't silently swallow it
        text = completion.choices[0].message.content or ""
        if text.strip():
            return text, attempt
        budget = budget * 2
    return "", max_retries


def generate_completions_teacher(dataset_path, out_path, target_model_name, teacher_model="o3-mini",
                                  max_new_tokens=150, train_prompt_suffix=None,
                                  reasoning_effort="low", api_max_completion_tokens=1000, seed=0):
    """
    max_new_tokens: kept for interface parity with generate_completions.py
    and used only for logging -- it does NOT set the API request budget
    for a reasoning-model teacher (see module docstring for why). Use
    api_max_completion_tokens for that, and train_prompt_suffix (e.g.
    "Answer in 15 words or less.") to actually control visible length.
    """
    with open(dataset_path) as f:
        dataset = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(target_model_name)

    def format_chat(prompt_text):
        chat = [{"role": "user", "content": prompt_text}]
        formatted = tokenizer.apply_chat_template(chat, tokenize=False)
        return formatted + ASSISTANT_TAG

    train_examples = dataset["train"]
    print(f"Generating completions for {len(train_examples)} training prompts via {teacher_model} "
          f"(target model tokenizer: {target_model_name})...")
    print(f"  API request budget (reasoning+output): {api_max_completion_tokens} tokens, "
          f"escalated on retry if exceeded")
    if train_prompt_suffix:
        print(f"  visible-length instruction (generation only): {train_prompt_suffix!r}")

    n_empty = 0
    n_retried = 0
    for ex in tqdm(train_examples):
        gen_input_text = f"{ex['prompt']} {train_prompt_suffix}" if train_prompt_suffix else ex["prompt"]
        completion, n_attempts = call_teacher(gen_input_text, teacher_model, api_max_completion_tokens,
                                               reasoning_effort)
        if n_attempts > 0:
            n_retried += 1
        if not completion.strip():
            n_empty += 1
        ex["completion"] = completion
        # target model's own template -- NOT the teacher's -- see module docstring
        ex["formatted_prompt"] = format_chat(ex["prompt"])

    if n_retried:
        print(f"  {n_retried}/{len(train_examples)} examples needed a retry with a larger token budget "
              f"(reasoning tokens likely consumed the initial budget).")
    if n_empty:
        print(f"  WARNING: {n_empty}/{len(train_examples)} completions are STILL EMPTY after retries -- "
              f"these examples will contribute no signal to training. Consider raising "
              f"--teacher_api_max_completion_tokens.")

    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--out_path", required=True)
    p.add_argument("--target_model", required=True,
                    help="the model being steered -- used ONLY for its tokenizer/chat template, not loaded as weights")
    p.add_argument("--teacher_model", default="o3-mini")
    p.add_argument("--max_new_tokens", type=int, default=150,
                    help="logging only -- does not set the API budget for a reasoning-model teacher, "
                         "see module docstring")
    p.add_argument("--api_max_completion_tokens", type=int, default=1000,
                    help="actual API request budget -- must cover hidden reasoning tokens plus visible "
                         "output. Keep this generous; control visible length via --train_prompt_suffix instead.")
    p.add_argument("--train_prompt_suffix", default=None)
    p.add_argument("--reasoning_effort", default="low", choices=["low", "medium", "high"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    generate_completions_teacher(args.dataset_path, args.out_path, args.target_model, args.teacher_model,
                                  args.max_new_tokens, args.train_prompt_suffix, args.reasoning_effort,
                                  args.api_max_completion_tokens, args.seed)
