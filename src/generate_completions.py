"""
Step 2: generate completions z_i for every training prompt, using the
SAME model we'll later steer (not GPT-4o -- see the earlier discussion
on why: we want trajectories that reflect the target model's own
continuation dynamics, since that's what we intervene on at test time).

Adds a "completion" field to each training example in the dataset JSON.
Requires a GPU -- not runnable in a CPU-only sandbox.

train_prompt_suffix (optional): appended to each training prompt ONLY
for the text actually sent to the model for generation (e.g.
"Answer in 15 words or less." for the short_instructed_cut strategy).
The dataset's original ex["prompt"] field is left untouched -- baselines.py
reads that field directly and should stay identical across strategies,
since mean_difference/rfm never touch completions at all. The suffix IS
visible in ex["formatted_prompt"] (the full chat-templated text actually
used), so it's fully recoverable/inspectable from the output file.
"""
import argparse
import json
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, ".")
from utils import load_model


def generate_completions(dataset_path, out_path, model_name, cache_dir=None,
                          max_new_tokens=150, batch_size=8, seed=0,
                          train_prompt_suffix=None):
    torch.manual_seed(seed)

    with open(dataset_path) as f:
        dataset = json.load(f)

    language_model, tokenizer = load_model(model_name, cache_dir=cache_dir)
    language_model.eval()

    vocab = tokenizer.get_vocab()
    if '<|start_header_id|>' in vocab:
        assistant_tag = '<|start_header_id|>assistant<|end_header_id|>'
    elif '<|im_start|>' in vocab:
        assistant_tag = '<|im_start|>assistant\n'
    else:
        assistant_tag = ''

    def format_chat(prompt_text):
        chat = [{"role": "user", "content": prompt_text}]
        formatted = tokenizer.apply_chat_template(chat, tokenize=False)
        return formatted + assistant_tag

    train_examples = dataset["train"]
    print(f"Generating completions for {len(train_examples)} training prompts...")
    if train_prompt_suffix:
        print(f"  (appending suffix for generation only: {train_prompt_suffix!r})")

    for start in tqdm(range(0, len(train_examples), batch_size)):
        batch = train_examples[start:start + batch_size]
        gen_input_texts = [
            f"{ex['prompt']} {train_prompt_suffix}" if train_prompt_suffix else ex["prompt"]
            for ex in batch
        ]
        formatted_prompts = [format_chat(t) for t in gen_input_texts]

        inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True,
                            add_special_tokens=False).to(language_model.device)

        with torch.no_grad():
            output_ids = language_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for i, ex in enumerate(batch):
            input_len = inputs["input_ids"].shape[1]
            gen_ids = output_ids[i][input_len:]
            completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
            ex["completion"] = completion
            ex["formatted_prompt"] = formatted_prompts[i]  # includes the suffix, if any -- ex["prompt"] itself does not

    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", default="../data/shakespeare_dataset.json")
    p.add_argument("--out_path", default="../data/shakespeare_dataset_with_completions.json")
    p.add_argument("--model", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--max_new_tokens", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_prompt_suffix", default=None,
                    help='e.g. "Answer in 15 words or less." -- appended for generation only')
    args = p.parse_args()

    generate_completions(args.dataset_path, args.out_path, args.model,
                          args.cache_dir, args.max_new_tokens, args.batch_size, args.seed,
                          train_prompt_suffix=args.train_prompt_suffix)
