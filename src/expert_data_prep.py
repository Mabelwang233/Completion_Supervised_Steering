"""
Per-concept (x, y) data builder for the Experts class, mirroring
persona_data_prep.py / fear_data_prep.py: label 1 is every statement
WITH the expert prefix, label 0 is every statement WITHOUT any prefix,
so the only distinguishing signal is the prefix.

--n_statements picks the statement pool:
  400 (default, matches data_400x2) -- class_1 + class_0, 800 train examples
  200 (matches data_200x2)          -- class_1 only,      400 train examples

--pairing split (only with --n_statements 200) makes the classes unpaired
("200/200"): label 1 = class_1 statements WITH the prefix, label 0 =
class_0 statements WITHOUT it, so no statement appears in both classes.

The concept list is a seeded random sample from experts.txt
(--sample_n/--sample_seed), written to --sample_out so the selection is
fixed and reusable. Pass --sample_n 0 to use every line of --experts_path
as-is (e.g. an existing expert_30.txt).

Test set: 20 fixed expert-eliciting questions. The first 5 are from the
previous paper; the remaining 15 are new.
"""
import json
import os
import random


CLASS_1_PREFIX_TEMPLATE = "Take on the role of an expert {expert}."
QUESTION_TEMPLATE = "What are your thoughts on the following statement?\nStatement: {statement}"

TWENTY_EXPERT_QUESTIONS = [
    # from the previous paper
    "What is your favorite subject?",
    "What's the kind of problem that makes you forget to eat or sleep?",
    "If you had unlimited resources, what would you work on?",
    "Describe your work in a way a 5-year-old can understand.",
    "What's a question that got you hooked?",
    # new
    "What does a typical day at work look like for you?",
    "What tools or instruments could you not do your job without?",
    "What's something ordinary that you see completely differently because of your training?",
    "What's the biggest misconception people have about what you do?",
    "What's an unsolved problem you think about the most?",
    "What's the most exciting recent development in your area?",
    "What's a debate in your field you have strong opinions about?",
    "How do you know when you've gotten something right in your work?",
    "What skill took you the longest to master?",
    "What book would you recommend to someone who wants to think like you?",
    "What advice would you give a student who wants to do what you do?",
    "What would you put in a museum exhibit about your work?",
    "What news headline would make your whole week?",
    "If you could travel to any time or place for your research, where would you go?",
    "What's the first thing you notice when you walk into a new place?",
]
LENGTH_SUFFIX = "Answer in 20 words or less."


def load_statements(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_experts(path):
    return load_statements(path)


def sample_experts(experts_path, n, seed, out_path):
    """Seeded sample of n experts, kept in experts.txt order, written to out_path."""
    all_experts = load_experts(experts_path)
    if len(set(all_experts)) != len(all_experts):
        raise ValueError(f"duplicate entries in {experts_path}")
    if n > len(all_experts):
        raise ValueError(f"sample_n={n} exceeds {len(all_experts)} experts in {experts_path}")
    chosen = set(random.Random(seed).sample(all_experts, n))
    experts = [e for e in all_experts if e in chosen]
    with open(out_path, "w") as f:
        f.write("\n".join(experts) + "\n")
    print(f"Sampled {n}/{len(all_experts)} experts (seed={seed}) -> {out_path}")
    return experts


def safe_name(expert):
    return expert.lower().replace(" ", "_").replace("(", "").replace(")", "")


def build_concept_dataset(expert, class_0_statements, class_1_statements, n_statements=400,
                          pairing="paired"):
    """
    paired: every statement in the pool is used twice:
      label 1: statement WITH the expert prefix
      label 0: statement WITHOUT any prefix (plain)
    split (n_statements=200 only): label 1 = class_1 WITH prefix,
      label 0 = class_0 WITHOUT prefix -- disjoint statements.
    """
    if pairing == "split":
        if n_statements != 200:
            raise ValueError("pairing='split' requires n_statements=200")
        label_1_statements, label_0_statements = class_1_statements, class_0_statements
    elif n_statements == 400:
        label_1_statements = label_0_statements = class_1_statements + class_0_statements
    elif n_statements == 200:
        label_1_statements = label_0_statements = class_1_statements
    else:
        raise ValueError(f"n_statements must be 200 or 400, got {n_statements}")
    label_1_prefix = CLASS_1_PREFIX_TEMPLATE.format(expert=expert)

    train_examples = []
    for s in label_1_statements:
        prompt = f"{label_1_prefix} {QUESTION_TEMPLATE.format(statement=s)}"
        train_examples.append({"statement": s, "label": 1, "prompt": prompt})
    for s in label_0_statements:
        prompt = QUESTION_TEMPLATE.format(statement=s)
        train_examples.append({"statement": s, "label": 0, "prompt": prompt})

    test_examples = [
        {"question": q, "prompt": f"{q} {LENGTH_SUFFIX}"}
        for q in TWENTY_EXPERT_QUESTIONS
    ]

    # "type" lets evaluate.py pick expert_judge without guessing from keys
    return {"expert": expert, "type": "expert", "train": train_examples, "test": test_examples}


def build_all_concept_datasets(experts, data_dir, out_dir, n_statements=400, manifest_name="manifest.json",
                               pairing="paired"):
    class_0 = load_statements(os.path.join(data_dir, "class_0.txt"))
    class_1 = load_statements(os.path.join(data_dir, "class_1.txt"))
    assert len(class_0) == 200 and len(class_1) == 200

    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for expert in experts:
        dataset = build_concept_dataset(expert, class_0, class_1, n_statements, pairing)
        out_path = os.path.abspath(os.path.join(out_dir, f"expert_{safe_name(expert)}.json"))
        with open(out_path, "w") as f:
            json.dump(dataset, f, indent=2)
        manifest.append({"expert": expert, "path": out_path})
        print(f"  {expert}: wrote {out_path}")

    manifest_path = os.path.join(out_dir, manifest_name)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(experts)} concept datasets ({n_statements} statements x 2 labels, "
          f"{pairing} = {n_statements * 2} train examples each). Manifest: {manifest_path}")
    return manifest


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--experts_path", default="../data/experts.txt")
    p.add_argument("--sample_n", type=int, default=30,
                    help="randomly sample this many experts from --experts_path; 0 = use all lines as-is")
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument("--sample_out", default="../data/expert_30.txt",
                    help="where to write the sampled expert list")
    p.add_argument("--data_dir", default="../data", help="directory with class_0.txt / class_1.txt")
    p.add_argument("--out_dir", default="../data_400x2/experts")
    p.add_argument("--n_statements", type=int, default=400, choices=[200, 400])
    p.add_argument("--manifest_name", default="manifest30.json")
    p.add_argument("--pairing", default="paired", choices=["paired", "split"])
    args = p.parse_args()

    if args.sample_n > 0:
        experts = sample_experts(args.experts_path, args.sample_n, args.sample_seed, args.sample_out)
    else:
        experts = load_experts(args.experts_path)
    build_all_concept_datasets(experts, args.data_dir, args.out_dir, args.n_statements, args.manifest_name,
                               args.pairing)
