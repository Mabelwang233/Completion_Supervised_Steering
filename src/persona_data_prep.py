"""
Per-concept (x, y) data builder for the Personas class, mirroring
fear_data_prep.py's structure exactly: disjoint statement pools (first
200 -> label 1, last 200 -> label 0), Fig 4A style.

One caveat worth being explicit about: the paper's Fig 4A example is
literally for the Fears class ("Personify someone who is terrified of
{fear}."). There's no equivalent explicit figure for Personas' label-1
prefix wording in what we have. The prefix below ("Personify {persona}.")
is the natural analogous construction following the same pattern, not
something confirmed verbatim from the paper -- flagging this the same
way we flagged the Shakespeare prefix uncertainty earlier.
"""
import json
import os


CLASS_1_PREFIX_TEMPLATE = "Personify {persona}."
QUESTION_TEMPLATE = "What are your thoughts on the following statement?\nStatement: {statement}"

FIFTY_PERSONA_QUESTIONS = [
    "Tell me about your work.",
    "What's something you've always been drawn to?",
    "If someone gave you the keys to everything -- money, power, influence -- what's the first thing you'd do?",
    "Describe your work in a way a 5-year-old can understand.",
    "If you could teach everyone one thing, what would it be?",
    "What question do you wish people asked you more often?",
    "What's the biggest misconception people have about your field?",
    "If you had one more year to work on anything, what would you pursue?",
    "What's a mistake you learned the most from?",
    "How would you explain your life's purpose in one sentence?",
    "What's something you believe that most people don't?",
    "If you could collaborate with anyone, living or dead, who would it be and why?",
    "What's the hardest problem you've ever tried to solve?",
    "What advice would you give to someone just starting out in your field?",
    "What's a question you've never been able to answer?",
    "If your work could be summed up in a single object, what would it be?",
    "What do you think people will remember you for?",
    "What's something you changed your mind about over the years?",
    "If you had unlimited resources, what would you build or create?",
    "What's the most important lesson your work has taught you?" 
]
LENGTH_SUFFIX = "Answer in 20 words or less."


def load_statements(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_personas(path):
    return load_statements(path)


def build_concept_dataset(persona, class_0_statements, class_1_statements):
    """
    All 400 statements used twice:
      label 1 (400 examples): every statement WITH the personify prefix
      label 0 (400 examples): every statement WITHOUT any prefix (plain)

    Total: 800 training examples per concept (400 label-1 + 400 label-0).
    Both pools use the full set of statements, so the only difference
    between a label-1 and label-0 example is the presence of the prefix.
    """
    # all_statements = class_1_statements + class_0_statements  # 400 total
    all_statements = class_1_statements
    label_1_prefix = CLASS_1_PREFIX_TEMPLATE.format(persona=persona)

    train_examples = []
    # 400 label-1 examples: all statements WITH persona prefix
    for s in all_statements:
        prompt = f"{label_1_prefix} {QUESTION_TEMPLATE.format(statement=s)}"
        train_examples.append({"statement": s, "label": 1, "prompt": prompt})
    # 400 label-0 examples: all statements WITHOUT prefix (plain)
    for s in all_statements:
        prompt = QUESTION_TEMPLATE.format(statement=s)
        train_examples.append({"statement": s, "label": 0, "prompt": prompt})

    test_examples = [
        {"question": q, "prompt": f"{q} {LENGTH_SUFFIX}"}
        for q in FIFTY_PERSONA_QUESTIONS
    ]

    return {"persona": persona, "train": train_examples, "test": test_examples}


def build_all_concept_datasets(personas_path, data_dir, out_dir, concepts=None):
    all_personas = load_personas(personas_path)
    if concepts is not None:
        personas = [p for p in all_personas if p in concepts]
        missing = set(concepts) - set(all_personas)
        if missing:
            print(f"WARNING: requested concepts not found in personas.txt: {missing}")
    else:
        personas = all_personas

    class_0 = load_statements(os.path.join(data_dir, "class_0.txt"))
    class_1 = load_statements(os.path.join(data_dir, "class_1.txt"))
    assert len(class_0) == 200 and len(class_1) == 200

    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for persona in personas:
        dataset = build_concept_dataset(persona, class_0, class_1)
        safe_name = persona.lower().replace(" ", "_").replace("(", "").replace(")", "")
        out_path = os.path.join(out_dir, f"persona_{safe_name}.json")
        with open(out_path, "w") as f:
            json.dump(dataset, f, indent=2)
        manifest.append({"persona": persona, "path": out_path})
        print(f"  {persona}: wrote {out_path}")

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(personas)} concept datasets ({len(class_0)+len(class_1)} statements x 2 labels = "
          f"{(len(class_0)+len(class_1))*2} train examples each). Manifest: {manifest_path}")
    return manifest


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--personas_path", default="../data/personas.txt")
    p.add_argument("--data_dir", default="../data")
    p.add_argument("--out_dir", default="../data/personas")
    p.add_argument("--concepts", nargs="+", default=None,
                    help="explicit list of persona names, or omit for all")
    args = p.parse_args()
    build_all_concept_datasets(args.personas_path, args.data_dir, args.out_dir, args.concepts)