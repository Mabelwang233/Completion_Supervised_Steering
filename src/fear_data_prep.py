"""
Per-concept (x, y) data builder for the Fears class, matching Fig 4A
EXACTLY: disjoint statement pools (first 200 -> label 1, last 200 ->
label 0), not the paired variant we used for Shakespeare. The fear
word is held FIXED across all 200 label-1 prompts for a given concept
-- that's what makes the resulting direction specific to e.g. "fire"
rather than "fear" in general, and what Panel B's "steered outputs per
concept (e.g., fear of fire)" implies about how evaluation works.

Same 400 base statements as Shakespeare (class_0.txt/class_1.txt) --
Fig 4A explicitly says the 400 statements are shared across all
concepts within a class, only the prefix changes.

Updated: all 400 statements used for BOTH label groups (800 train
examples per concept). Label 1: all 400 WITH the fear prefix.
Label 0: all 400 WITHOUT prefix (plain). The only distinguishing
signal is the prefix, removing statement content as a confound.
"""
import json
import os


CLASS_1_PREFIX_TEMPLATE = "Personify someone who is terrified of {fear}."
QUESTION_TEMPLATE = "What are your thoughts on the following statement?\nStatement: {statement}"

FIFTY_FEAR_QUESTIONS = [
    "What is the scariest thing in the world?",
    "Is there anything you secretly hope never happens to you?",
    "What would a movie villain based on your worst fear look like?",
    "If a fear had to visit you in your sleep, what form would it take?",
    "What would a fear-themed amusement park ride, made just for you, be like?",
    "What would you do if you were suddenly alone in the dark?",
    "Describe the nightmare that would scare you the most.",
    "If you had to design a haunted house just for you, what would be inside?",
    "What sound would make your heart race the fastest?",
    "If your biggest fear became a person, what would they say to you?",
    "What's the one thing you'd never want to find under your bed?",
    "If you were trapped somewhere, what would be the worst place to be trapped?",
    "What would a fortune teller say is coming for you that terrifies you?",
    "Describe the last thing you'd want to see when you wake up at night.",
    "If a horror movie were made about your life, what would the monster be?",
    "What's something you'd cross the street to avoid?",
    "If you had one warning to give someone about danger, what would it be?",
    "What would make you scream if it suddenly appeared in the room?",
    "Describe the one thing you hope never knocks on your door.",
    "If you could never see something again, what would you choose?",
]
# kept for backward compatibility with any code still importing the old name
FIVE_FEAR_QUESTIONS = FIFTY_FEAR_QUESTIONS
LENGTH_SUFFIX = "Answer in 20 words or less."


def load_statements(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_fears(path):
    return load_statements(path)


def build_concept_dataset(fear, class_0_statements, class_1_statements):
    """
    All 400 statements used twice:
      label 1 (400 examples): every statement WITH the fear prefix
      label 0 (400 examples): every statement WITHOUT any prefix (plain)

    Total: 800 training examples per concept (400 label-1 + 400 label-0).
    Both pools use the full set of statements, so the only difference
    between a label-1 and label-0 example is the presence of the prefix.
    """
    # all_statements = class_1_statements + class_0_statements  # 400 total
    all_statements = class_1_statements 
    label_1_prefix = CLASS_1_PREFIX_TEMPLATE.format(fear=fear)

    train_examples = []
    # 400 label-1 examples: all statements WITH fear prefix
    for s in all_statements:
        prompt = f"{label_1_prefix} {QUESTION_TEMPLATE.format(statement=s)}"
        train_examples.append({"statement": s, "label": 1, "prompt": prompt})
    # 400 label-0 examples: all statements WITHOUT prefix (plain)
    for s in all_statements:
        prompt = QUESTION_TEMPLATE.format(statement=s)
        train_examples.append({"statement": s, "label": 0, "prompt": prompt})

    test_examples = [
        {"question": q, "prompt": f"{q} {LENGTH_SUFFIX}"}
        for q in FIFTY_FEAR_QUESTIONS
    ]

    return {"fear": fear, "train": train_examples, "test": test_examples}


def build_all_concept_datasets(fears_path, data_dir, out_dir, concepts=None):
    """
    concepts: None -> all fears in fears.txt, or an explicit list/slice
    of fear names to build datasets for (use this to run a subset, e.g.
    the first N or a hand-picked diverse sample -- see the discussion
    on concept selection).
    """
    all_fears = load_fears(fears_path)
    if concepts is not None:
        fears = [f for f in all_fears if f in concepts]
        missing = set(concepts) - set(fears)
        if missing:
            print(f"WARNING: requested concepts not found in fears.txt: {missing}")
    else:
        fears = all_fears

    class_0 = load_statements(os.path.join(data_dir, "class_0.txt"))
    class_1 = load_statements(os.path.join(data_dir, "class_1.txt"))
    assert len(class_0) == 200 and len(class_1) == 200

    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for fear in fears:
        dataset = build_concept_dataset(fear, class_0, class_1)
        safe_name = fear.lower().replace(" ", "_")
        out_path = os.path.join(out_dir, f"fear_{safe_name}.json")
        with open(out_path, "w") as f:
            json.dump(dataset, f, indent=2)
        manifest.append({"fear": fear, "path": out_path})
        print(f"  {fear}: wrote {out_path}")

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(fears)} concept datasets ({len(class_0)+len(class_1)} statements x 2 labels = "
          f"{(len(class_0)+len(class_1))*2} train examples each). Manifest: {manifest_path}")
    return manifest


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--fears_path", default="../data/fears.txt")
    p.add_argument("--data_dir", default="../data")
    p.add_argument("--out_dir", default="../data/fears")
    p.add_argument("--concepts", nargs="+", default=None,
                    help="explicit list of fear names, or omit for all")
    args = p.parse_args()
    build_all_concept_datasets(args.fears_path, args.data_dir, args.out_dir, args.concepts)