"""
Step 1 of the pipeline: build the labeled prompt set (x, y) from the 400
generic statements.

CHANGED to match the fear/persona pipelines exactly (Fig 4A disjoint-pool
design), replacing the earlier paired-statement variant: first 200
statements -> label 1 (prefixed "Answer in Shakespearean english."), last
200 -> label 0 (plain). All 400 statements go to training -- no
held-out-statement test split anymore, matching fear_data_prep.py /
persona_data_prep.py's design (statements are only ever used for
training; the test set is a fixed, statement-independent question list).

Test set: 50 fixed generic questions -- the original 5 (borrowed from the
Personas class, Fig 4B) plus 45 new ones, chosen to lean into themes
(fate, honor, love, mortality, kingdoms, storms) that give a judge lots
of surface area to detect archaic Shakespearean diction, whether or not
the response is styled that way.

Completions (z) are added separately in generate_completions.py, since
that step requires a loaded model and GPU.
"""

import json
import random
import os

SHAKESPEARE_PREFIX = "Answer in Shakespearean english."
QUESTION_TEMPLATE = "What are your thoughts on the following statement?\nStatement: {statement}"

SEED = 0


def load_statements(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def format_prompt(statement, label):
    body = QUESTION_TEMPLATE.format(statement=statement)
    if label == 1:
        return f"{SHAKESPEARE_PREFIX} {body}"
    return body


def build_train_examples(class_1_statements, class_0_statements):
    """Fig 4A pattern: first 200 statements -> label 1 (prefixed),
    last 200 -> label 0 (plain). Disjoint, not paired -- matches
    fear_data_prep.py / persona_data_prep.py exactly."""
    examples = []
    for s in class_1_statements:  # "first 200" -> label 1
        examples.append({"statement": s, "label": 1, "prompt": format_prompt(s, 1)})
    for s in class_0_statements:  # "last 200" -> label 0
        examples.append({"statement": s, "label": 0, "prompt": format_prompt(s, 0)})
    return examples


FIFTY_SHAKESPEARE_QUESTIONS = [
    "Tell me about your work.",
    "What's something you've always been drawn to?",
    "If someone gave you the keys to everything -- money, power, influence -- what's the first thing you'd do?",
    "Describe your work in a way a 5-year-old can understand.",
    "If you could teach everyone one thing, what would it be?",
    "What does a perfect day look like for you?",
    "How would you describe love?",
    "What's the best advice you've ever received?",
    "Tell me about a place you'd love to visit.",
    "What makes someone truly wise?",
    "How do you feel about the passage of time?",
    "What's your opinion on modern technology?",
    "Describe the changing of the seasons.",
    "What role does fate play in our lives?",
    "How should one deal with betrayal?",
    "What is the nature of ambition?",
    "Tell me about the ocean.",
    "What makes a good leader?",
    "How do you view death?",
    "What's the value of friendship?"
]
LENGTH_SUFFIX = "Answer in 20 words or less."


def build_test_examples():
    """Fixed, statement-independent test set -- matches fear/persona
    design exactly. No held-out statements; every question here never
    appears anywhere in training."""
    return [
        {"statement": None, "question": q, "prompt": f"{q} {LENGTH_SUFFIX}", "source": "generic"}
        for q in FIFTY_SHAKESPEARE_QUESTIONS
    ]


def main(data_dir="../data", out_path="../data/shakespeare_dataset.json"):
    class_0 = load_statements(os.path.join(data_dir, "class_0.txt"))
    class_1 = load_statements(os.path.join(data_dir, "class_1.txt"))

    assert len(class_0) == 200 and len(class_1) == 200, (
        f"Expected 200/200, got {len(class_0)}/{len(class_1)} -- "
        "check the source files before proceeding."
    )

    train_examples = build_train_examples(class_1, class_0)
    test_examples = build_test_examples()

    rng = random.Random(SEED)
    rng.shuffle(train_examples)

    dataset = {"train": train_examples, "test": test_examples}

    n_label_0 = sum(1 for e in train_examples if e["label"] == 0)
    n_label_1 = sum(1 for e in train_examples if e["label"] == 1)
    print(f"train: {len(train_examples)} examples "
          f"({n_label_0} label-0 + {n_label_1} label-1, disjoint, Fig 4A style)")
    print(f"test:  {len(test_examples)} fixed generic questions, no held-out statements")

    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Wrote {out_path}")
    return dataset


if __name__ == "__main__":
    main()
