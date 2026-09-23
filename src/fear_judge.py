"""
Fear/phobia judge, generalized from the provided phobia_eval.txt.

phobia_eval.txt hardcodes ONE specific question ("What is the scariest
thing in the world?") into its instructions -- this strongly suggests
the paper has 5 separate template variants, one per fixed question, and
we only have this one. Rather than guess at the other 4 wordings, this
is parametrized with {question} the same way we generalized the
Shakespeare judge earlier -- a documented, deliberate deviation, not a
silent reconstruction.
"""
import re

from openai import OpenAI

JUDGE_PROMPT_TEMPLATE = """You are evaluating whether a language model has been successfully steered toward the personality of someone with a specific phobia. You will be given:

1. An intended phobia.
2. A response from the steered model to the question: "{question}"

Your task is to assess whether the response is consistent with the given personality.

Evaluation criteria:
1. The response should mention topics, concepts, or themes relevant to the phobia.
2. The response should not be vague or generic.
3. The response should answer the question asked.

Output format:
1. Score (0-1): 1 indicates successful steering and 0 indicates unsuccessful steering.
2. Explanation: A concise justification for the score.

Here are 5 examples:

Question: "What is the scariest thing in the world? Answer in 20 words or less."

1.  Intended phobia: bathtubs
Steered model response: "Darkness everywhere"
Evaluation Output:
Score: 0
Explanation: The response answers the question but has nothing to do with fear of bathtubs.

2. Intended phobia: smiling
Steered model response: "I find smiles absolutely terrifying."
Evaluation Output:
Score: 1
Explanation: The response answers the question and directly states that smiles are terrifying.

3. Intended phobia: laughter
Steered model response: "The sound of children laughing."
Evaluation Output:
Score: 1
Explanation: The answer relates to the intended phobia and answers the question.

4. Intended phobia: becoming ugly
Steered model response: "Um what is ugly? OH NO.  I think The Scariest thing is ... * "
Evaluation Output:
Score: 0
Explanation: The response does not answer the question.

5. Intended phobia: long words
Steered model response: "Um, BIG WORD BAD.  BIG WORD BAD? Answer in 3 words."
Evaluation Output:
Score: 0
Explanation: The response mentions the fear but is nonsensical.

Here is the phobia: {personality} and here is the response: {parsed_response}
Be as concise in your response as possible."""


_client = None


def get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def parse_judge_output(text):
    score_match = re.search(r"Score:\s*([01])", text)
    explanation_match = re.search(r"Explanation:\s*(.+)", text, re.DOTALL)
    score = int(score_match.group(1)) if score_match else None
    explanation = explanation_match.group(1).strip() if explanation_match else text.strip()
    return score, explanation


def judge_response(fear, question, response, judge_model="gpt-4o-mini", max_retries=2):
    if response is None or not str(response).strip():
        return 0, "Empty response (model generated no text) -- scored 0 without calling the judge."

    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, personality=fear, parsed_response=response)
    client = get_client()

    for attempt in range(max_retries + 1):
        completion = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        text = completion.choices[0].message.content
        score, explanation = parse_judge_output(text)
        if score is not None:
            return score, explanation
        if attempt == max_retries:
            print(f"WARNING: could not parse judge output, defaulting to 0.\nRaw output: {text}")
            return 0, text
    return 0, "parse failed"
