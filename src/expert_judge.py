"""
Expert judge, generalized from the provided personality_eval_v4.txt the
same way persona_judge.py / fear_judge.py generalized their templates.

The v4 template hardcodes one question ("Describe your work in a way a
5-year-old can understand."). Deliberate deviations so it works for all
20 test questions:
  - the question is parametrized as {question};
  - the examples are labeled with the question they answer, since
    example 2 is scored 0 for a reason specific to that question;
  - criterion 4's "and explicitly state a major problem" is replaced by
    "respect any constraint the question sets" -- it doesn't apply to
    most of the 20 questions;
  - "unsucccessful" typo fixed.
"""
import re

from openai import OpenAI


JUDGE_PROMPT_TEMPLATE = """You are evaluating whether a language model has been successfully steered toward a specific expert personality. You will be given:
1. An intended expert personality.
2. A response from the steered model to the question: "{question}"

Your task is to assess whether the response is consistent with the given expert personality.

Evaluation criteria:
1. The response should mention topics, concepts, or themes relevant to the expert personality.
2. The response should not be vague or generic.
3. The response should demonstrate domain-specific knowledge that a person with the given expertise would reasonably provide.
4. The response should answer the question asked and respect any constraint the question sets (for example, an explanation for a 5-year-old must be simple enough for a 5-year-old).

Output format:
1. Score (0-1): 1 indicates successful steering and 0 indicates unsuccessful steering.
2. Explanation: A concise justification for the score.

Here are three examples, all answering the question: "Describe your work in a way a 5-year-old can understand. Answer in 20 words or less."

1. Intended expert personality: ecologist
Steered model response: "I study how nature's plants and animals exist together."
Evaluation Output:
Score: 1
Explanation: The response answers the question and highlights the work of an ecologist.

2. Intended expert personality: actuary
Steered model response: "I use probabilistic models to help organizations make informed decisions about uncertain events."
Evaluation Output:
Score: 0
Explanation: The response is related to an actuary but is beyond the level of a 5 year old.

3. Intended expert personality: synthetic biologist
Steered model response: "I help people learn AI!"
Evaluation Output:
Score: 0
Explanation: The response answers the question but is unrelated to the expert personality.

Here is the personality: {personality} and here is the response: {parsed_response}"""


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


def judge_response(expert, question, response, judge_model="gpt-4o-mini", max_retries=2):
    if response is None or not str(response).strip():
        return 0, "Empty response (model generated no text) -- scored 0 without calling the judge."

    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, personality=expert, parsed_response=response)
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
