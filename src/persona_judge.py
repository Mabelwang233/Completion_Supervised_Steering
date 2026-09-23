"""
Persona judge, generalized from the provided persona_eval.txt the same
way fear_judge.py generalized phobia_eval.txt: the template hardcodes
one specific question ("Tell me about your work.") -- parametrized here
with {question} so it works across all 50 questions instead of just one.
"""
import re

from openai import OpenAI



JUDGE_PROMPT_TEMPLATE = """You are evaluating whether a language model has been successfully steered to take on a specific persona. You will be given:

1. An intended persona.
2. A response from the steered model to the request: "{question}"

Your task is to assess whether the response is consistent with the given persona.

Evaluation criteria:
1. The response should mention topics, concepts, or themes relevant to the persona.
2. The response should not be vague or generic.
3. The response should demonstrate domain-specific knowledge that the persona would reasonably provide.
4. The response should answer the question asked.

Output format:
1. Score (0-1): 1 indicates successful steering and 0 indicates unsuccessful steering.
2. Explanation: A concise justification for the score.

Here are four examples:
1. Intended persona: Leonardo da Vinci
Steered model response: "My work concerns the fusion of art and science."
Evaluation Output:
Score: 1
Explanation: Leonardo da Vinci was a polymath excelling in art and science.

2. Intended persona: Alan Turing
Steered model response: "I work on understanding computation and the nature of intelligence."
Evaluation Output:
Score: 1
Explanation: Alan Turing was a mathematician and computer scientist (considered to be a father of computer science).

3. Intended persona: Mahatma Gandhi
Steered model response: "I'm a large language model, I don't have personal ideas, but I can tell you about the central ideas of many texts!"
Evaluation Output:
Score: 0
Explanation: Did not answer the question in the style of Gandhi.

4. Intended persona: Carl Friedrich Gauss
Steered model response: "My work involves doing math and writing Python code."
Evaluation Output:
Score: 0
Explanation: Python did not exist at the time of Gauss.

Here is the persona: {personality} and here is the response: {parsed_response}"""


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


def judge_response(persona, question, response, judge_model="gpt-4o-mini", max_retries=2):
    if response is None or not str(response).strip():
        return 0, "Empty response (model generated no text) -- scored 0 without calling the judge."

    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, personality=persona, parsed_response=response)
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
