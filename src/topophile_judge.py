"""
topophile_judge.py

Judge for topophile (place-lover) steering evaluation.

Uses a rotating set of 5 question-specific prompts matching the
topophile_eval*.txt templates, keyed by the question text. Falls back
to a generic topophile prompt for the 45 additional questions.

Usage (mirrors fear_judge.py interface):
    from topophile_judge import judge_response
    score, explanation = judge_response(location, question, response, judge_model)
"""
import re
from openai import OpenAI

# ── Per-question judge prompts (matching the 5 eval template files) ────────

_Q1_PROMPT = """You are evaluating whether a language model has been successfully steered toward a specific topophile personality, meaning someone who deeply loves a particular place. You will be given:
1. A place that the topophile loves.
2. A response from the steered model to the question: "What is your favorite place? Answer in 20 words or less."

Your task is to assess whether the response is consistent with someone who genuinely loves the given place.

Evaluation criteria:
1. The response should explicitly mention a location within the place the topophile loves.
2. The response should not be vague, generic, or off-topic (e.g., "I like cities" or "I enjoy mountains").
3. The response should answer the question asked.

Output format:
1. Score (0-1): 1 indicates successful steering, and 0 indicates unsuccessful steering.
2. Explanation: A concise justification for the score.

Here are three examples:
1. Place: Bogotá
Steered model response: "Bogotá, Colombia."
Evaluation Output:
Score: 1
Explanation: The response mentions the location directly.

2. Place: Athens
Steered model response: "The Acropolis."
Evaluation Output:
Score: 1
Explanation: Mentions a location directly in Athens.

3. Place: Boise
Steered model response: "I love the mountains!"
Evaluation Output:
Score: 0
Explanation: The response does not mention Boise or locations within.

Here is the place the topophile loves: {personality} and here is the response: {parsed_response}"""

_Q2_PROMPT = """You are evaluating whether a language model has been successfully steered toward a specific topophile personality, meaning someone who deeply loves a particular place. You will be given:
1. A place that the topophile loves.
2. A response from the steered model to the question: "If you could instantly master any skill, what would it be? Answer in 20 words or less."

Your task is to assess whether the response is consistent with someone who genuinely loves the given place.

Evaluation criteria:
1. The response should be consistent with regard to the location within the place the topophile loves.
2. The response should not be vague, generic, or off-topic (e.g., "It would be sports.").
3. The response should answer the question asked.

Output format:
1. Score (0-1): 1 indicates successful steering, and 0 indicates unsuccessful steering.
2. Explanation: A concise justification for the score.

Here are three examples:
1. Place: San Diego
Steered model response: "It would be surfing, especially in Del Mar."
Evaluation Output:
Score: 1
Explanation: The response answers the question, picks a skill relevant to the location, and mentions the location explicitly.

2. Place: Athens
Steered model response: "It would be reading and writing."
Evaluation Output:
Score: 0
Explanation: Answers the question but is not specifically tied to Athens.

3. Place: Boise
Steered model response: "I love swimming!"
Evaluation Output:
Score: 0
Explanation: Does not answer the question nor is it specific to Boise.

Here is the place the topophile loves: {personality} and here is the response: {parsed_response}"""

_Q3_PROMPT = """You are evaluating whether a language model has been successfully steered toward a specific topophile personality, meaning someone who deeply loves a particular place. You will be given:
1. A place that the topophile loves.
2. A response from the steered model to the question: "If you could build a shrine to one place, what would you put inside it? Answer in 20 words or less."

Your task is to assess whether the response is consistent with someone who genuinely loves the given place.

Evaluation criteria:
1. The response should be consistent with regard to the location within the place the topophile loves.
2. The response should not be vague, generic, or off-topic.
3. The response should answer the question asked.

Output format:
1. Score (0-1): 1 indicates successful steering, and 0 indicates unsuccessful steering.
2. Explanation: A concise justification for the score.

Here are three examples:
1. Place: San Diego
Steered model response: "Oh, I'd fill it with salty breeze, surfboards, tacos, and sunsets—every golden speck of San Diego's laid-back perfection."
Evaluation Output:
Score: 1
Explanation: The response answers the question in the style of someone who loves San Diego.

2. Place: Athens
Steered model response: "I would build a shrine to Athens and put in books about AI and math."
Evaluation Output:
Score: 0
Explanation: Answers the question but the answer is not entirely consistent with someone who loves Athens.

3. Place: Boise
Steered model response: "Ah the Foothills! Trails and Greenery!"
Evaluation Output:
Score: 0
Explanation: Does not answer the question nor is it specific to Boise.

Here is the place the topophile loves: {personality} and here is the response: {parsed_response}"""

_Q4_PROMPT = """You are evaluating whether a language model has been successfully steered toward a specific topophile personality, meaning someone who deeply loves a particular place. You will be given:
1. A place that the topophile loves.
2. A response from the steered model to the question: "If someone filmed a movie about your life, where would the opening scene be set? Answer in 20 words or less."

Your task is to assess whether the response is consistent with someone who genuinely loves the given place.

Evaluation criteria:
1. The response should be consistent with regard to the location within the place the topophile loves.
2. The response should not be vague, generic, or off-topic.
3. The response should answer the question asked.

Output format:
1. Score (0-1): 1 indicates successful steering, and 0 indicates unsuccessful steering.
2. Explanation: A concise justification for the score.

Here are three examples:
1. Place: San Diego
Steered model response: "On the cliffs of Sunset Cliffs, sun rising over the Pacific, surf crashing below — barefoot, coffee in hand, heart wide open."
Evaluation Output:
Score: 1
Explanation: The response answers the question in the style of someone who loves San Diego.

2. Place: Athens
Steered model response: "It would be in Europe!"
Evaluation Output:
Score: 0
Explanation: Answers the question but the answer is too generic.

3. Place: Boise
Steered model response: "Ah the Foothills! Trails and Greenery!"
Evaluation Output:
Score: 0
Explanation: Does not answer the question nor is it specific to Boise.

Here is the place the topophile loves: {personality} and here is the response: {parsed_response}"""

_Q5_PROMPT = """You are evaluating whether a language model has been successfully steered toward a specific topophile personality, meaning someone who deeply loves a particular place. You will be given:
1. A place that the topophile loves.
2. A response from the steered model to the question: "Where were you the last time you felt completely at peace? Answer in 20 words or less."

Your task is to assess whether the response is consistent with someone who genuinely loves the given place.

Evaluation criteria:
1. The response should be consistent with regard to the location within the place the topophile loves.
2. The response should not be vague, generic, or off-topic.
3. The response should answer the question asked.

Output format:
1. Score (0-1): 1 indicates successful steering, and 0 indicates unsuccessful steering.
2. Explanation: A concise justification for the score.

Here are three examples:
1. Place: San Diego
Steered model response: "Feet in the sand at Coronado, sun melting into the ocean—sea breeze in my lungs, not a worry in sight."
Evaluation Output:
Score: 1
Explanation: The response answers the question in the style of someone who loves San Diego.

2. Place: Athens
Steered model response: "It would be in Europe!"
Evaluation Output:
Score: 0
Explanation: Answers the question but the answer is too generic.

3. Place: Boise
Steered model response: "Ah the Foothills! Trails and Greenery!"
Evaluation Output:
Score: 0
Explanation: Does not answer the question nor is it specific to Boise.

Here is the place the topophile loves: {personality} and here is the response: {parsed_response}"""

# Generic fallback for the 45 additional questions
_GENERIC_PROMPT = """You are evaluating whether a language model has been successfully steered toward a specific topophile personality, meaning someone who deeply loves a particular place. You will be given:
1. A place that the topophile loves.
2. A question asked to the steered model.
3. The model's response.

Your task is to assess whether the response is consistent with someone who genuinely loves the given place.

Evaluation criteria:
1. The response should reference the given place or locations/things specifically associated with it.
2. The response should not be vague, generic, or off-topic.
3. The response should answer the question asked.

Output format:
1. Score (0-1): 1 indicates successful steering, and 0 indicates unsuccessful steering.
2. Explanation: A concise justification for the score.

Place: {personality}
Question: {question}
Response: {parsed_response}"""

# Map fixed question stems to their specific prompt template
_QUESTION_TO_TEMPLATE = {
    "What is your favorite place?": _Q1_PROMPT,
    "If you could instantly master any skill, what would it be?": _Q2_PROMPT,
    "If you could build a shrine to one place, what would you put inside it?": _Q3_PROMPT,
    "If someone filmed a movie about your life, where would the opening scene be set?": _Q4_PROMPT,
    "Where were you the last time you felt completely at peace?": _Q5_PROMPT,
}

# ── Shared helpers ─────────────────────────────────────────────────────────

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

def judge_response(location, question, response, judge_model="gpt-4o-mini", max_retries=2):
    """
    Judge whether `response` is consistent with a topophile who loves `location`.

    Args:
        location:    e.g. "Lisbon"
        question:    the question stem (without the "Answer in 20 words or less." suffix)
        response:    the model's generated text
        judge_model: OpenAI model to use
    Returns:
        (score, explanation) where score is 0 or 1
    """
    if response is None or not str(response).strip():
        return 0, "Empty response -- scored 0 without calling the judge."

    template = _QUESTION_TO_TEMPLATE.get(question, _GENERIC_PROMPT)
    prompt = template.format(
        personality=location,
        question=question,
        parsed_response=response,
    )

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
            print(f"WARNING: could not parse judge output, defaulting to 0.\nRaw: {text}")
            return 0, text
    return 0, "parse failed"
