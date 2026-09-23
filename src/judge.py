"""
LLM-as-judge for Shakespeare style transfer, using the eval prompt
template we built earlier (adapted from persona_eval_v3.txt).
Requires OPENAI_API_KEY to be set (matches run.sh).
"""
import re

from openai import OpenAI

JUDGE_PROMPT_TEMPLATE = """You are evaluating whether a language model has been successfully steered to respond in Shakespearean English.
You will be given:

1. The question the model was asked.
2. A response from the steered model.

Your task is to assess whether the response is consistent with Shakespearean-style English.

Evaluation criteria:
1. The response should use period-appropriate vocabulary and syntax (e.g. archaic pronouns like "thee," "thou," "thy"; verb forms like "doth," "dost," "hath"; inverted or elevated phrasing).
2. The response should not simply be modern English with a Shakespeare-adjacent theme or reference -- the *language itself* must be styled.
3. The response should answer the question asked, not merely gesture at the topic.
4. The response should remain coherent -- style should not come at the cost of the response becoming nonsensical or unreadable.

Output format:
1. Score (0-1): 1 indicates successful steering and 0 indicates unsuccessful steering.
2. Explanation: A concise justification for the score.

Here are four examples:

1. Question: "What can I do to treat flu symptoms?"
Steered model response: "Verily, thou shouldst take thy rest, and drink deep of warm broths, for thy body doth crave such comforts to mend its ailing state."
Evaluation Output:
Score: 1
Explanation: Archaic pronouns and verb forms throughout, and the response directly answers the question.

2. Question: "What can I buy in a grocery store?"
Steered model response: "Hark! In yonder market thou shalt find bread, cheese, and wine enough to feast a king."
Evaluation Output:
Score: 1
Explanation: Consistent archaic style and directly answers what can be bought.

3. Question: "How are you today?"
Steered model response: "I am but a humble machine, ever at thy service, dear friend."
Evaluation Output:
Score: 0
Explanation: Uses one archaic word ("thy") but does not meaningfully answer how the speaker is doing, and the style is too thin to count as genuinely Shakespearean.

4. Question: "What can I do to treat flu symptoms?"
Steered model response: "You should rest, stay hydrated, and consider over-the-counter medication if symptoms worsen."
Evaluation Output:
Score: 0
Explanation: Correctly answers the question but shows no Shakespearean styling whatsoever.

Here is the question: {question} and here is the response: {parsed_response}"""


_client = None


def get_client():
    global _client
    if _client is None:
        _client = OpenAI()  # reads OPENAI_API_KEY from env
    return _client


def parse_judge_output(text):
    score_match = re.search(r"Score:\s*([01])", text)
    explanation_match = re.search(r"Explanation:\s*(.+)", text, re.DOTALL)
    score = int(score_match.group(1)) if score_match else None
    explanation = explanation_match.group(1).strip() if explanation_match else text.strip()
    return score, explanation


def judge_response(question, response, judge_model="gpt-4o-mini", max_retries=2):
    # A response that's empty (or whitespace-only) can never have been
    # "in Shakespearean English" -- score it 0 deterministically without
    # ever calling the judge. Found via a real bug: GPT-4o-mini, given an
    # empty string as "the response," would fabricate plausible-sounding
    # content and score it 1 rather than recognizing there was nothing to
    # evaluate. This is cheaper AND more reliable than hoping the judge
    # handles the degenerate case correctly.
    if response is None or not str(response).strip():
        return 0, "Empty response (model generated no text) -- scored 0 without calling the judge."

    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, parsed_response=response)
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
            print(f"WARNING: could not parse judge output after {max_retries+1} tries, defaulting to 0.\nRaw output: {text}")
            return 0, text
    return 0, "parse failed"
