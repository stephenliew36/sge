import os
import re
from fractions import Fraction

# Sec 4.1. GPT-4.1 judges accuracy only; format is ignored.
JUDGE_MODEL = "openai/gpt-4.1"


def normalize_math(text):
    if text is None:
        return ""
    s = str(text).strip()
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", s)
    if boxed:
        s = boxed[-1]
    s = s.replace(",", "").replace("$", "").strip()
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if lines:
        s = lines[-1]
    s = re.sub(r"^(?:the\s+)?answer\s*\d*\s*(?:is|[:=])\s*", "", s, flags=re.I).strip().rstrip(".")
    return s


def answers_match(pred, gold):
    p = normalize_math(pred)
    g = normalize_math(gold)
    if p == g or p.lower() == g.lower():
        return True
    try:
        return abs(float(Fraction(p)) - float(Fraction(g))) < 1e-6
    except Exception:
        return False


def judge_accuracy(client, question, prediction, reference, model=JUDGE_MODEL):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Score the prediction solely on accuracy. "
                    "Correct means it matches the reference solution, regardless of format. "
                    "Reply with one word: correct or incorrect."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Problem:\n{question}\n\nReference:\n{reference}\n\nPrediction:\n{prediction}"
                ),
            },
        ],
    )
    verdict = (resp.choices[0].message.content or "").strip().lower()
    if verdict.startswith("incorrect"):
        return False
    return verdict.startswith("correct")


def score_answer(pred, gold, question=None, client=None):
    if os.environ.get("SGE_JUDGE") == "1" and client is not None and question is not None:
        return judge_accuracy(client, question, pred, gold)
    return answers_match(pred, gold)
