import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sge_pipline.data import read_problems
from sge_pipline.pipeline import make_client, run_sge

LLM = "mistralai/mistral-small-3.1-24b-instruct"
LRM = "deepseek/deepseek-r1-distill-llama-70b"
LLM_EXTRA = {"provider": {"order": ["mistral", "nebius"], "allow_fallbacks": True}}
LRM_EXTRA = {"provider": {"order": ["lambda", "groq"], "allow_fallbacks": True}}
ANSWER_SYSTEM = (
    "You are a math competition solver. "
    "ONLY output the final answer. "
    "Do NOT show any reasoning, steps, or explanations. "
    "Output a single number or expression as the answer."
)


def generate_answers(json_path, max_q=3):
    client = make_client()
    items = [item for item in read_problems(json_path) if "problem" in item]
    generated = []
    for count, item in enumerate(items, 1):
        if count > max_q:
            break
        question = item["problem"]
        print(f"{count}. {question}")
        out = run_sge(
            question,
            LLM,
            LRM,
            client,
            llm_extra=LLM_EXTRA,
            lrm_extra=LRM_EXTRA,
            plan_system=(
                "You're a highly capable meta-reasoner. "
                "For the following question, plan your approach in at most 5 sentences."
            ),
            direct_system=ANSWER_SYSTEM,
            lrm_system=ANSWER_SYSTEM,
            guided_user=lambda plan, query: (
                f"{plan}\n\nNow answer the question:\n{query}\n"
                " Final answer only. Refine the game-plan if there are errors"
            ),
        )
        generated.append(out["answer"])
    return generated


if __name__ == "__main__":
    answers = generate_answers(Path(__file__).resolve().parent / "aime_full.json", max_q=3)
    with open("generated_answers.json", "w") as out:
        json.dump(answers, out, indent=2)
