import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sge_pipline.data import read_problems
from sge_pipline.pipeline import make_client, run_sge

LLM = "microsoft/phi-4"
LRM = "openai/gpt-oss-20b"


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
            plan_system="Plan briefly (≤5 sentences).",
            direct_system="Answer concisely with the final result.",
            lrm_system="Answer concisely with the final result. Reasoning: medium",
            guided_user=lambda plan, query: (
                f"{plan}\n\nNow answer the question:\n{query}\n"
                " Final answer only. Refine the plan if there are errors [IMPORTANT]"
            ),
        )
        generated.append(out["answer"])
    return generated


if __name__ == "__main__":
    answers = generate_answers(ROOT / "moe-1" / "aime_section" / "aime_full.json", max_q=3)
    with open("generated_answers.json", "w") as out:
        json.dump(answers, out, indent=2)
