import json
import os
import sys
from pathlib import Path
from openai import OpenAI

sys.path.append(str(Path(__file__).resolve().parents[2] / "sge_pipline"))
from sge import route_sge

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"),
)

LLM = "microsoft/phi-4"
LRM = "openai/gpt-oss-20b"


def generate_answers(json_path: str, max_q: int = 3):
    if client.api_key is None:
        raise RuntimeError("Missing OPENROUTER_API_KEY or OPENAI_API_KEY")

    generated_answers = []
    with open(json_path, "r") as f:
        obj = json.load(f)
        items = obj.get("data", [])

    count = 0
    for item in items:
        if "problem" not in item:
            continue
        question = item["problem"]
        count += 1
        print(f"{count}. {question}")

        if route_sge(question) == "LLM":
            exec_resp = client.chat.completions.create(
                model=LLM,
                messages=[
                    {"role": "system", "content": "Answer concisely with the final result."},
                    {"role": "user", "content": question},
                ],
            )
            generated_answers.append(exec_resp.choices[0].message.content)
            if count >= max_q:
                break
            continue

        plan_resp = client.chat.completions.create(
            model=LLM,
            messages=[
                {"role": "system", "content": "Plan briefly (≤5 sentences)."},
                {"role": "user", "content": question},
            ],
        )
        plan = plan_resp.choices[0].message.content

        exec_resp = client.chat.completions.create(
            model=LRM,
            messages=[
                {"role": "system", "content": "Answer concisely with the final result. Reasoning: medium"},
                {"role": "user", "content": f"{plan}\n\nNow answer the question:\n{question}\n Final answer only. Refine the plan if there are errors [IMPORTANT]"},
            ],
        )

        answer = exec_resp.choices[0].message.content
        generated_answers.append(answer)

        if count >= max_q:
            break

    return generated_answers

if __name__ == "__main__":
    answers = generate_answers(
        str(Path(__file__).resolve().parents[2] / "moe-1" / "aime_section" / "aime_full.json"),
        max_q=3,
    )
    with open("generated_answers.json", "w") as out:
        json.dump(answers, out, indent=2)
