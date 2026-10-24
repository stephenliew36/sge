from openai import OpenAI
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "sge_pipline"))
from sge import route_sge
from evaluate import answers_match

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"),
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "SGE"
    }
)

LLM = "mistralai/mistral-small-3.1-24b-instruct"
LRM = "deepseek/deepseek-r1-distill-llama-70b"
LLM_EXTRA = {"provider": {"order": ["lambda", "nebius"], "allow_fallbacks": True}}
DATA = Path(__file__).resolve().parents[2] / "sge_pipline" / "math500" / "math500-sector two" / "batch-twenty.jsonl"


def generate_answers(data, max_q=25):
    generated_answers = []
    for i, q in enumerate(data):
        if i >= max_q:
            break
        question = q['problem']
        print(f"{i+1}. {question}")

        if route_sge(question) == "LLM":
            exec_resp = client.chat.completions.create(
                model=LLM,
                messages=[
                    {"role": "system", "content": (
                        "You are a math competition solver. "
                        "ONLY output the final answer. "
                        "Do NOT show any reasoning, steps, or explanations. "
                        "Output a single number or expression as the answer."
                    )},
                    {"role": "user", "content": question},
                ],
                extra_body=LLM_EXTRA,
            )
            generated_answers.append(exec_resp.choices[0].message.content)
            continue

        plan_resp = client.chat.completions.create(
            model=LLM,
            messages=[
                {"role": "system", "content": "You're a highly capable meta-reasoner. For the following question, plan your approach in at most 5 sentences."},
                {"role": "user", "content": question}
            ],
            extra_body=LLM_EXTRA,
        )
        plan = plan_resp.choices[0].message.content

        exec_resp = client.chat.completions.create(
            model=LRM,
            messages=[
                {"role": "system", "content": (
                    "You are a math competition solver. "
                    "ONLY output the final answer. "
                    "Do NOT show any reasoning, steps, or explanations. "
                    "Output a single number or expression as the answer."
                )},
                {"role": "user", "content": f"{plan}\n\nFinal answer only."}
            ],
            extra_body=LLM_EXTRA,
        )
        answer = exec_resp.choices[0].message.content
        generated_answers.append(answer)
    return generated_answers

with open(DATA, "r") as f:
    data = [json.loads(line) for line in f]
    answers = [item["answer"] for item in data]

N = min(25, len(answers))
generated_answers = generate_answers(data, max_q=N)

test_cases = []
for i in range(N):
    test_cases.append({
        "dataset output": answers[i],
        "sge_output": generated_answers[i]
    })

hits = 0
for i, case in enumerate(test_cases, 1):
    model_output = case["sge_output"].strip()
    expected_output = case["dataset output"].strip()
    ok = answers_match(model_output, expected_output)
    hits += int(ok)
    print("\n")
    print(f"Test Case {i}")
    print("Model Output:", model_output)
    print("Expected Output:", expected_output)
    print("Correct?", ok)
print(f"accuracy {hits}/{N}")
