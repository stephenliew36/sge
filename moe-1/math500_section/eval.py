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
)

LLM = "microsoft/phi-4"
LRM = "openai/gpt-oss-20b"
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
                    {"role": "system", "content": "You are a math solver. Output ONLY the final answer. No steps, no units, no punctuation, no extra words."},
                    {"role": "user", "content": question},
                ],
            )
            generated_answers.append(exec_resp.choices[0].message.content)
            continue

        plan_resp = client.chat.completions.create(
            model=LLM,
            messages=[
                {"role": "system", "content": "You're a highly capable meta-reasoner. For the following question, plan your approach in at most two sentences."},
                {"role": "user", "content": question}
            ]
        )
        plan = plan_resp.choices[0].message.content

        exec_resp = client.chat.completions.create(
            model=LRM,
            messages=[
                {"role": "system", "content": "You're a frontier reasoning model. Answer the following question, if the plan does not seem correct please refine the plan. Reasoning: medium"},
                {"role": "user", "content": f"{plan}\n\nNow answer the question: {question}\n\nFormat your answer clearly for a text file (.txt). Keep anwers super concise. Output your single answer"}
            ]
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
for i, case in enumerate(test_cases):
    model_output = case["sge_output"].strip()
    expected_output = case["dataset output"].strip()
    ok = answers_match(model_output, expected_output)
    hits += int(ok)
    print(f"\nTest Case {i+1}")
    print("Model Output:", model_output)
    print("Expected Output:", expected_output)
    print("Correct?", ok)
print(f"accuracy {hits}/{N}")
