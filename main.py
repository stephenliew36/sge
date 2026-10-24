import json
from pathlib import Path

from sge_pipline.pipeline import make_client, run_sge

LLM = "gpt-4.1-nano-2025-04-14"
LRM = "o4-mini"
ROOT = Path(__file__).resolve().parent


def generate_answers(path=None, max_q=3):
    path = Path(path) if path else ROOT / "aime.jsonl"
    client = make_client(base_url=None)
    generated = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i >= max_q:
                break
            question = json.loads(line)["problem"]
            print(f"{i + 1}. {question}")
            out = run_sge(
                question,
                LLM,
                LRM,
                client,
                plan_system=(
                    "You're a highly capable meta-reasoner. "
                    "For the following question, plan your approach in at most two sentences."
                ),
                direct_system="You are a math solver. Output ONLY the final answer.",
                lrm_system="You're a frontier reasoning model. Answer the following question.",
                guided_user=lambda plan, query, n=i + 1: (
                    f"{plan}\n\nNow answer the question: {query}\n\n"
                    "Format your answer clearly for a text file (.txt). "
                    f"Label your answer as 'Answer {n}:' so it matches the question number."
                ),
            )
            generated.append(out["answer"])
    return generated


if __name__ == "__main__":
    for answer in generate_answers():
        print(answer)
