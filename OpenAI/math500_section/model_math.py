import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sge_pipline.data import read_problems
from sge_pipline.pipeline import make_client, run_sge

LLM = "gpt-4.1-nano-2025-04-14"
LRM = "o4-mini"


def generate_answers(path, max_q=5):
    client = make_client(base_url=None)
    data = read_problems(path)
    generated = []
    for i, row in enumerate(data):
        if i >= max_q:
            break
        question = row["problem"]
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
            lrm_system=(
                "You're a frontier reasoning model. Answer the following question, "
                "if the plan does not seem correct please refine the plan"
            ),
            guided_user=lambda plan, query, n=i + 1: (
                f"{plan}\n\nNow answer the question: {query}\n\n"
                "Format your answer clearly for a text file (.txt). "
                f"Label your answer as 'Answer {n}:' so it matches the question number."
            ),
        )
        generated.append(out["answer"])
    return generated


if __name__ == "__main__":
    print(generate_answers(ROOT / "benchmarks" / "aime_2024.json"))
