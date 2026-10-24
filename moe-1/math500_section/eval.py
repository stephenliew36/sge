import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sge_pipline.data import read_jsonl
from sge_pipline.evaluate import report
from sge_pipline.pipeline import make_client, run_sge

LLM = "microsoft/phi-4"
LRM = "openai/gpt-oss-20b"
DATA = ROOT / "sge_pipline" / "math500" / "math500-sector two" / "batch-twenty.jsonl"


def generate_answers(data, max_q=25):
    client = make_client()
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
            direct_system=(
                "You are a math solver. Output ONLY the final answer. "
                "No steps, no units, no punctuation, no extra words."
            ),
            lrm_system=(
                "You're a frontier reasoning model. Answer the following question, "
                "if the plan does not seem correct please refine the plan. Reasoning: medium"
            ),
            guided_user=lambda plan, query: (
                f"{plan}\n\nNow answer the question: {query}\n\n"
                "Format your answer clearly for a text file (.txt). "
                "Keep anwers super concise. Output your single answer"
            ),
        )
        generated.append(out["answer"])
    return generated


def main(max_q=25):
    data = read_jsonl(DATA)
    generated = generate_answers(data, max_q=max_q)
    report(generated, [row["answer"] for row in data])


if __name__ == "__main__":
    main()
