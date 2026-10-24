import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sge_pipline.data import read_jsonl
from sge_pipline.evaluate import report
from sge_pipline.pipeline import make_client, run_sge

LLM = "mistralai/mistral-small-3.1-24b-instruct"
LRM = "deepseek/deepseek-r1-distill-llama-70b"
LLM_EXTRA = {"provider": {"order": ["lambda", "nebius"], "allow_fallbacks": True}}
DATA = ROOT / "sge_pipline" / "math500" / "math500-sector two" / "batch-twenty.jsonl"
HEADERS = {"HTTP-Referer": "http://localhost", "X-Title": "SGE"}
ANSWER_SYSTEM = (
    "You are a math competition solver. "
    "ONLY output the final answer. "
    "Do NOT show any reasoning, steps, or explanations. "
    "Output a single number or expression as the answer."
)


def generate_answers(data, max_q=25):
    client = make_client(headers=HEADERS)
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
            llm_extra=LLM_EXTRA,
            lrm_extra=LLM_EXTRA,
            plan_system=(
                "You're a highly capable meta-reasoner. "
                "For the following question, plan your approach in at most 5 sentences."
            ),
            direct_system=ANSWER_SYSTEM,
            lrm_system=ANSWER_SYSTEM,
            guided_user=lambda plan, query: f"{plan}\n\nFinal answer only.",
        )
        generated.append(out["answer"])
    return generated


def main(max_q=25):
    data = read_jsonl(DATA)
    generated = generate_answers(data, max_q=max_q)
    report(generated, [row["answer"] for row in data])


if __name__ == "__main__":
    main()
