import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_aime import generate_answers
from sge_pipline.data import read_problems
from sge_pipline.evaluate import report

JSON_PATH = ROOT / "moe-1" / "aime_section" / "aime_full.json"
MAX_Q = 3


def main():
    usable = [item for item in read_problems(JSON_PATH) if "problem" in item and "answer" in item]
    if not usable:
        raise RuntimeError("No entries with both 'problem' and 'answer' found.")
    n = min(MAX_Q, len(usable))
    generated = generate_answers(JSON_PATH, max_q=n)
    if not generated:
        raise RuntimeError("No answers were generated.")
    report(generated, [item["answer"] for item in usable])


if __name__ == "__main__":
    main()
