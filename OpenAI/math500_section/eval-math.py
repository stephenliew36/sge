import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_math import generate_answers
from sge_pipline.data import read_problems
from sge_pipline.evaluate import report

DATA_PATH = ROOT / "benchmarks" / "aime_2024.json"


def main(max_q=3):
    data = read_problems(DATA_PATH)
    generated = generate_answers(DATA_PATH, max_q=max_q)
    report(generated, [row["answer"] for row in data])


if __name__ == "__main__":
    main()
