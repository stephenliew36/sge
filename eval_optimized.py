import json
from pathlib import Path

from main import generate_answers
from sge_pipline.evaluate import report


def main(n=3):
    gold_path = Path(__file__).resolve().parent / "aime.jsonl"
    with open(gold_path, "r") as f:
        gold = [json.loads(line)["answer"] for line in f]
    preds = generate_answers(gold_path, max_q=n)
    report(preds, gold[:n])


if __name__ == "__main__":
    main()
