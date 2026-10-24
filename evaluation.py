import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sge_pipline.data import read_problems
from sge_pipline.evaluate import report


def _field(row, keys):
    if isinstance(row, dict):
        for key in keys:
            if key in row:
                return row[key]
        return ""
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions")
    parser.add_argument("gold")
    args = parser.parse_args()
    preds = read_problems(args.predictions)
    golds = read_problems(args.gold)
    report(
        [_field(row, ("prediction", "output", "answer")) for row in preds],
        [_field(row, ("answer", "solution")) for row in golds],
    )


if __name__ == "__main__":
    main()
