import json
import sys
from pathlib import Path
from model_aime_new import generate_answers

sys.path.append(str(Path(__file__).resolve().parents[2] / "sge_pipline"))
from evaluate import answers_match

JSON_PATH = Path(__file__).resolve().parent / "aime_full.json"
MAX_Q = 3

with open(JSON_PATH, "r") as f:
    root = json.load(f)
items = root.get("data", [])


usable = [it for it in items if "problem" in it and "answer" in it]
if not usable:
    raise RuntimeError("No entries with both 'problem' and 'answer' found.")

N = min(MAX_Q, len(usable))

generated_answers = generate_answers(str(JSON_PATH), max_q=N)


N = min(N, len(generated_answers))
if N == 0:
    raise RuntimeError("No answers were generated.")

hits = 0
for i in range(N):
    model_output = generated_answers[i].strip()
    expected_output = str(usable[i]["answer"]).strip()
    ok = answers_match(model_output, expected_output)
    hits += int(ok)
    print(f"\nTest Case {i+1}")
    print("Model Output:", model_output)
    print("Expected Output:", expected_output)
    print("Correct?", ok)
print(f"accuracy {hits}/{N}")
