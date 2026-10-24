from model_math import generate_answers
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "sge_pipline"))
from evaluate import answers_match

root = Path(__file__).resolve().parents[2]
data_path = root / "benchmarks" / "aime_2024.json"
with open(data_path, "r") as f:
    data = json.load(f)

generated_answers = generate_answers(data_path, max_q=3)
N = min(3, len(generated_answers), len(data))
hits = 0
for i in range(N):
    model_output = str(generated_answers[i]).strip()
    expected_output = str(data[i]["answer"]).strip()
    ok = answers_match(model_output, expected_output)
    hits += int(ok)
    print(f"\nTest Case {i+1}")
    print("Model Output:", model_output)
    print("Expected Output:", expected_output)
    print("Correct?", ok)
print(f"accuracy {hits}/{N}")
