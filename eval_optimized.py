from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from main import generated_answers
import json


with open("test.jsonl", "r") as f:
    answers = [json.loads(line)["answer"] for line in f]

test_cases = []
for i in range(500):
    test_cases.append(LLMTestCase(
        input=generated_answers[i],
        actual_output=answers[i],
        retrieval_context=[""]
    ))

metric = AnswerRelevancyMetric(threshold=0.7)

evaluate(test_cases, [metric])