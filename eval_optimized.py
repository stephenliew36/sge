from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from main import generated_answers
import os
from openai import OpenAI
import json
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("math500-sector one/batch-ten.jsonl", "r") as f:
    answers = [json.loads(line)["answer"] for line in f]

test_cases = []
for i in range(25):
    test_cases.append(LLMTestCase(
        input=generated_answers[i],
        actual_output=answers[i],
        retrieval_context=[""]
    ))

metric = AnswerRelevancyMetric(threshold=0.7)

evaluate(test_cases, [metric])