from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from main import generated_answers
import os
from openai import OpenAI, RateLimitError
import time
import json

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("math500-sector two/batch-twenty.jsonl", "r") as f:
    answers = [json.loads(line)["answer"] for line in f]

test_cases = []
for i in range(15):
    test_cases.append(LLMTestCase(
        input=generated_answers[i],
        actual_output=answers[i],
        retrieval_context=[""]  
    ))

metric = AnswerRelevancyMetric(threshold=0.7)


batch_size = 5  
for i in range(0, len(test_cases), batch_size):
    batch = test_cases[i:i+batch_size]
    while True:
        try:
            evaluate(batch, [metric])
            break 
        except RateLimitError as e:
            print(f"Rate limit hit on batch starting with test case {i+1}, retrying in 2 seconds... ({e})")
            time.sleep(2)
        except Exception as e:
            print(f"Other error on batch starting with test case {i+1}: {e}")
            break
    time.sleep(1)