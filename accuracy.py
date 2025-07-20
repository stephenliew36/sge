# eval
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from evaluation import *
from main import *
import numpy as np
import json


# local data conversion
math500_local = open("test.jsonl", "r")

next(math500_local)
for i, line in enumerate(math500_local):
    data = json.loads(line)
    third_key = list(data.keys())[2]
    answer_500 = data[third_key]
    if i >= 500:
        break

math500_local.close()


answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
test_case = LLMTestCase(
    input=f"{answer}",
    actual_output=f"{answer_500}",
    retrieval_context=[""]
)

evaluate([test_case], [answer_relevancy_metric])