# eval
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from evaluation import *
from main import *
import numpy as np
import json


# local data conversion
answers = []
math500_local = open("test.jsonl", "r")
count = 0
for line in math500_local:
    data = json.loads(line)
    answers.append(data["answer"])
    count += 1
math500_local.close()


# basic eval test-case
for times in range(500):
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
    test_case = LLMTestCase(
        input=f"{answer}",
        actual_output=f"{answers[times]}",
        retrieval_context=[""]
    )

    evaluate([test_case], [answer_relevancy_metric])