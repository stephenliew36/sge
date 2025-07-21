# eval
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from main import *
import numpy as np
import json



# local data conversion
answers = []
math500_local = open("test.jsonl", "r")
for line in math500_local:
    data = json.loads(line)
    answers.append(data["answer"])
math500_local.close()


# basic eval test-case
for times in range(500):
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
    test_case = LLMTestCase(
        input=f"{generated_answers[times]}",
        actual_output=f"{answers[times]}",
        retrieval_context=[""]
    )
    evaluate([test_case], [answer_relevancy_metric])