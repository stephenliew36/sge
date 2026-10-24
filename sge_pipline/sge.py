import re

import nltk
import numpy as np
import spacy
from textstat import textstat

_nlp = None
_nltk_ready = False


def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise OSError(
                "spaCy model en_core_web_sm is not installed. "
                "Run: python -m spacy download en_core_web_sm"
            ) from exc
    return _nlp


def _ensure_nltk():
    global _nltk_ready
    if _nltk_ready:
        return
    for res in ("cmudict", "punkt", "punkt_tab"):
        nltk.download(res, quiet=True)
    _nltk_ready = True


class QueryClassifier:
    def __init__(self):
        self.patterns = {
            'factual': [r'^(what|who|when|where|which)\s', r'define\s', r'what is\s'], # llm
            'how_to': [r'^how (to|do|can)', r'steps to', r'guide to'], # lrm > llm
            'reasoning': [r'^why\s', r'explain why', r'reasoning behind'], # lrm
            'creative': [r'write (a|an)', r'create', r'generate', r'imagine'], # llm
            'comparison': [r'(compare|difference between|versus|vs)', r'better than', r'pros and cons'] # lrm
        }

    def classify(self, prompt):
        prompt_lower = prompt.lower()
        scores = {}
        for category, patterns in self.patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, prompt_lower))
            if score > 0:
                scores[category] = score
        return max(scores.items(), key=lambda x: x[1])[0] if scores else ' '


def get_complexity_metrics(query):
    _ensure_nltk()
    doc = get_nlp()(query)

    depth = max([len(list(token.ancestors)) for token in doc]) if len(doc) > 0 else 0
    negations = [tok for tok in doc if tok.dep_ == "neg" or tok.lemma_ in ["avoid", "exclude"]]
    logic_ops = [tok for tok in doc if tok.lemma_ in ["if", "then", "unless", "whenever"]]

    readability = textstat.flesch_kincaid_grade(query)
    score = (depth * 0.4) + (len(logic_ops) * 0.3) + (len(negations) * 0.3)

    return {
        "score": score,
        "depth": depth,
        "logic_count": len(logic_ops),
        "negation_count": len(negations),
        "readability": readability
    }


def ner_analysis(query=""):
    doc = get_nlp()(query)
    count = {}
    for ent in doc.ents:
        count[ent.label_] = count.get(ent.label_, 0) + 1

    # numerical conversion: domain-crossing only
    ignore_types = {"DATE", "TIME", "PERSON"}
    breadth = len([t for t in count if t not in ignore_types])
    if breadth <= 0:
        return 0.0
    if breadth == 1:
        return 0.25
    if breadth == 2:
        return 0.5
    if breadth == 3:
        return 0.75
    return 1.0


# Sec 3.2: intent one-hot, then depth, logic, negation, readability, breadth.
# Alg. 1 type_count is mapped into breadth (0 / .25 / .5 / .75 / 1).
INTENT_INDEX = {'reasoning': 0, 'comparison': 1, 'how_to': 2, 'creative': 3, 'factual': 4}
W_INTENT = np.array([4.0, 4.0, 1.0, -3.0, -0.5])
W_METRICS = np.array([0.3, 1.2, 1.6, 0.05, 0.8])
BIAS = -2.5
TAU = 0.5


def feature_vector(query):
    intent = QueryClassifier().classify(query)
    metrics = get_complexity_metrics(query)
    breadth = ner_analysis(query)
    features = np.zeros(10)
    if intent in INTENT_INDEX:
        features[INTENT_INDEX[intent]] = 1.0
    features[5] = metrics['depth']
    features[6] = metrics['logic_count']
    features[7] = metrics['negation_count']
    features[8] = metrics['readability']
    features[9] = breadth
    return features


def route_detail(query, tau=TAU):
    features = feature_vector(query)
    logit = float(np.dot(features, np.concatenate([W_INTENT, W_METRICS])) + BIAS)
    probability = 1.0 / (1.0 + np.exp(-logit))
    decision = "LRM" if probability > tau else "LLM"
    return decision, probability, features


def route_sge(query):
    decision, _, _ = route_detail(query)
    return decision


def route_simple(query):
    intent = QueryClassifier().classify(query)
    if intent in ("reasoning", "comparison", "how_to"):
        return "LRM"
    return "LLM"


if __name__ == "__main__":
    content = (
        "Trinamool Congress leader Mahua Moitra has moved the Supreme Court "
        "against her expulsion from the Lok Sabha over the cash-for-query "
        "allegations against her."
    )
    print(route_sge(content))
