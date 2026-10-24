import spacy, re, nltk
import numpy as np
from textstat import textstat


try: 
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


def initialize_dependencies():
    for res in ['cmudict', 'punkt', 'punkt_tab']:
        nltk.download(res, quiet=True)
    try:
        from nltk.corpus import cmudict
        _ = cmudict.dict()
        print("NLP dependencies initialized")
    except Exception as e:
        print(f"Warning: NLTK pre-load encountered an issue: {e}")
initialize_dependencies()


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
    doc = nlp(query)
    
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
    doc = nlp(query)
    try:
        count = {}
        for ent in doc.ents:
            count[ent.label_] = count.get(ent.label_, 0) + 1

        # numerical conversion: domain-crossing only
        IGNORE_TYPES = {"DATE", "TIME", "PERSON"}
        breadth = len([t for t in count if t not in IGNORE_TYPES])
        if (breadth <= 0): return 0.0
        elif (breadth == 1): return 0.25
        elif (breadth == 2): return 0.5
        elif (breadth == 3): return 0.75
        else:  return 1.0 # multi-domain 
        
    except:
        raise ValueError
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
    # intent-only baseline; comments on QueryClassifier.patterns
    intent = QueryClassifier().classify(query)
    if intent in ("reasoning", "comparison", "how_to"):
        return "LRM"
    return "LLM"


if __name__  == "__main__":
    content = "Trinamool Congress leader Mahua Moitra has moved the Supreme Court against her expulsion from the Lok Sabha over the cash-for-query allegations against her. Moitra was ousted from the Parliament last week after the Ethics Committee of the Lok Sabha found her guilty of jeopardising national security by sharing her parliamentary portal's login credentials with businessman Darshan Hiranandani."
    print(route_sge(content))
