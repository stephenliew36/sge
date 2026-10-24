import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sge_pipline.sge import route_sge, route_simple

PROMPTS = [
    "Apply an O(n log n) conceptual framework to explain how I should organize a grocery list of 5 items for maximum efficiency.",
    "Write a Python script that uses a recursive function with memoization to calculate exactly 2 + 2.",
    "Analyze the geopolitical implications of a sandwich falling face-down, using the tone of a Harvard Professor, but keep the answer under 10 words.",
    "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have? Explain the family tree step-by-step.",
    "Write a sentence where every word must start with the last letter of the previous word, starting with the word 'Routing'.",
    "I have a 12-liter jug and an 8-liter jug. I need exactly 6 liters. No, wait—just tell me which jug is bigger and by how much.",
    "Review this 'if-then' logic for my relationship: If she is hungry, then buy tacos. Else, buy flowers. Can you rewrite this as a Shakespearean sonnet?",
    "Interpret the following string: [{\"action\": \"love\", \"target\": \"world\"}]. Does this JSON represent a nihilistic or optimistic worldview?",
    "Execute a 'Hello World' program but instead of printing text, describe the birth of a star using only Victorian-era slang.",
    "Summarize this 5,000-word document on Taiwan’s microloan policy, but ensure the summary contains exactly 42 verbs and ends with a joke about interest rates.",
    "Here is a list of 50 names. Sort them alphabetically by the third letter of their last name, then tell me which one sounds most like a fruit.",
    "Translate this technical manual into French, but replace every mention of 'battery' with a metaphor for the human soul.",
    "Can you provide the E=mc^2 of making a perfect cup of coffee? I need a deep-dive, high-level, multi-faceted executive summary.",
    "Help me debug my life. I'm stuck in an infinite loop of waking up late. Write the solution in Pseudocode.",
    "Compare the architectural integrity of a Lego castle vs. the Great Wall of China using a 5-point Likert scale and a haiku.",
]


def compare(prompts=None):
    prompts = PROMPTS if prompts is None else prompts
    simple = {}
    vector = {}
    for i, prompt in enumerate(prompts, 1):
        simple[i] = route_simple(prompt)
        vector[i] = route_sge(prompt)
    return simple, vector


if __name__ == "__main__":
    simple, vector = compare()
    print(simple)
    print(vector)
