# comparison of convetional and vectorised routing
from sge import route_sge, route_simple



r1_resposnes = {}
r2_resposnes = {}


if __name__ == "__main__":
    test_prompts = [
    # --- Category: False Complexity 
    "Apply an O(n log n) conceptual framework to explain how I should organize a grocery list of 5 items for maximum efficiency.",
    "Write a Python script that uses a recursive function with memoization to calculate exactly 2 + 2.",
    "Analyze the geopolitical implications of a sandwich falling face-down, using the tone of a Harvard Professor, but keep the answer under 10 words.",

    # --- Category: Hidden Reasoning
    "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have? Explain the family tree step-by-step.",
    "Write a sentence where every word must start with the last letter of the previous word, starting with the word 'Routing'.",
    "I have a 12-liter jug and an 8-liter jug. I need exactly 6 liters. No, wait—just tell me which jug is bigger and by how much.",

    # --- Category: Semantic Overlap 
    "Review this 'if-then' logic for my relationship: If she is hungry, then buy tacos. Else, buy flowers. Can you rewrite this as a Shakespearean sonnet?",
    "Interpret the following string: [{\"action\": \"love\", \"target\": \"world\"}]. Does this JSON represent a nihilistic or optimistic worldview?",
    "Execute a 'Hello World' program but instead of printing text, describe the birth of a star using only Victorian-era slang.",

    # --- Category: Constraint Collisions 
    "Summarize this 5,000-word document on Taiwan’s microloan policy, but ensure the summary contains exactly 42 verbs and ends with a joke about interest rates.",
    "Here is a list of 50 names. Sort them alphabetically by the third letter of their last name, then tell me which one sounds most like a fruit.",
    "Translate this technical manual into French, but replace every mention of 'battery' with a metaphor for the human soul.",

    # --- Category: Vibe/Keyword Traps
    "Can you provide the E=mc^2 of making a perfect cup of coffee? I need a deep-dive, high-level, multi-faceted executive summary.",
    "Help me debug my life. I'm stuck in an infinite loop of waking up late. Write the solution in Pseudocode.",
    "Compare the architectural integrity of a Lego castle vs. the Great Wall of China using a 5-point Likert scale and a haiku."
    ]

    print("\nRouting Decisions")

    for q in range(len(test_prompts)): 
        decision1 = route_simple(test_prompts[q])
        r1_resposnes[q+1] = decision1
        decision2 = route_sge(test_prompts[q])
        r2_resposnes[q+1] = decision2


       

print(r1_resposnes)
print(r2_resposnes)