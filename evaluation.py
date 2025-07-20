from datasets import load_dataset   
ds = load_dataset("HuggingFaceH4/MATH-500", split="test")


def math500problem():
    problemcount = 0
    questions = []
    for question in ds:
        if problemcount >= 1:
            break
        questions.append(question)
        problemcount += 1
    return questions


for i, q in enumerate(math500problem()):
    print(f"{i+1}. {q['problem']}")
    print("—" * 40)


