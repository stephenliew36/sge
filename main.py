from openai import OpenAI
import json

client = OpenAI(api_key="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")

generated_answers = []

with open("math500-sector two/batch-twenty.jsonl", "r") as f:
    for i, line in enumerate(f):
        if i >= 15:
            break
        q = json.loads(line)
        question = q['problem']
        print(f"{i+1}. {question}")

        plan_resp = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=[
                {"role": "system", "content": "You're a highly capable meta-reasoner. For the following question, plan your approach in at most two sentences."},
                {"role": "user", "content": question}
            ]
        )
        plan = plan_resp.choices[0].message.content

        exec_resp = client.chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "system", "content": "You're a frontier reasoning model. Answer the following question."},
                {"role": "user", "content": f"{plan}\n\nNow answer the question: {question}\n\nFormat your answer clearly for a text file (.txt). Label your answer as 'Answer {i+1}:' so it matches the question number."}
            ]
        )
        answer = exec_resp.choices[0].message.content
        generated_answers.append(answer) 
