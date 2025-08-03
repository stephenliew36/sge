from openai import OpenAI
import json

client = OpenAI(api_key="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")
generated_answers = []
with open("aime.jsonl", "r") as f:
    for i, line in enumerate(f):
        if i >= 3:
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
