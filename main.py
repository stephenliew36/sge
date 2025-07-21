from openai import OpenAI
import json

client = OpenAI(api_key="sk-proj-Wxiayo3dxvvPv9_0HCu5B7IQbsEX7-Qzt3VWy9-"
    "s2HXUQEG4a8n4ajuI96zm46KG3vCfG3MmhuT3BlbkFJGP0u"
    "ZHGm9L2u9NN6S_nf7UTC7dEGNqjQ"
    "r3pSbXyIbGokvYOGbVRPqFOQpQGRmG2VOdn-pVpHcA")

generated_answers = []

with open("test.jsonl", "r") as f:
    for i, line in enumerate(f):
        if i >= 500:
            break
        q = json.loads(line)
        question = q['problem']
        print(f"{i+1}. {question}")

        plan_resp = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
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
