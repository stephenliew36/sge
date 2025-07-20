from openai import OpenAI
from evaluation import math500problem

client = OpenAI(api_key="sk-proj-Wxiayo3dxvvPv9_0HCu5B7IQbsEX7-Qzt3VWy9-"
    "s2HXUQEG4a8n4ajuI96zm46KG3vCfG3MmhuT3BlbkFJGP0u"
    "ZHGm9L2u9NN6S_nf7UTC7dEGNqjQ"
    "r3pSbXyIbGokvYOGbVRPqFOQpQGRmG2VOdn-pVpHcA")

for i, q in enumerate(math500problem()):
    question = q['problem']
    print(f"{i+1}. {question}")

    plan_resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You're a highly capable meta-reasoner. For the following question, plan your approach in at most two sentences."},
            {"role": "user", "content": question}
        ]
    )
    plan = plan_resp.choices[0].message.content
    print("Plan:", plan)


    exec_resp = client.chat.completions.create(
        model="o4-mini",
        messages=[
            {"role": "system", "content": "You're a frontier reasoning model. Answer the following question."},
            {"role": "user", "content": f"{plan}\n\nNow answer the question: {question}\n\nFormat your answer clearly for a text file (.txt). Label your answer as 'Answer {i+1}:' so it matches the question number."}
        ]
    )
    answer = exec_resp.choices[0].message.content
    print("\n")
    print("Answer:", answer)
    print("-" * 100)




