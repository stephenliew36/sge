from openai import OpenAI
import pandas as pd
df = pd.read_json("hf://datasets/HuggingFaceH4/MATH-500/test.jsonl", lines=True)

client = OpenAI(api_key="sk-proj-Wxiayo3dxvvPv9_0HCu5B7IQbsEX7-Qzt3VWy9-"
    "s2HXUQEG4a8n4ajuI96zm46KG3vCfG3MmhuT3BlbkFJGP0u"
    "ZHGm9L2u9NN6S_nf7UTC7dEGNqjQ"
    "r3pSbXyIbGokvYOGbVRPqFOQpQGRmG2VOdn-pVpHcA")


userquestion = "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates. Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$"

plan_resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You’re a meta-reasoner: output a step-by-step plan for a smaller model."},
        {"role": "user", "content": userquestion}
    ]
)

reasoning_plan = plan_resp.choices[0].message.content
print(reasoning_plan)

exec_resp = client.chat.completions.create(
    model="o4-mini",
    messages=[
        {
            "role": "system",
            "content": "You're a frontier reasoning model, you will receive a plan to evaluate, verify its correct "
            "and continue with the plan to produce the result"
        },
        {
            "role": "user",
            "content": f"""{reasoning_plan} Execute the plan and verify to ensure accuracy."""
        }
    ]
)
print(exec_resp.choices[0].message.content)
