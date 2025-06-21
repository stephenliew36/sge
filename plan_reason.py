import os
import openai

PLAN_MODEL = "gpt-4o"  # Plan with a more capable model
REASON_MODEL = "gpt-3.5-turbo"  # Use a cheaper model for reasoning

openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_plan(user_prompt: str) -> str:
    """Ask a powerful model to create a reasoning plan."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant that devises a reasoning plan."},
        {"role": "user", "content": user_prompt},
    ]
    response = openai.ChatCompletion.create(model=PLAN_MODEL, messages=messages)
    return response.choices[0].message["content"].strip()


def apply_plan(plan: str, user_prompt: str) -> str:
    """Use a cheaper model to reason using the plan from the larger model."""
    system_prompt = (
        "Follow this reasoning plan to produce the best answer:\n" + plan
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = openai.ChatCompletion.create(model=REASON_MODEL, messages=messages)
    return response.choices[0].message["content"].strip()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plan with GPT-4o then reason with GPT-3.5-turbo")
    parser.add_argument("prompt", help="User prompt for the assistant")
    args = parser.parse_args()

    plan = generate_plan(args.prompt)
    print("=== Plan ===")
    print(plan)

    result = apply_plan(plan, args.prompt)
    print("=== Result ===")
    print(result)


if __name__ == "__main__":
    main()
