import os

try:
    from .sge import TAU, route_detail
except ImportError:
    from sge import TAU, route_detail

# Table 1. OpenRouter ids for the paper's LLM / LRM batches.
BATCHES = {
    1: {"llm": "openai/gpt-4.1-nano", "lrm": "openai/o4-mini"},
    2: {"llm": "microsoft/phi-4", "lrm": "openai/gpt-oss-20b", "reasoning": "medium"},
    3: {
        "llm": "mistralai/mistral-small-3.1-24b-instruct",
        "lrm": "deepseek/deepseek-r1-distill-llama-70b",
    },
}

PLAN_SYSTEM = (
    "You're a highly capable meta-reasoner. "
    "For the following question, plan your approach in at most 5 sentences."
)
DIRECT_SYSTEM = (
    "You are a math solver. Output ONLY the final answer. "
    "No steps, no units, no punctuation, no extra words."
)
LRM_SYSTEM = (
    "You're a frontier reasoning model. Answer the following question. "
    "If the plan does not seem correct, refine the plan, then give one final answer."
)


def make_client(api_key=None, base_url="https://openrouter.ai/api/v1", headers=None):
    from openai import OpenAI
    if base_url:
        key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Set OPENROUTER_API_KEY or OPENAI_API_KEY")
        kwargs = {"base_url": base_url, "api_key": key}
        if headers:
            kwargs["default_headers"] = headers
        return OpenAI(**kwargs)
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY")
    return OpenAI(api_key=key)


def complete(client, model, system, user, extra_body=None):
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if extra_body:
        kwargs["extra_body"] = extra_body
    resp = client.chat.completions.create(**kwargs)
    if not resp.choices or resp.choices[0].message.content is None:
        raise RuntimeError(f"empty completion from {model}")
    return resp.choices[0].message.content


def run_sge(
    query,
    llm_model,
    lrm_model,
    client,
    tau=TAU,
    llm_extra=None,
    lrm_extra=None,
    plan_system=None,
    direct_system=None,
    lrm_system=None,
    guided_user=None,
):
    # Sec 3.2.2: P > tau uses the reasoning path (LLM plan, then LRM verifies).
    # Otherwise the LLM answers directly so the LRM does not overthink.
    decision, probability, _ = route_detail(query, tau=tau)
    if decision == "LLM":
        answer = complete(client, llm_model, direct_system or DIRECT_SYSTEM, query, llm_extra)
        return {"route": "LLM", "probability": probability, "plan": None, "answer": answer}

    plan = complete(client, llm_model, plan_system or PLAN_SYSTEM, query, llm_extra)
    if guided_user is None:
        user = (
            f"{plan}\n\nNow answer the question:\n{query}\n"
            "Final answer only. Refine the plan if there are errors."
        )
    else:
        user = guided_user(plan, query)
    answer = complete(client, lrm_model, lrm_system or LRM_SYSTEM, user, lrm_extra)
    return {"route": "LRM", "probability": probability, "plan": plan, "answer": answer}


if __name__ == "__main__":
    samples = [
        "What is 2 + 2?",
        "If a 12-liter jug and an 8-liter jug must measure exactly 6 liters, how?",
    ]
    for q in samples:
        decision, probability, _ = route_detail(q)
        print(decision, round(probability, 3), q)
