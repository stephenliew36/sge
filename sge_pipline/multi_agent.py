import os
import asyncio
import json
from pipeline import BATCHES, LRM_SYSTEM, run_sge


def make_client():
    from openai import OpenAI
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"),
    )


def read_jsonl(path, limit=None):
    items = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
            if limit is not None and len(items) >= limit:
                break
    return items


async def run_all(q_count, file_path=None, batch=1, client=None):
    question_list = read_jsonl(file_path, limit=q_count)
    models = BATCHES[batch]
    if client is None:
        client = make_client()
    lrm_system = LRM_SYSTEM
    if models.get("reasoning"):
        lrm_system = f"{LRM_SYSTEM} Reasoning: {models['reasoning']}"

    results = []
    for item in question_list:
        prompt = item["problem"]
        out = await asyncio.to_thread(
            run_sge,
            prompt,
            models["llm"],
            models["lrm"],
            client,
            lrm_system=lrm_system,
        )
        results.append(out)
    return results


if __name__ == "__main__":
    results = asyncio.run(run_all(25, "math500/math500-sector one/batch-one.jsonl"))
    for res in results:
        print(res["route"], res["answer"])
