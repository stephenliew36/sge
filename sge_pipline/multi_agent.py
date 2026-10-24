import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sge_pipline.data import read_jsonl
from sge_pipline.pipeline import BATCHES, LRM_SYSTEM, make_client, run_sge

DEFAULT_DATA = Path(__file__).resolve().parent / "math500" / "math500-sector one" / "batch-one.jsonl"


async def run_all(q_count, file_path=None, batch=1, client=None):
    path = Path(file_path) if file_path else DEFAULT_DATA
    question_list = read_jsonl(path, limit=q_count)
    models = BATCHES[batch]
    if client is None:
        client = make_client()
    lrm_system = LRM_SYSTEM
    if models.get("reasoning"):
        lrm_system = f"{LRM_SYSTEM} Reasoning: {models['reasoning']}"

    results = []
    for item in question_list:
        out = await asyncio.to_thread(
            run_sge,
            item["problem"],
            models["llm"],
            models["lrm"],
            client,
            lrm_system=lrm_system,
        )
        results.append(out)
    return results


if __name__ == "__main__":
    results = asyncio.run(run_all(25))
    for res in results:
        print(res["route"], res["answer"])
