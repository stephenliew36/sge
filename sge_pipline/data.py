import json
from pathlib import Path


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


def read_problems(path, limit=None):
    path = Path(path)
    if path.suffix == ".jsonl":
        return read_jsonl(path, limit=limit)
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        items = obj.get("data", [])
    elif isinstance(obj, list):
        items = obj
    else:
        raise ValueError(f"unsupported dataset: {path}")
    if limit is not None:
        return items[:limit]
    return items
