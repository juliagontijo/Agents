import time
from uuid import uuid4
import json
from pathlib import Path
from typing import Any



def new_id() -> str:
    return uuid4().hex

def now() -> int:
    return int(time.time())


def read_json_file(path: str | Path) -> Any:
    path = Path(path)
    extension = path.suffix.lower()

    if extension not in {".json", ".jsonl"}:
        raise ValueError(f"Unsupported file format: {extension}")

    with path.open(encoding="utf-8") as source:
        if extension == ".json":
            return json.load(source)

        return [
            json.loads(line)
            for line in source
            if line.strip()
        ]


def estimate_tokens(value: Any) -> int:
    """Approximate text/JSON tokens at four characters per token."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return (len(text) + 3) // 4
