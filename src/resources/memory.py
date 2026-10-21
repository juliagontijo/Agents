"""Read complete local memory files or select matching summaries/JSONL records.

Reports use a header block, a blank line, and a prose body. A line of exactly
72 hyphens separates reports. Entries without an unambiguous header are ignored
so text mentioning an ID cannot select another client's report.
"""

import json
from pathlib import Path


def read_memory_file(path: Path) -> str:
    """Read the whole file as text; a missing file means no saved memory yet."""
    try:
        with Path(path).open("r", encoding="utf-8") as source:
            return source.read()
    except FileNotFoundError:
        return ""


def load_jsonl_memory_context(path: Path, client_id: str, conversation_id: str) -> list[dict]:
    """Return matching user, tool, and model payload objects in file order.

    Keep duplicates and each payload's structure. Skip invalid records and
    nonobject payloads; return an empty list for missing or unreadable files.
    """
    payloads = []
    try:
        with Path(path).open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    isinstance(record, dict)
                    and record.get("client_id") == client_id
                    and record.get("conversation_id") == conversation_id
                    and record.get("type") in ("user_message", "tool_call", "tool_result", "model_response")
                    and isinstance(record.get("payload"), dict)
                ):
                    payloads.append(record["payload"])
    except (OSError, UnicodeError):
        return []
    return payloads
