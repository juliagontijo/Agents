"""Load stored conversation events and project their payloads into model input."""

import json
from pathlib import Path


MESSAGE_TYPES = {"user_message", "model_response", "tool_call", "tool_result"}


def read_memory_file(path: Path) -> str:
    """Read the whole file as text; a missing file means no saved memory yet."""
    try:
        with Path(path).open("r", encoding="utf-8") as source:
            return source.read()
    except FileNotFoundError:
        return ""


def load_jsonl_memory_records(path: Path, client_id: str, conversation_id: str) -> list[dict]:
    """Load this conversation's records in file order, including their metadata."""
    records = []
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
                    and isinstance(record.get("payload"), dict)
                ):
                    records.append(record)
    except (OSError, UnicodeError):
        return []
    return records


def to_api_messages(records: list[dict]) -> list[dict]:
    """Keep message/tool payloads; storage metadata and system prompts stay out."""
    return [record["payload"] for record in records if record.get("type") in MESSAGE_TYPES]


def load_jsonl_memory_context(path: Path, client_id: str, conversation_id: str) -> list[dict]:
    return to_api_messages(load_jsonl_memory_records(path, client_id, conversation_id))
