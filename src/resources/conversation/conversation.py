from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

from ..memory import (
    MESSAGE_TYPES,
    load_jsonl_memory_records,
    read_memory_file,
    to_api_messages,
)


def new_id() -> str:
    return uuid4().hex


def now() -> int:
    return int(time.time())


@dataclass
class ConversationItem:
    client_id: str
    conversation_id: str
    type: str
    payload: dict[str, Any]
    turn_id: str | None = None
    step_id: int | None = None
    item_id: str = field(default_factory=new_id)
    recorded_at: int = field(default_factory=now)


@dataclass
class Conversation:
    client_id: str
    conversation_id: str = field(default_factory=new_id)
    created_at: int = field(default_factory=now)
    context: dict[str, Any] = field(default_factory=lambda: {
        "current_conversation": [],
        "memory": {"structured": [], "summaries": []},
    })
    _pending_events: list[ConversationItem] = field(
        default_factory=list, init=False, repr=False,
    )

    def add_event(
        self,
        type: str,
        payload: dict[str, Any],
        turn_id: str,
        step_id: int | None = None,
    ) -> None:
        self._pending_events.append(ConversationItem(
            client_id=self.client_id,
            conversation_id=self.conversation_id,
            type=type,
            payload=payload,
            turn_id=turn_id,
            step_id=step_id,
        ))
        if type in MESSAGE_TYPES:
            self.context["current_conversation"].append(payload)

    def build_messages(self, base_system_prompt: str = "") -> list[dict]:
        memory = self.context["memory"]
        system_prompt = base_system_prompt
        if memory["summaries"]:
            summary = "## Conversation memory\n\n" + "\n\n".join(memory["summaries"])
            system_prompt = "\n\n".join(part for part in (system_prompt, summary) if part)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        return messages + memory["structured"] + self.context["current_conversation"]

    def finish_turn(self, repository: "ConversationRepository") -> None:
        repository.save_events(self._pending_events)
        self.context["memory"]["structured"].extend(self.context["current_conversation"])
        self.context["current_conversation"].clear()
        self._pending_events.clear()


class ConversationRepository:
    def __init__(self, filename: str | Path):
        self.filename = Path(filename)

    def open(
        self,
        client_id: str,
        conversation_id: str | None = None,
        summary_paths: list[Path] | tuple[Path, ...] = (),
    ) -> Conversation:
        conversation_id = conversation_id or new_id()
        records = load_jsonl_memory_records(self.filename, client_id, conversation_id)
        lookup = self._read_lookup()
        metadata = next((record for record in lookup if (
            record.get("client_id") == client_id
            and record.get("conversation_id") == conversation_id
        )), None)
        created_at = metadata["created_at"] if metadata else (
            records[0].get("recorded_at", now()) if records else now()
        )
        conversation = Conversation(
            client_id=client_id,
            conversation_id=conversation_id,
            created_at=created_at,
        )
        memory = conversation.context["memory"]
        memory["structured"] = to_api_messages(records)
        # The caller explicitly selects summary files belonging to this context.
        memory["summaries"] = [
            text for path in summary_paths if (text := read_memory_file(path).strip())
        ]

        if metadata is None:
            lookup.append({
                "client_id": client_id,
                "conversation_id": conversation_id,
                "created_at": created_at,
            })
            path = self.filename.with_name("conversations_lookup.txt")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(
                json.dumps(record, ensure_ascii=False) + "\n" for record in lookup
            ), encoding="utf-8")
        return conversation

    def save_events(self, events: list[ConversationItem]) -> None:
        if not events:
            return
        content = "".join(
            json.dumps(asdict(event), ensure_ascii=False) + "\n" for event in events
        )
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        with self.filename.open("a", encoding="utf-8") as output:
            output.write(content)

    def _read_lookup(self) -> list[dict]:
        text = read_memory_file(self.filename.with_name("conversations_lookup.txt"))
        if not text.strip():
            return []
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            records = []
            for line in text.splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
            return records
        # The previous writer stored one pretty-printed conversation object.
        return [record] if isinstance(record, dict) else []
