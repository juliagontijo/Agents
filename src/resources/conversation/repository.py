from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Literal

from .conversation import Conversation, ConversationItem, trim_to_latest_compaction
from ..common import estimate_tokens, now, read_json_file


@dataclass
class FileMetadata:
    path: str
    type: Literal["structured", "summary"]
    client_id: str
    conversation_id: str
    size_bytes: int = 0
    estimated_tokens: int = 0
    created_at: int = field(default_factory=now)
    updated_at: int = field(default_factory=now)


@dataclass
class ConversationLookup:
    client_id: str
    conversation_id: str
    created_at: int = field(default_factory=now)
    updated_at: int = field(default_factory=now)
    structured_files: list[str] = field(default_factory=list)
    active_structured_file: str | None = None


def _file_size(text: str) -> tuple[int, int]:
    # Storage token estimate, separate from the model's actual input token usage.
    return len(text.encode("utf-8")), estimate_tokens(text)


class ConversationRepository:
    def __init__(self, root_dir: Path, max_file_bytes: int = 1_000_000, max_file_tokens: int | None = None):
        self.root_dir = Path(root_dir)
        self.index_path = self.root_dir / "index.json"
        self.max_file_bytes = max_file_bytes
        self.max_file_tokens = max_file_tokens

    def open(self, client_id: str, conversation_id: str | None = None) -> Conversation:
        index = self._read_index()
        if conversation_id is None:
            conversation = Conversation(client_id=client_id)
            record = ConversationLookup(
                client_id=client_id, conversation_id=conversation.conversation_id,
                created_at=conversation.created_at, updated_at=conversation.updated_at,
            )
            index["lookup"].setdefault(client_id, {})[record.conversation_id] = record
            self._write_index(index)
            return conversation

        record = index["lookup"].get(client_id, {}).get(conversation_id)
        if record is None:
            raise ValueError(f"Conversation {conversation_id!r} does not exist for client {client_id!r}.")
        self._recover_conversation(client_id, conversation_id, index)
        conversation = Conversation(
            client_id=client_id, conversation_id=conversation_id,
            created_at=record.created_at, updated_at=record.updated_at,
        )
        conversation.events = trim_to_latest_compaction(self.load_events(client_id, conversation_id))
        return conversation

    def load_events(
        self, client_id: str, conversation_id: str,
    ) -> list[ConversationItem]:
        record = self._read_index()["lookup"][client_id][conversation_id]
        events = [
            ConversationItem(**{
                key: value for key, value in event.items()
                if key in ConversationItem.__dataclass_fields__
            })
            for path in record.structured_files
            for event in read_json_file(self.root_dir / path)
        ]
        if any(
            (event.client_id, event.conversation_id) != (client_id, conversation_id)
            for event in events
        ):
            raise ValueError("Stored events must belong to this conversation.")
        return events

    def save_events(self, events: list[ConversationItem]) -> None:
        if not events:
            return
        client_id, conversation_id = events[0].client_id, events[0].conversation_id
        if any((event.client_id, event.conversation_id) != (client_id, conversation_id) for event in events):
            raise ValueError("Save events from one conversation at a time.")
        index = self._read_index()
        record = index["lookup"][client_id][conversation_id]
        self._recover_conversation(client_id, conversation_id, index)

        for event in events:
            text = json.dumps(asdict(event), ensure_ascii=False) + "\n"
            size_bytes, tokens = _file_size(text)
            path = self._select_append_file(record, index, size_bytes, tokens)
            if path is None:
                path = self._create_structured_file(record, index)
            with (self.root_dir / path).open("a", encoding="utf-8") as target:
                target.write(text)
            metadata = index["file_metadata"][path]
            metadata.size_bytes += size_bytes
            metadata.estimated_tokens += tokens
            metadata.updated_at = now()
            record.updated_at = max(record.updated_at, event.recorded_at)
        self._write_index(index)

    def _read_index(self) -> dict:
        if not self.index_path.exists():
            return {"lookup": {}, "file_metadata": {}}
        index = read_json_file(self.index_path)
        for conversations in index["lookup"].values():
            for conversation_id, values in conversations.items():
                conversations[conversation_id] = ConversationLookup(**{
                    key: value for key, value in values.items()
                    if key in ConversationLookup.__dataclass_fields__
                })
        index["file_metadata"] = {
            path: FileMetadata(**{
                key: value for key, value in values.items()
                if key in FileMetadata.__dataclass_fields__
            })
            for path, values in index["file_metadata"].items()
        }
        return index

    def _write_index(self, index: dict) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.index_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(index, default=asdict, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.index_path)

    def _select_append_file(
        self, record: ConversationLookup, index: dict, append_bytes: int, append_tokens: int,
    ) -> str | None:
        path = record.active_structured_file
        if path is None:
            return None
        metadata = index["file_metadata"][path]
        # An oversized event stays whole in its own segment.
        if metadata.size_bytes == 0:
            return path
        if metadata.size_bytes + append_bytes > self.max_file_bytes:
            return None
        if self.max_file_tokens is not None and metadata.estimated_tokens + append_tokens > self.max_file_tokens:
            return None
        return path

    def _create_structured_file(self, record: ConversationLookup, index: dict) -> str:
        path = f"{record.client_id}/structured/{record.conversation_id}_{len(record.structured_files) + 1:03d}.jsonl"
        target = self.root_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch(exist_ok=False)
        record.structured_files.append(path)
        record.active_structured_file = path
        index["file_metadata"][path] = FileMetadata(
            path=path, type="structured", client_id=record.client_id, conversation_id=record.conversation_id,
        )
        return path

    def _recover_conversation(self, client_id: str, conversation_id: str, index: dict) -> None:
        """Reconcile complete appends left behind by an interrupted index write."""
        record = index["lookup"][client_id][conversation_id]
        folder = self.root_dir / client_id / "structured"
        paths = {path.relative_to(self.root_dir).as_posix() for path in folder.glob(f"{conversation_id}_*.jsonl")}
        paths.update(record.structured_files)
        changed = False
        for path in sorted(paths):
            target = self.root_dir / path
            metadata = index["file_metadata"].get(path)
            if metadata is not None and metadata.size_bytes == target.stat().st_size:
                continue
            events = read_json_file(target)
            if any((event["client_id"], event["conversation_id"]) != (client_id, conversation_id) for event in events):
                raise ValueError("Structured file contains events from another conversation.")
            text = target.read_text(encoding="utf-8")
            size_bytes, _ = _file_size(text)
            metadata = metadata or FileMetadata(
                path=path, type="structured", client_id=client_id, conversation_id=conversation_id,
                created_at=int(target.stat().st_ctime),
            )
            metadata.size_bytes = size_bytes
            metadata.estimated_tokens = sum(_file_size(line + "\n")[1] for line in text.split("\n") if line.strip())
            metadata.updated_at = int(target.stat().st_mtime)
            index["file_metadata"][path] = metadata
            if path not in record.structured_files:
                record.structured_files.append(path)
            changed = True
        record.structured_files.sort(key=lambda path: int(Path(path).stem.rsplit("_", 1)[1]))
        if changed:
            record.active_structured_file = record.structured_files[-1] if record.structured_files else None
            record.updated_at = max([record.updated_at] + [index["file_metadata"][path].updated_at for path in record.structured_files])
            self._write_index(index)
