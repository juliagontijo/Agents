from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Literal, Sequence

from .conversation import Conversation, ConversationItem, Summary
from ..common import estimate_tokens, now, read_json_file
from ..memory import MESSAGE_TYPES


@dataclass
class FileMetadata:
    path: str
    type: Literal["structured", "summary"]
    client_id: str
    conversation_id: str
    size_bytes: int = 0
    estimated_tokens: int = 0
    first_event_seq: int | None = None
    last_event_seq: int | None = None
    created_at: int = field(default_factory=now)
    updated_at: int = field(default_factory=now)


@dataclass
class SummaryReference:
    summary_id: str
    revision: int
    path: str
    from_event_seq: int
    through_event_seq: int


@dataclass
class ConversationLookup:
    client_id: str
    conversation_id: str
    created_at: int = field(default_factory=now)
    updated_at: int = field(default_factory=now)
    last_event_seq: int = 0
    structured_files: list[str] = field(default_factory=list)
    active_structured_file: str | None = None
    active_summaries: list[SummaryReference] = field(default_factory=list)


def is_covered(event_seq: int, summaries: Sequence[Summary | SummaryReference]) -> bool:
    return any(summary.from_event_seq <= event_seq <= summary.through_event_seq for summary in summaries)


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
            last_event_seq=record.last_event_seq,
        )
        events = self.load_events(client_id, conversation_id)
        conversation._memory_events = events
        conversation.context["memory"]["structured"] = [
            event.payload for event in events if event.type in MESSAGE_TYPES
        ]
        for reference in sorted(record.active_summaries, key=lambda item: item.from_event_seq):
            conversation._apply_summary(self._load_summary(reference, client_id, conversation_id))
        return conversation

    def load_events(
        self, client_id: str, conversation_id: str,
        from_event_seq: int = 1, through_event_seq: int | None = None,
    ) -> list[ConversationItem]:
        record = self._read_index()["lookup"][client_id][conversation_id]
        events = [
            ConversationItem(**event)
            for path in record.structured_files
            for event in read_json_file(self.root_dir / path)
        ]
        events.sort(key=lambda event: event.event_seq)
        if any(
            (event.client_id, event.conversation_id) != (client_id, conversation_id)
            or event.event_seq != position
            for position, event in enumerate(events, 1)
        ):
            raise ValueError("Stored events must belong to this conversation and have consecutive sequences.")
        return [
            event for event in events
            if event.event_seq >= from_event_seq
            and (through_event_seq is None or event.event_seq <= through_event_seq)
        ]

    def save_events(self, events: list[ConversationItem]) -> None:
        if not events:
            return
        client_id, conversation_id = events[0].client_id, events[0].conversation_id
        if any((event.client_id, event.conversation_id) != (client_id, conversation_id) for event in events):
            raise ValueError("Save events from one conversation at a time.")
        index = self._read_index()
        record = index["lookup"][client_id][conversation_id]
        self._recover_conversation(client_id, conversation_id, index)

        # A retry after a partial save must not append the same events twice.
        if events[0].event_seq <= record.last_event_seq:
            saved = {event.event_seq: event for event in self.load_events(client_id, conversation_id)}
            for event in events:
                if event.event_seq <= record.last_event_seq and saved.get(event.event_seq) != event:
                    raise ValueError("An event sequence is already occupied by a different event.")
            events = [event for event in events if event.event_seq > record.last_event_seq]
        if any(event.event_seq != record.last_event_seq + position for position, event in enumerate(events, 1)):
            raise ValueError("New events must continue the saved event sequence.")

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
            if metadata.first_event_seq is None:
                metadata.first_event_seq = event.event_seq
            metadata.last_event_seq = event.event_seq
            metadata.updated_at = now()
            record.last_event_seq = event.event_seq
            record.updated_at = max(record.updated_at, event.recorded_at)
        self._write_index(index)

    def save_summary(
        self, conversation: Conversation, summary: Summary, activate: bool = False,
        replace_summary_ids: Sequence[str] = (),
    ) -> None:
        """Save a summary; activation can explicitly replace fully covered active summaries."""
        if replace_summary_ids and not activate:
            raise ValueError("Replacing active summaries requires activate=True.")
        client_id, conversation_id = conversation.client_id, conversation.conversation_id
        saved = self.open(client_id, conversation_id)
        if activate:
            if summary not in conversation.context["memory"]["summaries"]:
                conversation._validate_summary(summary, replace_summary_ids)
            if summary not in saved.context["memory"]["summaries"]:
                saved._validate_summary(summary, replace_summary_ids)
        else:
            # Inactive summaries may cover raw events already summarized elsewhere.
            saved._memory_events = self.load_events(client_id, conversation_id)
            saved.context["memory"] = {
                "structured": [event.payload for event in saved._memory_events if event.type in MESSAGE_TYPES],
                "summaries": [],
            }
            saved._validate_summary(summary)

        folder = self.root_dir / client_id / "summaries"
        folder.mkdir(parents=True, exist_ok=True)
        prefix = f"{conversation_id}_{summary.summary_id}_r"
        revisions = sorted(folder.glob(f"{prefix}*.json"), key=lambda path: int(path.stem.rsplit("_r", 1)[1]))
        if revisions:
            previous = Summary(**read_json_file(revisions[-1]))
            if summary != previous and (
                summary.revision <= previous.revision
                or (summary.from_event_seq, summary.through_event_seq, summary.created_at)
                != (previous.from_event_seq, previous.through_event_seq, previous.created_at)
            ):
                raise ValueError("A summary revision must be newer and keep its range and creation time.")
        path = folder / f"{prefix}{summary.revision:04d}.json"
        text = json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n"
        if not path.exists():
            path.write_text(text, encoding="utf-8")
        relative_path = path.relative_to(self.root_dir).as_posix()
        index = self._read_index()
        record = index["lookup"][client_id][conversation_id]
        size_bytes, tokens = _file_size(text)
        index["file_metadata"].setdefault(relative_path, FileMetadata(
            path=relative_path, type="summary", client_id=client_id, conversation_id=conversation_id,
            size_bytes=size_bytes, estimated_tokens=tokens,
        ))
        if activate:
            record.active_summaries = [
                item for item in record.active_summaries
                if item.summary_id != summary.summary_id and item.summary_id not in replace_summary_ids
            ]
            record.active_summaries.append(SummaryReference(
                summary_id=summary.summary_id, revision=summary.revision, path=relative_path,
                from_event_seq=summary.from_event_seq, through_event_seq=summary.through_event_seq,
            ))
            record.active_summaries.sort(key=lambda item: item.from_event_seq)
        record.updated_at = now()
        self._write_index(index)
        conversation.updated_at = record.updated_at
        if activate and summary not in conversation.context["memory"]["summaries"]:
            conversation._apply_summary(summary, replace_summary_ids)

    def _read_index(self) -> dict:
        if not self.index_path.exists():
            return {"lookup": {}, "file_metadata": {}}
        index = read_json_file(self.index_path)
        for conversations in index["lookup"].values():
            for conversation_id, values in conversations.items():
                values["active_summaries"] = [SummaryReference(**item) for item in values["active_summaries"]]
                conversations[conversation_id] = ConversationLookup(**values)
        index["file_metadata"] = {path: FileMetadata(**values) for path, values in index["file_metadata"].items()}
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

    def _load_summary(self, reference: SummaryReference, client_id: str, conversation_id: str) -> Summary:
        summary = Summary(**read_json_file(self.root_dir / reference.path))
        if (
            summary.client_id, summary.conversation_id, summary.summary_id, summary.revision,
            summary.from_event_seq, summary.through_event_seq,
        ) != (
            client_id, conversation_id, reference.summary_id, reference.revision,
            reference.from_event_seq, reference.through_event_seq,
        ):
            raise ValueError("Summary does not match its active reference.")
        return summary

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
            events = [ConversationItem(**event) for event in read_json_file(target)]
            if any((event.client_id, event.conversation_id) != (client_id, conversation_id) for event in events):
                raise ValueError("Structured file contains events from another conversation.")
            text = target.read_text(encoding="utf-8")
            size_bytes, _ = _file_size(text)
            metadata = metadata or FileMetadata(
                path=path, type="structured", client_id=client_id, conversation_id=conversation_id,
                created_at=int(target.stat().st_ctime),
            )
            metadata.size_bytes = size_bytes
            metadata.estimated_tokens = sum(_file_size(line + "\n")[1] for line in text.split("\n") if line.strip())
            metadata.first_event_seq = min((event.event_seq for event in events), default=None)
            metadata.last_event_seq = max((event.event_seq for event in events), default=None)
            metadata.updated_at = int(target.stat().st_mtime)
            index["file_metadata"][path] = metadata
            if path not in record.structured_files:
                record.structured_files.append(path)
            changed = True
        record.structured_files.sort(key=lambda path: int(Path(path).stem.rsplit("_", 1)[1]))
        last_event_seq = max((index["file_metadata"][path].last_event_seq or 0 for path in record.structured_files), default=0)
        if last_event_seq != record.last_event_seq:
            record.last_event_seq = last_event_seq
            changed = True
        if changed:
            record.active_structured_file = record.structured_files[-1] if record.structured_files else None
            record.updated_at = max([record.updated_at] + [index["file_metadata"][path].updated_at for path in record.structured_files])
            self._write_index(index)
