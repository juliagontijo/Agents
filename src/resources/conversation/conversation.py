from dataclasses import dataclass, field
import time
from typing import Any, Sequence, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from .repository import ConversationRepository

from ..memory import MESSAGE_TYPES

from ..common import new_id, now


@dataclass
class ConversationItem:
    client_id: str
    conversation_id: str

    event_seq: int
    type: str
    payload: dict[str, Any]

    turn_id: str
    step_id: int | None = None

    item_id: str = field(default_factory=new_id)
    recorded_at: int = field(default_factory=now)

@dataclass
class Summary:
    client_id: str
    conversation_id: str

    text: str
    from_event_seq: int
    through_event_seq: int

    summary_id: str = field(default_factory=new_id)
    revision: int = 1

    created_at: int = field(default_factory=now)
    updated_at: int = field(default_factory=now)

@dataclass
class Conversation:
    client_id: str
    conversation_id: str = field(default_factory=new_id)

    created_at: int = field(default_factory=now)
    updated_at: int = field(default_factory=now)
    last_event_seq: int = 0

    context: dict[str, Any] = field(default_factory=lambda: {
        "current_conversation": [],
        "memory": {"structured": [], "summaries": []},
    })

    # New events from the current turn that haven’t been saved yet
    _pending_events: list[ConversationItem] = field(
        default_factory=list, init=False, repr=False,
    )
    # Previously saved events that haven’t been replaced by the summary
    _memory_events: list[ConversationItem] = field(
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
            event_seq=self.last_event_seq + 1,
            type=type,
            payload=payload,
            turn_id=turn_id,
            step_id=step_id,
        ))
        self.last_event_seq += 1
        self.updated_at = now()
        if type in MESSAGE_TYPES:
            self.context["current_conversation"].append(payload)

    def build_messages(self, base_system_prompt: str = "") -> list[dict]:
        memory = self.context["memory"]
        system_prompt = base_system_prompt
        if memory["summaries"]:
            summary = "## Conversation memory\n\n" + "\n\n".join(
                f"### Events {item.from_event_seq}-{item.through_event_seq}\n\n{item.text}"
                for item in sorted(memory["summaries"], key=lambda item: item.from_event_seq)
            )
            system_prompt = "\n\n".join(part for part in (system_prompt, summary) if part)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        return messages + memory["structured"] + self.context["current_conversation"]

    def finish_turn(self, repository: "ConversationRepository") -> None:
        repository.save_events(self._pending_events)
        self._memory_events.extend(self._pending_events)
        self.context["memory"]["structured"].extend(self.context["current_conversation"])
        self.context["current_conversation"].clear()
        self._pending_events.clear()

    def _apply_summary(self, summary: Summary, replace_summary_ids: Sequence[str] = ()) -> None:
        self._validate_summary(summary, replace_summary_ids)
        summaries = [
            existing
            for existing in self.context["memory"]["summaries"]
            if existing.summary_id != summary.summary_id and existing.summary_id not in replace_summary_ids
        ]
        summaries.append(summary)
        summaries.sort(key=lambda item: item.from_event_seq)


        event_seq_init = summary.from_event_seq
        event_seq_end = summary.through_event_seq

        remaining_events = [event for event in self._memory_events if not (event_seq_init <= event.event_seq <= event_seq_end)]
        structured = [event.payload for event in remaining_events if event.type in MESSAGE_TYPES]

        self.context["memory"]["summaries"] = summaries
        self._memory_events = remaining_events
        self.context["memory"]["structured"] = structured

    def _validate_summary(self, summary: Summary, replace_summary_ids: Sequence[str] = ()) -> None:
        if (summary.client_id, summary.conversation_id) != (self.client_id, self.conversation_id):
            raise ValueError("Summary must belong to this client and conversation.")

        start, end = summary.from_event_seq, summary.through_event_seq
        if type(start) is not int or type(end) is not int or not 1 <= start <= end <= self.last_event_seq:
            raise ValueError("Summary range must contain valid conversation event sequences.")
        if type(summary.revision) is not int or summary.revision < 1:
            raise ValueError("Summary revision must be a positive integer.")
        if not isinstance(summary.text, str) or not summary.text.strip():
            raise ValueError("Summary text must not be empty.")

        structured = [event.payload for event in self._memory_events if event.type in MESSAGE_TYPES]
        if self.context["memory"]["structured"] != structured:
            raise ValueError("Load saved event metadata together with structured messages before applying a summary.")

        active = {item.summary_id: item for item in self.context["memory"]["summaries"]}
        replacements = set(replace_summary_ids)
        if not replacements <= active.keys():
            raise ValueError("Only active summaries can be replaced.")
        if replacements and summary.summary_id in active:
            raise ValueError("A merged summary must have a new summary ID.")
        covered = [active[summary_id] for summary_id in replacements]
        if any(not start <= item.from_event_seq <= item.through_event_seq <= end for item in covered):
            raise ValueError("A merged summary must fully cover every summary it replaces.")

        previous = None
        for existing in active.values():
            if existing.summary_id in replacements:
                continue
            if existing.summary_id == summary.summary_id:
                previous = existing
            elif start <= existing.through_event_seq and existing.from_event_seq <= end:
                raise ValueError("Active summary ranges must not overlap.")

        if previous is not None:
            if (start, end) != (previous.from_event_seq, previous.through_event_seq):
                raise ValueError("A summary revision must keep its original event range.")
            if summary.revision <= previous.revision:
                raise ValueError("A summary revision must be newer than the active revision.")
            if summary.created_at != previous.created_at:
                raise ValueError("A summary revision must keep its original creation time.")
            return

        events = [event for event in self._memory_events if start <= event.event_seq <= end]
        ranges = sorted(
            [(event.event_seq, event.event_seq) for event in events]
            + [(item.from_event_seq, item.through_event_seq) for item in covered]
        )
        next_seq = start
        for first, last in ranges:
            if first != next_seq:
                raise ValueError("Summary range must contain a contiguous range of saved events or replaced summaries.")
            next_seq = last + 1
        if next_seq != end + 1:
            raise ValueError("Summary range must contain a contiguous range of saved events or replaced summaries.")
        turns = {}
        for event in events:
            turns.setdefault(event.turn_id, []).append(event)
        if any(
            turn[0].type != "user_message" or turn[-1].type != "run_result"
            for turn in turns.values()
        ):
            raise ValueError("Summary range must start and end at complete turn boundaries.")
        if any(
            event.turn_id in turns and not start <= event.event_seq <= end
            for event in self._memory_events + self._pending_events
        ):
            raise ValueError("Summary range must not split a turn or include pending events.")
