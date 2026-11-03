from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .repository import ConversationRepository

from ..memory import to_api_messages

from ..common import new_id, now


@dataclass
class ConversationItem:
    client_id: str
    conversation_id: str

    type: str
    payload: dict[str, Any]

    item_id: str = field(default_factory=new_id)
    recorded_at: int = field(default_factory=now)


@dataclass
class Conversation:
    client_id: str
    conversation_id: str = field(default_factory=new_id)

    created_at: int = field(default_factory=now)
    updated_at: int = field(default_factory=now)

    events: list[ConversationItem] = field(default_factory=list)
    _pending_events: list[ConversationItem] = field(default_factory=list, init=False, repr=False)


def add_event(conversation: Conversation,
               type: str,
               payload: dict[str, Any]):

    event = ConversationItem(
        client_id=conversation.client_id,
        conversation_id=conversation.conversation_id,
        type=type,
        payload=payload,
    )
    conversation.events.append(event)
    conversation._pending_events.append(event)

    conversation.updated_at = now()
    return conversation


def build_messages(conversation: Conversation) -> list[dict]:
    return to_api_messages([
        {"type": event.type, "payload": event.payload}
        for event in conversation.events
    ])

def trim_to_latest_compaction(
        events: list[ConversationItem],
    ) -> list[ConversationItem]:
    for index in range(len(events) - 1, -1, -1):
        if events[index].type == "compaction":
            return events[index:]  # Includes the latest summary.
    return events[:]


def finish_turn(conversation: Conversation, repository: "ConversationRepository") -> None:
    repository.save_events(conversation._pending_events)
    conversation._pending_events.clear()
