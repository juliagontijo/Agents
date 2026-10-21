from dataclasses import asdict, dataclass, field
import time
from pathlib import Path
import json
from typing import Any
from uuid import uuid4


def new_id() -> str:
    return uuid4().hex


def now() -> str:
    return int(time.time())

@dataclass
class Attachment:
    name: str
    location: dict[str, str] # {type: local, path: ""} || {type: remote, path: ""} etc
    media_type: str | None = None # pdf,url,video
    size_bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict) # summary of it maybe

@dataclass
class ModelResponse:
    raw: dict[str, Any]

    @property
    def output(self) -> list[dict[str, Any]]:
        return self.raw["output"]


@dataclass
class Conversation:
    conversation_id: str = field(default_factory=new_id)
    client_id: str = None
    created_at: str = field(default_factory=now)
    metadata: dict[str, Any] = field(default_factory=dict) 


@dataclass 
class ConversationItem:
    client_id: str
    conversation_id: str
    # sequence: int
    type: str
    payload: dict[str, Any] 
    item_id: str = field(default_factory=new_id)
    recorded_at: str = field(default_factory=now)
    # schema_version: int = 1


# converts conversation items into API input friendly format - STILL TO BE USED (be diff APIs format friendly)
class ResponsesAPIContextBuilder:
    def build(self, events: list[ConversationItem]) -> list[dict]:
        input_items = []

        for event in events:
            if event.type == "user_message":
                input_items.append(event.payload)

            elif event.type == "model_response":
                input_items.extend(event.payload["output"])

            elif event.type == "tool_result":
                input_items.append(event.payload)

        return input_items

class ConversationRepository:
    def __init__(self, filename: str):
        self.filename = filename
        self.items: list[ConversationItem] = []

    def create(self, client_id: str, metadata: dict[str, Any] | None = None) -> str:
        conversation = Conversation(client_id=client_id, metadata={} if metadata is None else metadata)
        path = Path(self.filename).with_name("conversations_lookup.txt")

        self._write_json(path, conversation)
        
        return conversation.conversation_id

    def append(
        self,
        client_id: str,
        conversation_id: str,
        type: str,
        payload: dict[str, Any],
    ):

        item = ConversationItem(client_id=client_id, conversation_id=conversation_id, type=type, payload=payload)
        self.items.append(item)

    def dump(self):
        if not self.items:
            return

        content = "".join(
            json.dumps(asdict(item), ensure_ascii=False) + "\n"
            for item in self.items
        )

        path = Path(self.filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as output:
            output.write(content)

        self.items.clear()


    @staticmethod
    def _write_json( path: str, item: ConversationItem):

        path = Path(path)
        # Serialize before opening the file.
        content = json.dumps(
            asdict(item),
            ensure_ascii=False,
            indent=2,
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
