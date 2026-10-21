import json
import requests

from dataclasses import dataclass, field

class MessageRole:
    ASSISTANT = "assistant"
    USER = "user"
    SYSTEM = "system"
    DEVELOPER = "developer"

class ChatMessageType:
    FUNCTION_CALL = "function_call"
    FUNCTION_CALL_OUTPUT = "function_call_output"
    MESSAGE = "message"

class Status:
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    ERROR = "error"

@dataclass
class Function:
    name: str
    arguments: dict = field(default_factory=dict)

    @property
    def argument(self) -> str:
        return self.arguments

    
@dataclass
class ChatMessageToolCall:
    id: str = None
    type: str = ChatMessageType.FUNCTION_CALL
    function: Function = None


@dataclass
class ChatMessage: 
    content: str = None
    function_call: bool = False
    tool_calls: list[ChatMessageToolCall] = None

@dataclass
class ChatChoice:
    finish_reason: str
    message: ChatMessage


@dataclass
class ChatResponse:
    id: int
    choices: list[ChatChoice]
    created_at: int | None = None
    completed_at: int | None = None
    output: list[dict] = field(default_factory=list)



class ChatStructuredResponse:
    """
    Represents a structured response from the model, including the generated text and any tool calls.
    """

    def __new__(cls, response: requests.Response) -> ChatResponse:
    
        response.raise_for_status()
    
        # Parse the JSON yourself into ordinary Python dictionaries.
        data = json.loads(response.text)

        text_parts = []
        tool_calls = []
        # role = MessageRole.ASSISTANT

        for item in data.get("output", []):
            if item.get("type") == ChatMessageType.MESSAGE:
                role = item.get("role")
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        text_parts.append(part["text"])
            elif item.get("type") == ChatMessageType.FUNCTION_CALL:
                tool_calls.append(ChatMessageToolCall(
                    id=item["call_id"],
                    type=ChatMessageType.FUNCTION_CALL,
                    function=Function(
                        name=item["name"],
                        arguments=json.loads(item["arguments"]),
                    ),
                ))

        created_at = data.get("created_at")
        completed_at = data.get("completed_at")
        status = data.get("status")
        if status == Status.COMPLETED:
            finish_reason = "tool_calls" if tool_calls else "stop"
        elif status == Status.INCOMPLETE:
            finish_reason = (data.get("incomplete_details") or {}).get("reason") or status
        else:
            finish_reason = status

        return ChatResponse(
            id=data["id"],
            created_at=created_at,
            completed_at=completed_at,
            output=data.get("output", []),
            choices=[ChatChoice(
                finish_reason=finish_reason,
                message=ChatMessage(
                    content="".join(text_parts) or None,
                    function_call=bool(tool_calls),
                    tool_calls=tool_calls or None,
                ),
            )],
        )
