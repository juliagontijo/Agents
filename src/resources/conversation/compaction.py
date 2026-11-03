import json
from typing import TYPE_CHECKING

from .conversation import Conversation
from ..prompts import SUMMARY_PROMPT
from ..system.system import call_model

if TYPE_CHECKING:
    from ..tracer import Tracer


DEFAULT_COMPACTION_TOKEN_LIMIT = 1_000_000_000_000


def compact_memory(
    conversation: Conversation,
    model: str,
    system_prompt: str = SUMMARY_PROMPT,
    trace: "Tracer | None" = None,
) -> str:
    """Summarize messages, summaries, or a mix without changing their storage or context."""

    if not conversation.events:
        raise ValueError("Compaction needs at least one message or summary.")

    messages = []
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt,
        })
    messages.append({
        "role": "user",
        "content": json.dumps([
            {"type": event.type, "payload": event.payload}
            for event in conversation.events
        ], ensure_ascii=False),
    })

    response, _ = call_model(messages, [], model, trace=trace)

    text = response.choices[0].message.content or ""
    if not text.strip():
        raise ValueError("Compaction returned an empty summary.")

    return text.strip()

def compact_conversation(
    conversation: Conversation,
    model: str,
    token_count: int,
    token_limit: int = DEFAULT_COMPACTION_TOKEN_LIMIT,
    trace: "Tracer | None" = None,
) -> str | None:
    """Summarize active context when the last response's token usage reaches the limit."""

    if token_limit < 1:
        raise ValueError("Compaction token limit must be positive.")

    if token_count < token_limit:
        return None

    return compact_memory(conversation, model, trace=trace)
