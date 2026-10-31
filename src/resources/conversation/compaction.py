from dataclasses import asdict
import json
from itertools import groupby
from typing import TYPE_CHECKING

from .conversation import Conversation, ConversationItem, Summary
from ..common import estimate_tokens
from ..memory import MESSAGE_TYPES
from ..prompts import SUMMARY_PROMPT
from ..system.system import call_model

if TYPE_CHECKING:
    from .repository import ConversationRepository
    from ..tracer import Tracer


DEFAULT_COMPACTION_TOKEN_LIMIT = 1_000_000_000_000


def _compaction_inputs(inputs) -> list[dict]:
    if isinstance(inputs, (list, tuple)):
        return [item for source in inputs for item in _compaction_inputs(source)]
    if isinstance(inputs, Summary):
        return [{"kind": "summary", **asdict(inputs)}]
    if isinstance(inputs, ConversationItem):
        return [{"kind": "event", **asdict(inputs)}]
    if isinstance(inputs, str):
        return [{"kind": "summary", "text": inputs}]
    if isinstance(inputs, dict):
        if "text" in inputs and "from_event_seq" in inputs:
            return [{**inputs, "kind": "summary"}]
        if "payload" in inputs and "event_seq" in inputs:
            return [{**inputs, "kind": "event"}]
        return [{"kind": "message", "payload": inputs}]
    raise TypeError("Compaction accepts messages, conversation events, summaries, or lists of these.")


def compact_memory(
    inputs: Summary | ConversationItem | dict | str | list | tuple,
    model: str,
    trace: "Tracer | None" = None,
) -> str:
    """Summarize messages, summaries, or a mix without changing their storage or context."""
    sources = _compaction_inputs(inputs)
    if not sources:
        raise ValueError("Compaction needs at least one message or summary.")
    response, _ = call_model([
        {"role": "system", "content": SUMMARY_PROMPT},
        {"role": "user", "content": json.dumps(sources, ensure_ascii=False)},
    ], [], model, trace=trace)
    text = response.choices[0].message.content or ""
    if not text.strip():
        raise ValueError("Compaction returned an empty summary.")
    return text.strip()


def _summary_events(conversation: Conversation, tokens_to_remove: int) -> list[ConversationItem]:
    selected = []
    removed_tokens = 0
    for turn_id, items in groupby(conversation._memory_events, key=lambda event: event.turn_id):
        turn = list(items)
        if turn[0].type != "user_message" or turn[-1].type != "run_result":
            if selected:
                break
            continue
        if selected and turn[0].event_seq != selected[-1].event_seq + 1:
            break
        if any(event.turn_id == turn_id for event in conversation._pending_events):
            break
        selected.extend(turn)
        removed_tokens += estimate_tokens([event.payload for event in turn if event.type in MESSAGE_TYPES])
        if removed_tokens >= tokens_to_remove:
            break
    return selected


def compact_conversation(
    conversation: Conversation,
    repository: "ConversationRepository",
    model: str,
    token_limit: int = DEFAULT_COMPACTION_TOKEN_LIMIT,
    system_prompt: str = "",
    tools: list[dict] | None = None,
    trace: "Tracer | None" = None,
    summary: Summary | None = None,
) -> None:
    """Compact older complete turns together with the current summary above an estimated input budget."""
    if token_limit < 1:
        raise ValueError("Compaction token limit must be positive.")
    summary = summary or next(iter(conversation.context["memory"]["summaries"]), None)
    while True:
        tokens = estimate_tokens({"input": conversation.build_messages(system_prompt), "tools": tools or []})
        if tokens < token_limit:
            return
        events = _summary_events(conversation, tokens - token_limit // 2)
        if not events:
            return
        text = compact_memory([summary, events] if summary else events, model, trace=trace)
        replace_summary_ids = [summary.summary_id] if summary else []
        summary = Summary(
            client_id=conversation.client_id,
            conversation_id=conversation.conversation_id,
            text=text,
            from_event_seq=min(summary.from_event_seq, events[0].event_seq) if summary else events[0].event_seq,
            through_event_seq=max(summary.through_event_seq, events[-1].event_seq) if summary else events[-1].event_seq,
        )
        repository.save_summary(conversation, summary, activate=True, replace_summary_ids=replace_summary_ids)
        if trace is not None:
            trace.append("memory_compaction", {
                "summary_id": summary.summary_id,
                "from_event_seq": summary.from_event_seq,
                "through_event_seq": summary.through_event_seq,
                "model": model,
                "estimated_tokens_before": tokens,
                "estimated_tokens_after": estimate_tokens({
                    "input": conversation.build_messages(system_prompt), "tools": tools or [],
                }),
            })
