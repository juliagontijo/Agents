import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..resources.system.system import call_model
from ..resources.tools.execute import select_tools, execute_tool
from ..resources.tracer import Tracer
from ..resources.conversation.conversation import Conversation, build_messages, add_event, finish_turn, trim_to_latest_compaction
from ..resources.conversation.compaction import DEFAULT_COMPACTION_TOKEN_LIMIT, compact_conversation
from ..resources.prompts import SYSTEM_PROMPT

if TYPE_CHECKING:
    from ..resources.conversation.repository import ConversationRepository


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEMORY_PATH = PROJECT_ROOT / "src" / "data" / "memory"
TRACE_PATH = PROJECT_ROOT / "traces" / "runs_eval.jsonl"


def run_agent(
    prompt: str,
    model: str,
    agent_tools: list[str],
    conversation: Conversation,
    repository: "ConversationRepository",
    max_steps: int = 25,
    eval_mode: bool = False,
    compaction_token_limit: int = DEFAULT_COMPACTION_TOKEN_LIMIT,
    summary_model: str | None = None,
) -> tuple[str | None, Conversation, str]:

    tools = select_tools(agent_tools)
    trace = Tracer(
        client_id=conversation.client_id,
        conversation_id=conversation.conversation_id,
        model=model,
    )

    system_prompt = "" if eval_mode else SYSTEM_PROMPT
    token_count = 0
    completion = {
        "termination": "interrupted",
        "model_steps": 0,
        "final_answer": None,
        "error": None,
    }

    add_event(
        conversation,
        "system_prompt", {"role": "system", "content": system_prompt},
    )

    add_event(
        conversation,
        "user_message", {"role": "user", "content": prompt},
    )

    try:
        for step in range(1, max_steps + 1):
            completion["model_steps"] = step
            response, raw = call_model(
                build_messages(conversation), tools, model, trace=trace,
            )
            token_count = raw["usage"]["total_tokens"]
            response_message = response.choices[0].message

            if response_message.content is not None:
                add_event(
                    conversation,
                    "model_response",
                    {"role": "assistant", "content": response_message.content},
                )

            if not response_message.tool_calls:
                completion.update(
                    termination="answer_returned",
                    final_answer=response_message.content,
                )
                return response_message.content, conversation, trace.run_id

            for call in response_message.tool_calls:
                tool_call = {
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.function.name,
                    "arguments": json.dumps(call.function.arguments),
                }
                trace.append("tool_call", {
                    "call_id": call.id,
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                })
                add_event(
                    conversation,"tool_call", tool_call)
                result = json.dumps({"error": "Tool execution interrupted"})
                try:
                    result = execute_tool(call.function.name, call.function.arguments)
                except Exception as error:
                    result = json.dumps({
                        "error": type(error).__name__, "message": str(error),
                    })
                    raise
                finally:
                    tool_output = {
                        "type": "function_call_output",
                        "call_id": call.id,
                        "output": result,
                    }
                    trace.append("tool_result", {
                        "call_id": call.id, "output": result,
                    })
                    add_event(
                        conversation,"tool_result", tool_output)

        answer = "Error: Max steps limit reached."
        completion.update(termination="step_limit", final_answer=answer)
        return answer, conversation, trace.run_id
    except Exception as error:
        completion.update(
            termination="exception",
            error={"type": type(error).__name__, "message": str(error)},
        )
        raise
    finally:
        try:
            add_event(
                conversation,
                "run_result",
                {key: value for key, value in completion.items() if key != "final_answer"},
            )
            finish_turn(conversation, repository)
            try:
                compacted_memory = compact_conversation(
                    conversation, summary_model or model, token_count,
                    token_limit=compaction_token_limit, trace=trace,
                )
            except Exception:
                logging.exception("Could not compact conversation")
                compacted_memory = None
            if compacted_memory is not None:
                add_event(
                    conversation,
                    "compaction",
                    {"role": "assistant", "content": compacted_memory},
                )
                finish_turn(conversation, repository)
                conversation.events = trim_to_latest_compaction(conversation.events)

                trace.append("memory_compaction", {
                    "summary_id": conversation.events[-1].item_id,
                    "model": summary_model or model,
                    "request_total_tokens": token_count,
                })
        finally:
            trace.append("run_result", completion)
            try:
                trace.dump(TRACE_PATH)
            except Exception:
                logging.exception("Could not save execution trace")
