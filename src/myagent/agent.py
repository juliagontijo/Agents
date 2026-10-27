import json
import logging
from pathlib import Path
from uuid import uuid4

from ..resources.system.system import call_model
from ..resources.tools.execute import select_tools, execute_tool
from ..resources.tracer import Tracer
from ..resources.conversation.conversation import Conversation, ConversationRepository
from ..resources.prompts import SYSTEM_PROMPT


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEMORY_PATH = PROJECT_ROOT / "src" / "data" / "memory_eval.jsonl"
TRACE_PATH = PROJECT_ROOT / "traces" / "runs_eval.jsonl"


def run_agent(
    prompt: str,
    model: str,
    agent_tools: list[str],
    conversation: Conversation,
    repository: ConversationRepository,
    max_steps: int = 25,
    eval_mode: bool = False,
) -> tuple[str | None, Conversation, str]:
    """Run one turn using the loaded conversation and save only its new events."""
    if conversation._pending_events or conversation.context["current_conversation"]:
        raise RuntimeError("Save the pending conversation events before starting another turn.")
    tools = select_tools(agent_tools)
    turn_id = uuid4().hex
    trace = Tracer(
        client_id=conversation.client_id,
        conversation_id=conversation.conversation_id,
        model=model,
    )
    base_system_prompt = "" if eval_mode else SYSTEM_PROMPT
    completion = {
        "termination": "interrupted",
        "model_steps": 0,
        "final_answer": None,
        "error": None,
    }

    try:
        conversation.add_event(
            "user_message", {"role": "user", "content": prompt}, turn_id,
        )
        for step in range(1, max_steps + 1):
            completion["model_steps"] = step
            response, raw = call_model(
                conversation.build_messages(base_system_prompt), tools, model, trace=trace,
            )
            response_message = response.choices[0].message

            if response_message.content is not None:
                conversation.add_event(
                    "model_response",
                    {"role": "assistant", "content": response_message.content},
                    turn_id, step,
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
                conversation.add_event("tool_call", tool_call, turn_id, step)
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
                    conversation.add_event("tool_result", tool_output, turn_id, step)

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
            conversation.add_event(
                "run_result",
                {key: value for key, value in completion.items() if key != "final_answer"},
                turn_id,
                completion["model_steps"] or None,
            )
            conversation.finish_turn(repository)
        finally:
            trace.append("run_result", completion)
            try:
                trace.dump(TRACE_PATH)
            except Exception:
                logging.exception("Could not save execution trace")
