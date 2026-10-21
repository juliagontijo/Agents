from dataclasses import asdict
from datetime import datetime
import json
import logging
from pathlib import Path
from uuid import uuid4

from ..resources.system.system import call_model
from ..resources.tools.execute import select_tools, execute_tool
from ..resources.tracer import Tracer
from ..resources.conversation.conversation import ConversationRepository
from ..resources.memory import load_jsonl_memory_context
from ..resources.prompts import (
    SYSTEM_PROMPT
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEMORY_PATH = PROJECT_ROOT / "src" / "data" / "memory.jsonl"
TRACE_PATH = PROJECT_ROOT / "traces" / "runs.jsonl"


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(sep=" ", timespec="seconds")

def run_agent(prompt: str, model: str, agent_tools: list[str], authenticated_client_id: str, conversation_id: str = None, max_steps: int = 25) -> tuple[str | None, str | int]:
    """Execute a domain-neutral workflow and save its events and two reports."""
    tools = select_tools(agent_tools)
    conversation = ConversationRepository(MEMORY_PATH)
    if conversation_id is None:
        conversation_id = conversation.create(client_id=authenticated_client_id)

    trace = Tracer(
        client_id=authenticated_client_id,
        conversation_id=conversation_id,
        model=model,
    )

    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    user_message = {"role": "user", "content": prompt}
    context = [system_message]
    context.extend(load_jsonl_memory_context(
        MEMORY_PATH, authenticated_client_id, conversation_id
    ))
    context.append(user_message)
    conversation.append(
        client_id=authenticated_client_id,
        conversation_id=conversation_id,
        type="system_message",
        payload=system_message,
    )
    conversation.append(
        client_id=authenticated_client_id,
        conversation_id=conversation_id,
        type="user_message",
        payload=user_message,
    )
    completion = {
        "termination": "interrupted",
        "model_steps": 0,
        "final_answer": None,
        "error": None,
    }

    try:
        for step in range(max_steps):
            completion["model_steps"] = step
            response, raw = call_model(context, tools, model, trace=trace)
            response_message = response.choices[0].message

            if response_message.content is not None:
                response_payload = {"role": "assistant", "content": response_message.content}
                context.append(response_payload)
                conversation.append(
                    client_id=authenticated_client_id,
                    conversation_id=conversation_id,
                    type="model_response",
                    payload=response_payload,
                )

            if not response_message.tool_calls:
                completion.update(
                    termination="answer_returned",
                    final_answer=response_message.content,
                )
                return response_message.content, conversation_id

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
                conversation.append(
                    client_id=authenticated_client_id,
                    conversation_id=conversation_id,
                    type="tool_call",
                    payload=tool_call,
                )
                context.append(tool_call)
                result = execute_tool(call.function.name, call.function.arguments)
                tool_output = {
                    "type": "function_call_output",
                    "call_id": call.id,
                    "output": result,
                }
                trace.append("tool_result", {
                    "call_id": call.id,
                    "output": result,
                })
                conversation.append(
                    client_id=authenticated_client_id,
                    conversation_id=conversation_id,
                    type="tool_result",
                    payload=tool_output,
                )
                context.append(tool_output)

        answer = "Error: Max steps limit reached."
        completion.update(termination="step_limit", final_answer=answer)
        return answer, conversation_id
    except Exception as error:
        completion.update(
            termination="exception",
            error={"type": type(error).__name__, "message": str(error)},
        )
        raise
    finally:
        try:
            conversation.append(
                client_id=authenticated_client_id,
                conversation_id=conversation_id,
                type="run_result",
                payload={
                    key: value
                    for key, value in completion.items()
                    if key != "final_answer"
                },
            )
            conversation.dump()
        finally:
            # Includes termination, final answer, step count, and any error.
            trace.append("run_result", completion)

            try:
                trace.dump(TRACE_PATH)
            except Exception:
                # A trace-writing failure should not replace the agent's error
                # or prevent its successful answer from being returned.
                logging.exception("Could not save execution trace")
