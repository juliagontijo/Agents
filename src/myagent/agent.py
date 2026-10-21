from dataclasses import asdict
from datetime import datetime
import json
import logging
from pathlib import Path
from uuid import uuid4

from ..resources.system.system import call_model
from ..resources.tools.execute import select_tools, execute_tool
from ..resources.conversation.conversation import ConversationRepository
from ..resources.memory import load_jsonl_memory_context
from ..resources.prompts import (
    SYSTEM_PROMPT
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEMORY_PATH = PROJECT_ROOT / "src" / "data" / "memory.jsonl"


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(sep=" ", timespec="seconds")

def run_agent(prompt: str, model: str, agent_tools: list[str], authenticated_client_id: str, conversation_id: str = None, turn: int = 0, max_steps: int = 25) -> str:
    """Execute a domain-neutral workflow and save its events and two reports."""
    tools = select_tools(agent_tools)
    conversation = ConversationRepository(MEMORY_PATH)
    if conversation_id is None:
        conversation_id = conversation.create(client_id=authenticated_client_id)

    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    user_message = {"role": "user", "content": prompt}
    context = [system_message, user_message]
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

    context.extend(load_jsonl_memory_context(
        MEMORY_PATH, authenticated_client_id, conversation_id
    ))

    try:
        for step in range(max_steps):
            completion["model_steps"] = step
            response = call_model(context, tools, model)
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
                return response_message.content

            for call in response_message.tool_calls:
                result = execute_tool(call.function.name, call.function.arguments)
                tool_call = {
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.function.name,
                    "arguments": json.dumps(call.function.arguments),
                }
                tool_output = {
                    "type": "function_call_output",
                    "call_id": call.id,
                    "output": result,
                }
                context.append(tool_call)
                context.append(tool_output)
                conversation.append(
                    client_id=authenticated_client_id,
                    conversation_id=conversation_id,
                    type="tool_call",
                    payload=tool_call,
                )
                conversation.append(
                    client_id=authenticated_client_id,
                    conversation_id=conversation_id,
                    type="tool_result",
                    payload=tool_output,
                )

        answer = "Error: Max steps limit reached."
        completion.update(termination="step_limit", final_answer=answer)
        return answer
    except Exception as error:
        completion.update(
            termination="exception",
            error={"type": type(error).__name__, "message": str(error)},
        )
        raise
    finally:
        conversation.append(
            client_id=authenticated_client_id,
            conversation_id=conversation_id,
            type="run_result",
            payload={key: value for key, value in completion.items() if key != "final_answer"},
        )
        conversation.dump()
