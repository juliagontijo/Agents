import json
from datetime import datetime
from .utils.system import call_model
from agent.tools.execute import select_tools, execute_tool


def run_agent(system_prompt: str, prompt: str, model: str, agent_tools: list[str], max_steps: int = 5) -> str:
    """
    Run agent with max 5 steps. 
    Generates a response to given prompt using OpenAI API until reaches max number of steps or response is complete.
    """

    system_prompt += f"\nCurrent local date: {datetime.now().date().isoformat()}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    tools = select_tools(agent_tools)

    step = 0
    while step < max_steps:
        step += 1

        response = call_model(messages, tools, model, temperature=0)
        response_message = response.choices[0].message

        if response_message.content is not None:
            messages.append({
                "role": "assistant",
                "content": response_message.content,
            })

        # no tool call
        if not response_message.tool_calls:
            return response_message.content

        for call in response_message.tool_calls:
            result = execute_tool(
                call.function.name,
                call.function.arguments,
            )
            messages.append({
                "type": "function_call",
                "call_id": call.id,
                "name": call.function.name,
                "arguments": json.dumps(call.function.arguments),
            })
            messages.append({
                "type": "function_call_output", 
                "call_id": call.id, 
                "output": result,
            })

    return "Error: Max steps limit reached."


