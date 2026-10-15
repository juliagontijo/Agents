import json
from datetime import datetime
from .utils.system import call_model
from agent.tools.tools import Tools


def run_agent(system_prompt: str, prompt: str, model: str, agent_tools: list[str], max_steps: int = 5):
    """
    Run agent with max 5 steps. 
    Generates a response to given prompt using OpenAI API until reaches max number of steps or response is complete.
    """

    system_prompt += f"\nCurrent local date: {datetime.now().date().isoformat()}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    tools = Tools(agent_tools)

    step = 0
    while step < max_steps:
        step += 1

        response = call_model(messages, tools.definitions, model, temperature=0)
        response_message = response.choices[0].message

        # no tool call
        if not response_message.tool_calls:
            print(response_message.content)
            return response_message.content

        if response_message.content is not None:
            messages.append({
                "role": "assistant",
                "content": response_message.content,
            })

        for call in response_message.tool_calls:
            result = tools.execute(
                call.function.name,
                call.function.arguments,
            )
            messages.append({
                "type": "function_call",
                "call_id": call.id,
                "name": call.function.name,
                "arguments": call.function.arguments,
            })
            messages.append({
                "type": "function_call_output", 
                "call_id": call.id, 
                "output": result,
            })

    raise RuntimeError("Max steps limit reached.")


