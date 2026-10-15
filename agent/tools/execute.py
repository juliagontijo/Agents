from agent.tools.web.tools import *
from copy import deepcopy
import json


TOOL_REGISTRY = {
    tool["name"]: {
        "definition": tool,
        "handler": {
            "search_web": search_web,
            "read_webpage": read_webpage,
        }[tool["name"]],
    }
    for tool in search_tools
}
        
def select_tools(names: list[str]) -> list[dict]:
    """
    Select tools from the registry based on their names.
    """
    selected_tools = []
    for name in names:
        if name not in TOOL_REGISTRY:
            raise ValueError(f"Tool '{name}' is not registered.")
        selected_tools.append(deepcopy(TOOL_REGISTRY[name]["definition"]))

    return selected_tools

def execute_tool(name: str, arguments: dict):
    """
    Execute a tool by its name with the provided arguments.
    """
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Tool '{name}' is not registered.")

    try:
        result = TOOL_REGISTRY[name]["handler"](**arguments)
        output = json.dumps(result, ensure_ascii=False)
    except Exception as error:
        output = json.dumps({
            "error": type(error).__name__,
            "message": "Tool execution failed.",
        })
        
    return output

