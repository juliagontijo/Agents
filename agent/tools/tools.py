from agent.tools.web.tools import *
from copy import deepcopy
import json


TOOL_REGISTRY = {
    definition["name"]: {
        "definition": definition,
        "handler": {
            "search_web": search_web,
            "read_webpage": read_webpage,
        }[definition["name"]],
    }
    for definition in search_tools
}

class Tools:
    """
    A class that provides access to the registered tools.
    """

    def __init__(self, names: list[str]):
        self.__selected = {}

        for name in names:
            self.add(name)

    def add(self, name: str):
        if name not in TOOL_REGISTRY:
            raise ValueError(f"Tool '{name}' is not registered.")
        
        self.__selected[name] = TOOL_REGISTRY[name]

    @property
    def definitions(self) -> list[dict]:
        return [
            deepcopy(tool["definition"])
            for tool in self.__selected.values()
        ]

    def execute(self, name: str, arguments: dict):
        if name not in self.__selected:
            raise ValueError(f"Tool is not enabled for this agent: {name}")

        try:
            arguments = json.loads(arguments)
            result = self.__selected[name]["handler"](**arguments)
            output = json.dumps(result, ensure_ascii=False)

        except Exception as error:
            output = json.dumps({
                "error": type(error).__name__,
                "message": "Tool execution failed.",
            })
            
        return output
