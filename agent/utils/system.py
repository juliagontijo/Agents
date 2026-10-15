import requests
import os

from .chat_template import ChatStructuredResponse

def call_model(messages, tools, model, temperature=0):
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": messages,
            "temperature": temperature,
            "tools": tools,
            "stream": False,
        },
        timeout=120
    )


    structured = ChatStructuredResponse(response)

    return structured
