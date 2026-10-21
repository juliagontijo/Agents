import requests
import os

from ..chat.chat_template import ChatStructuredResponse

def call_model(messages, tools, model, trace):
    payload = {
        "model": model,
        "input": messages,
        "tools": tools,
        "stream": False,
    }

    if trace is not None:
            trace.append("model_request", payload)

    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120
    )

    try:
        raw = response.json()
    except ValueError:
        raw = {"unparsed_text": response.text}

    # Record before parsing, so HTTP/parser failures still leave evidence.
    if trace is not None:
        trace.append("model_response", {
            "status_code": response.status_code,
            "body": raw,
        })

    structured = ChatStructuredResponse(response)
    return structured, raw
