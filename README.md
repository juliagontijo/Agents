# MyAgent

A hands-on project for learning how to build an agentic system from scratch: hardcore Python, no agent framework. The orchestration, tool calling, and conversation history are implemented directly. Regular libraries handle HTTP requests and web search.

## Implemented so far

- Calls OpenAI's Responses API through HTTP.
- Converts model responses into Python classes for text and tool calls.
- Lets each agent select tools by name from a registry.
- Searches the web with DDGS and reads text from web pages.
- Runs a loop: ask the model, execute its requested tools, return results, repeat until an answer or the step limit.

The current example explores companies and career opportunities with a job hunter prompt.

## Run

Requires Python 3.11+ and uv. Set `OPENAI_API_KEY` in your environment, then:

```bash
uv sync
uv run python script.py
```

Edit `script.py` to choose the prompt, model, and enabled tools. This is a learning project; search results and page extraction can be incomplete.
