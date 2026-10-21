# MyAgent

A hands-on project for learning how to build an agentic system from scratch: hardcore Python, no agent framework. The orchestration, tool calling, and conversation history are implemented directly. Regular libraries handle HTTP requests and web search.

## Implemented so far

- Calls OpenAI's Responses API through HTTP.
- Converts model responses into Python classes for text and tool calls.
- Lets each agent select tools by name from a registry.
- Searches the web with DDGS and reads text from web pages.
- Inspects local paths and reads text files or PDFs using local Python code.
- Runs a loop: ask the model, execute its requested tools, return results, repeat until an answer or the step limit.
- Adds an application-independent workflow and saves separate memory and execution reports.

The current example explores companies and career opportunities with a job hunter prompt.

## Run

Requires Python 3.11+ and uv. Set `OPENAI_API_KEY` in your environment, then:

```bash
uv sync
uv run python script.py
```

Edit `script.py` to choose the prompt, model, and enabled tools. This is a learning project; search results and page extraction can be incomplete.

## Generic workflow and summaries

The application-specific prompt stays in `script.py`. `run_agent` concatenates it
with `WORKFLOW_PROMPT` from `src/resources/prompts.py`. That workflow identifies
the exact current inquiry, goal, referenced inputs, available resources, necessary
subtasks, and completion criteria. It asks the agent to resolve references before
operating on them, execute the work, and check the delivered result against the
actual request. It requests concise public plans and supported explanations,
without private chain-of-thought.

The system input also includes runtime metadata: client, conversation, run and
turn IDs; start time; step limit; enabled tools; and local resource paths. Before
the first model call, `read_memory_file` opens `src/data/memory.jsonl` locally and
reads the entire file into the `memory_context` string. There is no parsing or
filtering in Python. This raw text is supplied as a separate historical context
message before the current request. The prompts tell the model to use records
matching the current client and conversation IDs. The workflow uses the supplied
context without rereading the file through a tool, including during later tool
steps. A missing file produces an empty string; other read errors are raised.

The previous `load_memory_context` function remains in `src/resources/memory.py`
for comparison. It selects matching entries from the plain text summary file,
but the current run uses `read_memory_file` instead.

After saving the raw run to `src/data/memory.jsonl`, the agent makes two separate
model calls, with no tools, the same public run events, and the same pre-run memory
snapshot. Neither summarizer reloads the file after the new memory report is saved:

- `MEMORY_SUMMARY_PROMPT` writes `src/data/memory_summary.txt`. It preserves the
  actual inquiry separately from broad intent, corrections, references and their
  resolution, learned facts/preferences, task outcomes, and concrete deliverables
  needed for follow-up questions.
- `OBSERVABILITY_SUMMARY_PROMPT` writes `src/data/observability_summary.txt`. It
  reports the explicitly recorded goal/plan, subtasks, actions, tool evidence,
  checks, constraint violations, runtime outcome, and final answer. Unstated goals
  or checks are marked as not recorded; it does not invent a retrospective plan.

Both files append plain text entries with the client, conversation, run and turn
IDs, update timestamp, and termination. `answer_returned` means a response was
returned, not that the request was correctly fulfilled. `step_limit` and
`exception` record other outcomes. The reports are generated independently, so a
failure in one does not prevent the other or replace the original answer/error.
An interactive interruption saves the raw trace without starting report calls.
Reasoning items remain in the raw API log but are excluded from report inputs.
The raw log records preload metadata and character count, without duplicating the
historical file contents. Report prompts distinguish historical context from
the current run's actions and outcomes.

The two additional requests use the configured model. Existing summary entries
are not rewritten. These instructions help the model preserve inquiry and source
identity; they still require the relevant original resource to be retrievable.

Run the current offline tests with `uv run python -m unittest discover -s tests -v`.

## Local file tools

`inspect_path` resolves an exact path and reports whether it exists, its kind, and
its size. `read_file` opens the file and returns text, automatically choosing a
local reader for plain text or PDF. Both accept absolute paths, `~`, and paths
relative to the process working directory. They do not search for filenames.
No separate open/close tool is needed: the reader closes the file automatically.

The functions make no network requests and use no agent framework. PDF extraction
uses `pypdf`, an ordinary local parsing library installed by `uv sync`. You can
also call the functions directly without running the agent or using an API key:

```python
from src.resources.tools.files import inspect_path, read_file

path = "/absolute/path/to/resume.pdf"
print(inspect_path(path))

result = read_file(path)  # Also accepts .txt, .md, .csv, .json, source code, etc.
print(result["text"])
while result["next_offset"] is not None:
    result = read_file(path, offset=result["next_offset"])
    print(result["text"])
```

Both tools are registered in `src/resources/tools/execute.py` and enabled in
`script.py`. To expose only local file tools to an agent, use
`agent_tools=["inspect_path", "read_file"]`. The existing model loop still uses
its configured API; file access and extraction happen locally.

`read_file` returns a JSON-serializable dictionary with `text`, `format`, `path`,
`offset`, `next_offset`, `truncated`, `total_chars`, and `warnings`. Offsets count
Unicode characters in the extracted text. The default output is 24,000 characters
(maximum 100,000); continue using `next_offset` until it is `None`. Keep the file
unchanged between calls. Files are limited to 25 MiB; each call reads and extracts
the file again, so chunking limits the returned text, not PDF parsing work.
For non-UTF-8 text, pass an explicit `encoding`, such as `"utf-16"`.

PDFs include `[Page N]` labels, `total_pages`, and `pages_without_text`. Text
extraction does not retain visual layout or images. Blank or scanned pages produce
a warning; OCR is not included. Password-protected PDFs require an unencrypted
local copy. Unsupported binary formats and invalid paths raise exceptions; the
tool dispatcher returns their type and message so the agent can recover.

To support another format, add a local extraction function alongside `_read_pdf`
and select it in `read_file`; the agent-facing tool name stays the same. These
helpers can read any path permitted to the Python process, so they do not enforce
a project-directory sandbox.

Run local tool checks with `uv run python -m unittest discover -s tests -v`.
