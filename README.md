# MyAgent

A Python agent built from scratch, without an agent framework. It calls the
OpenAI Responses API, runs tools, and maintains conversation memory with automatic
summarization.

## Run

Requires Python 3.11+, uv, and `OPENAI_API_KEY` set in your environment.

```bash
uv sync
uv run python script.py
```

Leave the conversation ID blank to start a new conversation, or enter an existing
ID to resume it. Type `exit` to stop. Configure the model, client ID, tools, and
`COMPACTION_TOKEN_LIMIT` in `script.py`. The default system prompt is in
`src/resources/prompts.py`.

Available tools: web search, webpage reading, local path inspection, and text/PDF
reading. Local file tools accept absolute paths or paths relative to the working
directory; PDF extraction does not include OCR.

## Memory

`Conversation` holds the client/conversation IDs, timestamps, current messages,
structured history, and summaries. `ConversationRepository` saves events as JSONL,
summaries as JSON, and file references in `index.json`.

Before each model step, the agent checks its estimated input token count. At the
configured limit, it combines the active summary with older completed turns into
one replacement summary. The current turn remains intact, and original events
and previous summary files stay on disk. Use a very high limit to compare against
full history without compaction.

- Chat memory: `src/data/memory/`
- Eval memory: `evals/data/memory/`
- Execution traces: `traces/runs_eval.jsonl`

Memory currently stays within one conversation; shared memory across conversations
is not implemented.

## Evals

```bash
uv run python -m evals.run --config evals/evals.toml
```

`evals/evals.toml` selects the runs and can override their settings, including
`limit`. `evals/memory_design/memory_eval_config.toml` defines the models,
compaction limit, datasets, tools, and metrics for each memory case.

The two memory suites each contain 100 sequential prompts: one without tools and
one using local file tools with fixed fixtures. Eval mode omits the default
application system prompt. Results include an LLM grade, token precision/recall/F1,
and ROUGE-L F1. Each run writes uniquely named answer JSONL and summary JSON files.
The summary's `summaries_created` currently counts active summaries; count
`memory_compaction` trace events for total compaction calls.

Preview commands without running models:

```bash
uv run python -m evals.run --config evals/evals.toml --dry-run
```

Score existing answers without model calls:

```bash
uv run python -m evals.metrics.run --input path/to/answers.jsonl --output path/to/scores.jsonl
```

Each input row must contain `response` and `expected_answer`.

## Offline tests

```bash
uv run python -m unittest src.resources.conversation.test_conversation src.resources.conversation.test_repository src.resources.conversation.test_summary src.resources.conversation.test_compaction evals.metrics.test_metrics
```
