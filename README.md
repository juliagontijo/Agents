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

## Run evals

After `uv sync`, run both memory suites from the project root:

```bash
.venv/bin/python -m evals.run --config evals/evals.toml
```

This runs 100 sequential prompts without tools, then 100 with local file tools
and fixed fixtures. Eval mode omits the default application system prompt.
Results include an LLM grade, token precision/recall/F1, and ROUGE-L F1.

Choose suites and turn limits in `evals/evals.toml`. Keep only the `[[runs]]`
entries you want to execute. For example, to run 30 turns without tools:

```toml
[[runs]]
name = "memory_without_tools"
config = "memory_design/memory_eval_config.toml"
case = "tools_disabled"
limit = 30
compaction_token_limit = 8_000
```

Use `case = "tools_enabled"` for the local-tool suite and `limit = 100` for a full
run. Models, tools, metrics, and default compaction settings are defined in
`evals/memory_design/memory_eval_config.toml`; values in `evals/evals.toml`
override those defaults. Set a very high `compaction_token_limit`, such as
`1_000_000_000`, to compare against full history without compaction.

Preview the selected commands without making model calls:

```bash
.venv/bin/python -m evals.run --config evals/evals.toml --dry-run
```

Results are saved in `evals/memory_design/` as
`tools_disabled_result_<eval_id>.jsonl` or `tools_enabled_result_<eval_id>.jsonl`,
with a matching `.summary.json` file. Each run gets a unique ID.
The summary's `summaries_created` currently counts active summaries; count
`memory_compaction` trace events for total compaction calls.

To calculate text metrics on saved answers without model calls:

```bash
.venv/bin/python -m evals.metrics.run --input path/to/answers.jsonl --output path/to/scores.jsonl
```

Replace the paths with your saved result file and desired output path. Each input
row must contain `response` and `expected_answer`; prompt-only datasets cannot be
scored by this standalone runner. It writes per-answer metrics and a run summary.

## Offline tests

```bash
uv run python -m unittest src.resources.conversation.test_conversation src.resources.conversation.test_repository src.resources.conversation.test_summary src.resources.conversation.test_compaction evals.metrics.test_metrics
```
