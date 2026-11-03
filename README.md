# MyAgent

A Python agent built from scratch, without an agent framework. It calls the
OpenAI Responses API, runs tools, and maintains conversation memory with automatic
summarization.

## Run

Requires Python 3.11+, uv, and `OPENAI_API_KEY` set in your environment.

```bash
uv sync
uv run python main.py
```

Leave the conversation ID blank to start a new conversation, or enter an existing
ID to resume it. Type `exit` to stop. Configure the model, client ID, tools, and
`COMPACTION_TOKEN_LIMIT` in `main.py`. The default system prompt is in
`src/resources/prompts.py`.

Available tools: web search, webpage reading, local path inspection, and text/PDF
reading. Local file tools accept absolute paths or paths relative to the working
directory; PDF extraction does not include OCR.

## Memory

`Conversation` and `ConversationItem` are dataclasses; functions handle adding
events, building model input, saving pending events, and trimming active context.
`ConversationRepository` stores events in JSONL files and file references in
`index.json`. Replay follows file append order, without turn or sequence counters.

System prompts, user messages, model responses, tool calls/results, run results,
and compactions are events. New events enter both `conversation.events` and a
pending-events buffer. Pending events are saved together at the end of a run.
Model input uses the latest system-prompt event, the active compaction summary,
and subsequent message/tool events.

After a run, compaction checks `usage.total_tokens` from the last model response
against `COMPACTION_TOKEN_LIMIT`. This is the completed request's input and output
usage; counts from different requests are not added together. Character-based
token estimates are used only for storage-file sizing.

When the limit is reached, the summarizer receives all active events, including
the previous summary and the latest messages. The resulting summary is appended
and saved as a `compaction` event. Active context is then trimmed to that new
summary alone. Original events and earlier compactions remain on disk.

When resuming a conversation, the repository loads its events and trims everything
before the latest compaction, keeping that summary and all later events. With no
compaction, it keeps the full history. Use a very high token limit to compare
against full history without compaction.

- Chat memory: `src/data/memory/`
- Eval memory: `evals/data/memory/`
- Execution traces: `traces/runs_eval.jsonl`

Memory currently stays within one conversation; shared memory across conversations
is not implemented.

To start fresh without deleting history, leave the conversation ID blank. To erase
all saved chat memory, stop the program and delete the entire `src/data/memory/`
folder, including `index.json` and the client folders. Restart with a blank
conversation ID; storage is recreated automatically. Evaluation memory is
separate in `evals/data/memory/`. Execution traces do not feed conversation memory.

## Run evals

After `uv sync`, run both memory suites from the project root:

```bash
.venv/bin/python -m evals.run --config evals/evals.toml
```

This runs 150 sequential prompts without tools, then 150 with local file tools
and fixed fixtures, using a 4,000-token compaction threshold. The original 100
prompts are followed by 50 additional corrections, history checks, list changes,
and delayed recall questions to test repeated compaction. The tools suite also
rereads old notes to check that later user corrections remain authoritative.
Eval mode omits the default application system prompt.
Results include an LLM grade, token precision/recall/F1, and ROUGE-L F1.

Choose suites and turn limits in `evals/evals.toml`. Keep only the `[[runs]]`
entries you want to execute. For example, to run 30 turns without tools:

```toml
[[runs]]
name = "memory_without_tools"
config = "memory_design/memory_eval_config.toml"
case = "tools_disabled"
limit = 30
compaction_token_limit = 4_000
```

Use `case = "tools_enabled"` for the local-tool suite and `limit = 150` for a full
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
The summary's `summaries_created` counts saved compaction events for that
conversation. Successful compactions also appear as `memory_compaction` events
in the execution trace.

To calculate text metrics on saved answers without model calls:

```bash
.venv/bin/python -m evals.metrics.run --input path/to/answers.jsonl --output path/to/scores.jsonl
```

Replace the paths with your saved result file and desired output path. Each input
row must contain `response` and `expected_answer`; prompt-only datasets cannot be
scored by this standalone runner. It writes per-answer metrics and a run summary.

## Offline tests

The checked-in offline suite covers text metrics, eval dispatch, and the memory
eval runner with mocked model calls:

```bash
uv run python -m unittest evals.metrics.test_metrics
```
