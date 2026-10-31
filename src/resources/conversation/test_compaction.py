from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.myagent.agent import run_agent
from src.resources.conversation.compaction import compact_conversation, compact_memory
from src.resources.conversation.conversation import Summary
from src.resources.conversation.repository import ConversationRepository


def response(text=None, calls=None):
    message = SimpleNamespace(content=text, tool_calls=calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)]), {}


class CompactionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.repository = ConversationRepository(self.root / "memory")
        self.conversation = self.repository.open("client")

    def add_turn(self, text="Remember the shelf", answer="Shelf B"):
        turn_id = f"turn-{self.conversation.last_event_seq + 1}"
        self.conversation.add_event("user_message", {"role": "user", "content": text}, turn_id)
        self.conversation.add_event("model_response", {"role": "assistant", "content": answer}, turn_id, 1)
        self.conversation.add_event("run_result", {"termination": "answer_returned"}, turn_id)
        self.conversation.finish_turn(self.repository)

    def compact(self, token_limit=1, **kwargs):
        return compact_conversation(self.conversation, self.repository, "mock-model", token_limit=token_limit, **kwargs)

    def raw_files(self):
        return {path: path.read_bytes() for path in (self.root / "memory").glob("*/structured/*.jsonl")}

    def test_compact_memory_accepts_each_input_form_without_mutating_it(self):
        self.add_turn("Read /notes/project.md", "Budget is USD 780")
        event = self.conversation._memory_events[0]
        summary = Summary("client", self.conversation.conversation_id, "Budget was USD 900.", 1, 3)
        message = {"role": "user", "content": "Correct the budget to USD 780."}
        cases = [
            (message, ["message"]),
            ([message, {"role": "assistant", "content": "Updated."}], ["message", "message"]),
            (event, ["event"]),
            (asdict(event), ["event"]),
            (summary, ["summary"]),
            (asdict(summary), ["summary"]),
            (summary.text, ["summary"]),
            ([summary, summary], ["summary", "summary"]),
            ([summary, [message, event]], ["summary", "message", "event"]),
        ]
        for inputs, kinds in cases:
            before = deepcopy(inputs)
            with self.subTest(kinds=kinds), patch("src.resources.conversation.compaction.call_model", return_value=response("  Consolidated memory.  ")) as model:
                self.assertEqual(compact_memory(inputs, "summary-model"), "Consolidated memory.")
                self.assertEqual(inputs, before)
                self.assertEqual(model.call_args.args[1:3], ([], "summary-model"))
                request = model.call_args.args[0]
                self.assertEqual([item["role"] for item in request], ["system", "user"])
                sources = json.loads(request[1]["content"])
                self.assertEqual([item["kind"] for item in sources], kinds)
                for source in sources:
                    if source["kind"] == "event":
                        self.assertEqual(source["event_seq"], event.event_seq)
                        self.assertEqual(source["recorded_at"], event.recorded_at)
                    elif source["kind"] == "summary" and "summary_id" in source:
                        self.assertEqual(source["summary_id"], summary.summary_id)
                        self.assertEqual(source["updated_at"], summary.updated_at)

    def test_mixed_compaction_can_replace_summary_and_raw_turns(self):
        self.add_turn("The budget is USD 900.")
        first = Summary("client", self.conversation.conversation_id, "Budget: USD 900.", 1, 3)
        self.repository.save_summary(self.conversation, first, activate=True)
        self.add_turn("Make it USD 780; /notes/project.md still says USD 900.")
        original_files = self.raw_files()
        sources = [first, self.conversation._memory_events]
        with patch("src.resources.conversation.compaction.call_model", return_value=response("Current: USD 780. Previously USD 900; /notes/project.md is stale.")) as model:
            text = compact_memory(sources, "summary-model")
        sent = json.loads(model.call_args.args[0][1]["content"])
        self.assertEqual([item["kind"] for item in sent], ["summary", "event", "event", "event"])
        self.assertEqual(sent[0]["from_event_seq"], 1)
        self.assertEqual(sent[1]["event_seq"], 4)
        merged = Summary("client", self.conversation.conversation_id, text, 1, 6)
        self.repository.save_summary(self.conversation, merged, activate=True, replace_summary_ids=[first.summary_id])
        self.assertEqual(self.conversation.context["memory"], {"structured": [], "summaries": [merged]})
        self.assertEqual(self.raw_files(), original_files)
        reopened = self.repository.open("client", self.conversation.conversation_id)
        self.assertEqual(reopened.context, self.conversation.context)

    def test_invalid_compaction_inputs_do_not_call_model(self):
        for inputs in ([], [[], []], None, 42, ["summary", object()]):
            with self.subTest(inputs=inputs), patch("src.resources.conversation.compaction.call_model") as model:
                with self.assertRaises((ValueError, TypeError)):
                    compact_memory(inputs, "summary-model")
                model.assert_not_called()

    def test_high_limit_keeps_raw_context_without_summarizer_call(self):
        self.add_turn("A long earlier message " * 300)
        original = deepcopy(self.conversation.context)
        with patch("src.resources.conversation.compaction.call_model") as model:
            self.compact(token_limit=10**9)
        model.assert_not_called()
        self.assertEqual(self.conversation.context, original)
        self.assertEqual(list((self.root / "memory").glob("*/summaries/*.json")), [])

    def test_summary_persists_applies_and_reopens_without_erasing_raw_files(self):
        self.add_turn("The project notes are in /notes/project.md. " * 300)
        original_files = self.raw_files()
        with patch("src.resources.conversation.compaction.call_model", return_value=response("Notes: /notes/project.md.")) as model:
            self.compact(token_limit=500)
        model.assert_called_once()
        self.assertEqual(model.call_args.args[1:3], ([], "mock-model"))
        messages = model.call_args.args[0]
        self.assertEqual([item["role"] for item in messages], ["system", "user"])
        historical = json.loads(messages[1]["content"])
        self.assertEqual(historical[0]["kind"], "event")
        self.assertEqual(historical[0]["event_seq"], 1)
        self.assertEqual(historical[0]["payload"]["role"], "user")
        self.assertIn("/notes/project.md", historical[0]["payload"]["content"])
        summary = self.conversation.context["memory"]["summaries"][0]
        self.assertEqual((summary.from_event_seq, summary.through_event_seq), (1, 3))
        self.assertEqual(self.conversation.context["memory"]["structured"], [])
        self.assertEqual(self.conversation._memory_events, [])
        self.assertEqual(self.raw_files(), original_files)
        reopened = self.repository.open("client", self.conversation.conversation_id)
        self.assertEqual(reopened.context, self.conversation.context)
        self.assertEqual(reopened.last_event_seq, 3)

    def test_existing_summaries_count_toward_the_trigger(self):
        self.add_turn()
        existing = Summary("client", self.conversation.conversation_id, "Earlier knowledge " * 500, 1, 3)
        self.repository.save_summary(self.conversation, existing, activate=True)
        self.add_turn("A short additional turn")
        with patch("src.resources.conversation.compaction.call_model", return_value=response("Additional information.")) as model:
            self.compact(token_limit=500)
        model.assert_called_once()
        summaries = self.conversation.context["memory"]["summaries"]
        self.assertEqual(len(summaries), 1)
        self.assertNotIn(existing, summaries)
        self.assertEqual((summaries[0].from_event_seq, summaries[0].through_event_seq), (1, 6))
        sources = json.loads(model.call_args.args[0][1]["content"])
        self.assertEqual([item["kind"] for item in sources], ["summary", "event", "event", "event"])
        self.assertEqual(sources[0]["text"], existing.text)
        self.assertEqual(sources[1]["event_seq"], 4)

    def test_system_prompt_and_tools_count_toward_the_trigger(self):
        for extra in ({"system_prompt": "Instructions " * 500}, {"tools": [{"type": "function", "name": "read_file", "description": "Read a resource " * 500}]}):
            with self.subTest(extra=next(iter(extra))):
                self.add_turn("A short saved turn")
                with patch("src.resources.conversation.compaction.call_model", return_value=response("Saved information.")) as model:
                    self.compact(token_limit=500, **extra)
                model.assert_called_once()

    def test_compaction_carries_one_summary_across_raw_event_gaps(self):
        for label in ("first", "already summarized", "third"):
            self.add_turn(label * 200)
        existing = Summary("client", self.conversation.conversation_id, "Existing middle summary.", 4, 6)
        self.repository.save_summary(self.conversation, existing, activate=True)
        original_files = self.raw_files()
        with patch("src.resources.conversation.compaction.call_model", return_value=response("New range summary.")) as model:
            self.compact()
        self.assertEqual(model.call_count, 2)
        summaries = self.conversation.context["memory"]["summaries"]
        self.assertEqual([(item.from_event_seq, item.through_event_seq) for item in summaries], [(1, 9)])
        self.assertEqual(self.raw_files(), original_files)
        first, second = [json.loads(call.args[0][1]["content"]) for call in model.call_args_list]
        self.assertEqual(sum(item["kind"] == "summary" for item in first), 1)
        self.assertEqual(sum(item["kind"] == "summary" for item in second), 1)
        self.assertEqual(first[0]["summary_id"], existing.summary_id)
        self.assertEqual((second[0]["from_event_seq"], second[0]["through_event_seq"]), (1, 6))
        self.assertEqual(second[1]["event_seq"], 7)
        self.assertEqual(self.repository.open("client", self.conversation.conversation_id).context, self.conversation.context)

    def test_pending_only_context_is_not_summarized(self):
        self.conversation.add_event("user_message", {"role": "user", "content": "Current input " * 500}, "live-turn")
        pending = deepcopy(self.conversation._pending_events)
        current = deepcopy(self.conversation.context["current_conversation"])
        with patch("src.resources.conversation.compaction.call_model") as model:
            self.compact()
        model.assert_not_called()
        self.assertEqual(self.conversation._pending_events, pending)
        self.assertEqual(self.conversation.context["current_conversation"], current)
        self.assertEqual(self.raw_files(), {})

    def test_compaction_preserves_current_tool_call_and_result(self):
        self.add_turn("Old information " * 300)
        for event_type, payload in (
            ("user_message", {"role": "user", "content": "Read the new notes"}),
            ("tool_call", {"type": "function_call", "call_id": "live", "name": "read_file", "arguments": "{}"}),
            ("tool_result", {"type": "function_call_output", "call_id": "live", "output": "New notes"}),
        ):
            self.conversation.add_event(event_type, payload, "live-turn", 1)
        pending = deepcopy(self.conversation._pending_events)
        current = deepcopy(self.conversation.context["current_conversation"])
        with patch("src.resources.conversation.compaction.call_model", return_value=response("Old information.")) as model:
            self.compact()
        self.assertEqual(self.conversation._pending_events, pending)
        self.assertEqual(self.conversation.context["current_conversation"], current)
        self.assertNotIn("New notes", model.call_args.args[0][1]["content"])
        self.assertEqual(self.conversation.build_messages()[-3:], current)

    def test_model_failure_preserves_context_and_files(self):
        self.add_turn()
        original = deepcopy(self.conversation.context)
        original_events = deepcopy(self.conversation._memory_events)
        original_files = self.raw_files()
        with patch("src.resources.conversation.compaction.call_model", side_effect=RuntimeError("summarizer failed")):
            with self.assertRaisesRegex(RuntimeError, "summarizer failed"):
                self.compact()
        self.assertEqual(self.conversation.context, original)
        self.assertEqual(self.conversation._memory_events, original_events)
        self.assertEqual(self.raw_files(), original_files)
        self.assertEqual(list((self.root / "memory").glob("*/summaries/*.json")), [])

    def test_empty_summary_preserves_context_and_files(self):
        self.add_turn()
        original = deepcopy(self.conversation.context)
        original_files = self.raw_files()
        for text in (None, "   "):
            with self.subTest(text=text), patch("src.resources.conversation.compaction.call_model", return_value=response(text)):
                with self.assertRaises(ValueError):
                    self.compact()
            self.assertEqual(self.conversation.context, original)
            self.assertEqual(self.raw_files(), original_files)
            self.assertEqual(list((self.root / "memory").glob("*/summaries/*.json")), [])

    def test_agent_repeated_compactions_keep_one_cumulative_summary(self):
        self.add_turn("Original information " * 300)
        with patch("src.myagent.agent.call_model", return_value=response("Done")), patch("src.resources.conversation.compaction.call_model", side_effect=[response(f"Memory {index}") for index in range(3)]) as summarizer, patch("src.myagent.agent.TRACE_PATH", self.root / "trace.jsonl"):
            previous = None
            for index in range(3):
                _, self.conversation, _ = run_agent(
                    "New request " * 200, "mock-model", [], self.conversation, self.repository,
                    eval_mode=True, compaction_token_limit=300,
                )
                summaries = self.conversation.context["memory"]["summaries"]
                self.assertEqual(len(summaries), 1)
                self.assertEqual((summaries[0].from_event_seq, summaries[0].through_event_seq), (1, 3 * (index + 1)))
                sources = json.loads(summarizer.call_args.args[0][1]["content"])
                inputs = [item for item in sources if item["kind"] == "summary"]
                self.assertEqual(len(inputs), int(previous is not None))
                if previous is not None:
                    self.assertEqual(inputs[0]["summary_id"], previous.summary_id)
                previous = summaries[0]
                reopened = self.repository.open("client", self.conversation.conversation_id)
                self.assertEqual(reopened.context, self.conversation.context)
                self.conversation = reopened
        self.assertEqual(summarizer.call_count, 3)
        self.assertEqual(len(self.repository.load_events("client", self.conversation.conversation_id)), 12)

    def test_agent_checks_compaction_before_each_model_step(self):
        call = SimpleNamespace(id="read", function=SimpleNamespace(name="read_file", arguments={"path": "notes.txt"}))
        order = []

        def check(*args, **kwargs):
            order.append("compact")

        replies = iter([response(calls=[call]), response("Done")])

        def model(*args, **kwargs):
            order.append("agent")
            return next(replies)

        with patch("src.myagent.agent.compact_conversation", side_effect=check), patch("src.myagent.agent.call_model", side_effect=model), patch("src.myagent.agent.execute_tool", return_value="notes"), patch("src.myagent.agent.TRACE_PATH", self.root / "trace.jsonl"):
            answer, _, _ = run_agent("Read notes", "mock-model", [], self.conversation, self.repository, eval_mode=True)
        self.assertEqual(answer, "Done")
        self.assertEqual(order, ["compact", "agent", "compact", "agent"])


    def test_eval_runs_compaction_and_records_settings(self):
        import contextlib
        import io
        import runpy
        import sys

        dataset = self.root / "questions.jsonl"
        dataset.write_text("".join(json.dumps(row) + "\n" for row in [
            {"id": 1, "prompt": "Remember /notes/project.md contains our blue colour preference. " * 100,
             "expected_answer": "blue"},
            {"id": 2, "prompt": "What colour did we choose?", "expected_answer": "blue"},
        ]))
        arguments = ["memory_eval", "--input", str(dataset), "--output", str(self.root / "result.jsonl"),
                     "--model", "agent-model", "--grader-model", "grader-model",
                     "--summary-model", "summary-model", "--compaction-token-limit", "300"]
        grader_response = response('{"score":2,"reason":"Correct"}')
        with patch.object(sys, "argv", arguments), patch("src.resources.conversation.repository.ConversationRepository", return_value=self.repository), patch("src.myagent.agent.call_model", return_value=response("blue")) as agent, patch("src.resources.conversation.compaction.call_model", return_value=response("Project notes: /notes/project.md; chosen colour: blue.")) as summarizer, patch("resources.system.system.call_model", return_value=grader_response), patch("src.myagent.agent.TRACE_PATH", self.root / "trace.jsonl"), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            runpy.run_module("evals.memory_design.eval_memory_script", run_name="__main__")
        summarizer.assert_called_once()
        self.assertEqual(summarizer.call_args.args[2], "summary-model")
        self.assertIn("chosen colour: blue", agent.call_args_list[1].args[0][0]["content"])
        result_path = next(self.root.glob("result_*.summary.json"))
        result = json.loads(result_path.read_text())
        self.assertEqual(result["compaction_token_limit"], 300)
        self.assertEqual(result["summary_model"], "summary-model")
        self.assertEqual(result["summaries_created"], 1)
        traces = [json.loads(line) for line in (self.root / "trace.jsonl").read_text().splitlines()]
        compacted = [event for trace in traces for event in trace["events"] if event["type"] == "memory_compaction"]
        self.assertEqual(len(compacted), 1)
        self.assertLess(compacted[0]["payload"]["estimated_tokens_after"], compacted[0]["payload"]["estimated_tokens_before"])


if __name__ == "__main__":
    unittest.main()
