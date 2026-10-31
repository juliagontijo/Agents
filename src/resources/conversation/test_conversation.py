import contextlib
import io
import json
from copy import deepcopy
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.myagent.agent import run_agent
from src.resources.conversation.conversation import Conversation, Summary
from src.resources.conversation.repository import ConversationRepository


def response(content=None, calls=None):
    message = SimpleNamespace(content=content, tool_calls=calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)]), {}


def tool_call(call_id, name="read_file", arguments=None):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments or {"path": "notes.txt"}),
    )


class ConversationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "memory"
        self.repository = ConversationRepository(self.path)

    def records(self):
        return [json.loads(line) for path in sorted(self.path.glob("*/structured/*.jsonl"))
                for line in path.read_text().splitlines() if line.strip()]

    def test_context_defaults_are_independent(self):
        first = Conversation(client_id="one")
        second = Conversation(client_id="two")
        first.context["memory"]["structured"].append({"role": "user", "content": "one"})
        first.context["memory"]["summaries"].append("summary")
        first.context["current_conversation"].append({"role": "user", "content": "current"})
        self.assertEqual(second.context["memory"], {"structured": [], "summaries": []})
        self.assertEqual(second.context["current_conversation"], [])
        self.assertEqual(second._pending_events, [])
        self.assertIsInstance(first.created_at, int)
        self.assertNotEqual(first.conversation_id, second.conversation_id)

    def test_events_share_api_payloads_and_keep_metadata_out_of_messages(self):
        conversation = Conversation(client_id="client")
        payloads = [
            ("user_message", {"role": "user", "content": "Read notes"}),
            ("model_response", {"role": "assistant", "content": "Looking"}),
            ("tool_call", {"type": "function_call", "call_id": "call", "name": "read_file", "arguments": "{}"}),
            ("tool_result", {"type": "function_call_output", "call_id": "call", "output": "notes"}),
        ]
        for event_type, payload in payloads:
            conversation.add_event(event_type, payload, "turn", step_id=1)
            self.assertIs(conversation.context["current_conversation"][-1], payload)
            self.assertIs(conversation._pending_events[-1].payload, payload)
        conversation.add_event("run_result", {"termination": "answer_returned"}, "turn", step_id=1)
        self.assertEqual(len(conversation.context["current_conversation"]), 4)
        self.assertEqual(len(conversation._pending_events), 5)
        event = conversation._pending_events[0]
        self.assertEqual((event.client_id, event.conversation_id, event.turn_id, event.step_id),
                         ("client", conversation.conversation_id, "turn", 1))
        self.assertIsInstance(event.recorded_at, int)

    def test_messages_include_summaries_once_without_changing_context(self):
        conversation = Conversation(client_id="client")
        previous = {"role": "assistant", "content": "Earlier answer"}
        current = {"role": "user", "content": "Next question"}
        conversation.context["memory"]["structured"].append(previous)
        conversation.context["memory"]["summaries"].extend([
            Summary("client", conversation.conversation_id, "# Plans\nUse Shelf B.", 1, 3),
            Summary("client", conversation.conversation_id, "# Preferences\nShort answers.", 4, 6),
        ])
        conversation.add_event("user_message", current, "turn")
        original = deepcopy(conversation.context)
        messages = conversation.build_messages("Base instructions")
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Base instructions", messages[0]["content"])
        self.assertIn("# Plans\nUse Shelf B.", messages[0]["content"])
        self.assertIn("# Preferences\nShort answers.", messages[0]["content"])
        self.assertEqual(messages[1:], [previous, current])
        self.assertIs(messages[1], previous)
        self.assertIs(messages[2], current)
        self.assertEqual(conversation.build_messages("Base instructions"), messages)
        self.assertEqual(conversation.context, original)
        summary_only = conversation.build_messages()
        self.assertEqual(summary_only[0]["role"], "system")
        self.assertEqual(summary_only[0]["content"].count("Use Shelf B."), 1)
        self.assertEqual(Conversation(client_id="blank").build_messages(), [])

    def test_finish_turn_saves_only_new_events_and_reopen_reconstructs_history(self):
        conversation = self.repository.open("client")
        first = {"role": "user", "content": "First question"}
        conversation.add_event("user_message", first, "first-turn")
        conversation.add_event("model_response", {"role": "assistant", "content": "First answer"}, "first-turn", 1)
        conversation.add_event("run_result", {"termination": "answer_returned"}, "first-turn", 1)
        conversation.finish_turn(self.repository)
        saved = self.records()
        self.assertEqual(len(self.records()), 3)
        self.assertEqual(len(conversation.context["memory"]["structured"]), 2)
        self.assertIs(conversation.context["memory"]["structured"][0], first)
        self.assertEqual(conversation.context["current_conversation"], [])
        self.assertEqual(conversation._pending_events, [])
        conversation.finish_turn(self.repository)
        self.assertEqual(self.records(), saved)
        conversation.add_event("user_message", {"role": "user", "content": "Second question"}, "second-turn")
        conversation.finish_turn(self.repository)
        self.assertEqual(len(self.records()), 4)
        restored = self.repository.open("client", conversation.conversation_id)
        self.assertEqual(restored.created_at, conversation.created_at)
        self.assertEqual(restored.context["memory"]["structured"], conversation.context["memory"]["structured"])
        self.assertEqual(restored.context["current_conversation"], [])
        self.assertEqual(restored._pending_events, [])

    def test_save_failure_retains_pending_events_and_current_messages(self):
        conversation = self.repository.open("client")
        message = {"role": "user", "content": "Keep me"}
        conversation.add_event("user_message", message, "turn")
        original_event = conversation._pending_events[0]
        failing_repository = Mock()
        failing_repository.save_events.side_effect = OSError("disk full")
        with self.assertRaisesRegex(OSError, "disk full"):
            conversation.finish_turn(failing_repository)
        self.assertIs(conversation._pending_events[0], original_event)
        self.assertIs(conversation.context["current_conversation"][0], message)
        self.assertEqual(conversation.context["memory"]["structured"], [])
        conversation.finish_turn(self.repository)
        self.assertEqual(len(self.records()), 1)
        self.assertEqual(conversation._pending_events, [])

    def test_open_scopes_events_by_both_ids(self):
        selected = self.repository.open("client")
        other = self.repository.open("client")
        for conversation, content in ((selected, "Chosen"), (other, "Other conversation")):
            conversation.add_event("user_message", {"role": "user", "content": content}, "turn")
            conversation.finish_turn(self.repository)
        restored = self.repository.open("client", selected.conversation_id)
        self.assertEqual(restored.context["memory"]["structured"], [{"role": "user", "content": "Chosen"}])
        with self.assertRaises(ValueError):
            self.repository.open("other", selected.conversation_id)


class AgentConversationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "memory"
        self.trace_path = self.root / "traces.jsonl"
        self.repository = ConversationRepository(self.path)
        self.conversation = self.repository.open("client")
        self.enterContext(patch("src.myagent.agent.TRACE_PATH", self.trace_path))

    def run_agent(self, prompt="Read the notes", **changes):
        return run_agent(prompt=prompt, model="mock-model", agent_tools=[],
                         conversation=self.conversation, repository=self.repository,
                         **changes)

    def records(self):
        return [json.loads(line) for path in sorted(self.path.glob("*/structured/*.jsonl"))
                for line in path.read_text().splitlines() if line.strip()]

    def test_tool_loop_and_next_turn_use_same_conversation_without_reading_memory(self):
        captured = []
        replies = iter([
            response(calls=[tool_call("first"), tool_call("second", "inspect_path")]),
            response("The keys are on Shelf B."),
            response("Shelf B."),
        ])

        def fake_model(messages, tools, model, trace):
            captured.append(deepcopy(messages))
            return next(replies)

        with patch("src.myagent.agent.call_model", side_effect=fake_model), patch("src.myagent.agent.execute_tool", return_value='{"shelf":"B"}') as execute:
            answer, returned, trace_id = self.run_agent(eval_mode=True)
            self.assertEqual(answer, "The keys are on Shelf B.")
            self.assertIs(returned, self.conversation)
            self.assertTrue(trace_id)
            self.assertEqual(execute.call_count, 2)
            self.assertEqual(captured[0], [{"role": "user", "content": "Read the notes"}])
            self.assertEqual([item.get("type") for item in captured[1][1:]],
                             ["function_call", "function_call_output", "function_call", "function_call_output"])
            original_open = Path.open

            def forbid_memory_reads(path, mode="r", *args, **kwargs):
                if self.path in Path(path).parents and Path(path).suffix == ".jsonl" and "r" in mode:
                    raise AssertionError("The next turn must use in-memory history")
                return original_open(path, mode, *args, **kwargs)

            with patch.object(Path, "open", forbid_memory_reads):
                answer, returned, second_trace = self.run_agent("Where were the keys?", eval_mode=True)
            self.assertEqual(answer, "Shelf B.")
            self.assertIs(returned, self.conversation)
            self.assertNotEqual(second_trace, trace_id)
        self.assertEqual(captured[2][-2:], [
            {"role": "assistant", "content": "The keys are on Shelf B."},
            {"role": "user", "content": "Where were the keys?"},
        ])
        records = self.records()
        users = [record for record in records if record["type"] == "user_message"]
        self.assertEqual(len(users), 2)
        self.assertIsNone(users[0]["step_id"])
        self.assertNotEqual(users[0]["turn_id"], users[1]["turn_id"])
        first_turn = [record for record in records if record["turn_id"] == users[0]["turn_id"]]
        self.assertEqual([record["step_id"] for record in first_turn if record["type"] == "tool_call"], [1, 1])
        self.assertEqual([record["step_id"] for record in first_turn if record["type"] == "model_response"], [2])
        self.assertEqual(self.conversation.context["current_conversation"], [])
        self.assertEqual(self.conversation._pending_events, [])
        self.assertEqual(len(self.conversation.context["memory"]["structured"]), 8)

    def test_normal_mode_adds_system_prompt_without_storing_it_as_history(self):
        with patch("src.myagent.agent.SYSTEM_PROMPT", "Base instructions"), patch("src.myagent.agent.call_model", return_value=response("Done")) as model:
            self.run_agent()
        self.assertEqual(model.call_args.args[0][0], {"role": "system", "content": "Base instructions"})
        self.assertFalse(any(row["type"] == "system_message" for row in self.records()))

    def test_model_failure_saves_observed_user_message_and_completion(self):
        with patch("src.myagent.agent.call_model", side_effect=RuntimeError("model unavailable")):
            with self.assertRaisesRegex(RuntimeError, "model unavailable"):
                self.run_agent(eval_mode=True)
        records = self.records()
        self.assertEqual([record["type"] for record in records], ["user_message", "run_result"])
        self.assertEqual(records[-1]["payload"]["termination"], "exception")
        self.assertEqual(records[-1]["payload"]["error"]["message"], "model unavailable")
        self.assertEqual(self.conversation.context["memory"]["structured"], [{"role": "user", "content": "Read the notes"}])
        self.assertEqual(self.conversation._pending_events, [])
        trace = json.loads(self.trace_path.read_text().splitlines()[0])
        self.assertEqual(trace["events"][-1]["payload"]["termination"], "exception")

    def test_tool_failure_closes_call_before_saving_and_raising(self):
        with patch("src.myagent.agent.call_model", return_value=response(calls=[tool_call("failed")])), patch("src.myagent.agent.execute_tool", side_effect=ValueError("invalid tool")):
            with self.assertRaisesRegex(ValueError, "invalid tool"):
                self.run_agent(eval_mode=True)
        records = self.records()
        self.assertEqual([record["type"] for record in records], ["user_message", "tool_call", "tool_result", "run_result"])
        result = records[2]
        self.assertEqual(result["payload"]["type"], "function_call_output")
        self.assertEqual(result["payload"]["call_id"], "failed")
        self.assertIn("invalid tool", result["payload"]["output"])
        self.assertEqual(result["step_id"], 1)
        self.assertEqual(records[1]["turn_id"], result["turn_id"])
        restored = self.repository.open("client", self.conversation.conversation_id)
        self.assertEqual(restored.context["memory"]["structured"], self.conversation.context["memory"]["structured"])
        with patch("src.myagent.agent.call_model", return_value=response("Recovered")) as model:
            answer, _, _ = self.run_agent("Continue", eval_mode=True)
        self.assertEqual(answer, "Recovered")
        self.assertEqual(model.call_args.args[0][-2]["type"], "function_call_output")

    def test_step_limit_saves_complete_tool_exchange_and_returns_same_object(self):
        with patch("src.myagent.agent.call_model", return_value=response(calls=[tool_call("limit")])), patch("src.myagent.agent.execute_tool", return_value="notes"):
            answer, returned, _ = self.run_agent(max_steps=1, eval_mode=True)
        self.assertEqual(answer, "Error: Max steps limit reached.")
        self.assertIs(returned, self.conversation)
        records = self.records()
        self.assertEqual(records[-1]["payload"]["termination"], "step_limit")
        self.assertEqual(records[-1]["payload"]["model_steps"], 1)
        self.assertEqual(records[-2]["payload"]["call_id"], "limit")
        self.assertEqual(self.conversation.context["current_conversation"], [])
        self.assertEqual(self.conversation._pending_events, [])


class CliConversationTests(unittest.TestCase):
    def test_cli_initializes_once_and_reuses_new_or_selected_conversation(self):
        for use_existing in (False, True):
            selected_id = ""
            with self.subTest(selected_id=selected_id), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                repository = ConversationRepository(root / "memory")
                if use_existing:
                    existing = repository.open("luca")
                    selected_id = existing.conversation_id
                    existing.add_event("user_message", {"role": "user", "content": "Old question"}, "old-turn")
                    existing.add_event("model_response", {"role": "assistant", "content": "Old answer"}, "old-turn", 1)
                    existing.finish_turn(repository)
                seen = []

                def record_call(**kwargs):
                    seen.append(kwargs["conversation"])
                    return run_agent(**kwargs)

                with patch("src.resources.conversation.repository.ConversationRepository", return_value=repository), patch.object(repository, "open", wraps=repository.open) as open_conversation, patch("src.myagent.agent.run_agent", side_effect=record_call), patch("src.myagent.agent.call_model", return_value=response("Answer")), patch("src.myagent.agent.TRACE_PATH", root / "traces.jsonl"), patch("builtins.input", side_effect=[selected_id, "First", "Second", "exit"]), contextlib.redirect_stdout(io.StringIO()):
                    runpy.run_path(str(Path(__file__).resolve().parents[3] / "script.py"), run_name="__main__")
                open_conversation.assert_called_once_with(client_id="luca", conversation_id=selected_id or None)
                self.assertIs(seen[0], seen[1])
                history = seen[0].context["memory"]["structured"]
                self.assertEqual(len(history), 6 if selected_id else 4)
                if selected_id:
                    self.assertEqual(seen[0].conversation_id, selected_id)
                    self.assertEqual(history[0]["content"], "Old question")


if __name__ == "__main__":
    unittest.main()
