from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.resources.conversation.conversation import Summary
from src.resources.conversation.repository import ConversationRepository


class ConversationRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "memory"
        self.repository = ConversationRepository(self.root)

    def add_turn(self, conversation, text="hello"):
        turn_id = f"turn-{conversation.last_event_seq + 1}"
        conversation.add_event("user_message", {"role": "user", "content": text}, turn_id)
        conversation.add_event("model_response", {"role": "assistant", "content": "answer"}, turn_id, 1)
        conversation.add_event("run_result", {"status": "completed"}, turn_id)
        conversation.finish_turn(self.repository)

    def lookup(self, conversation):
        return json.loads(self.repository.index_path.read_text())["lookup"][conversation.client_id][conversation.conversation_id]

    def summary(self, conversation, start=1, end=3):
        return Summary(
            client_id=conversation.client_id, conversation_id=conversation.conversation_id,
            text="A saved preference.", from_event_seq=start, through_event_seq=end,
        )

    def test_new_conversation_persists_lookup_without_creating_structured_file(self):
        conversation = self.repository.open("client")
        self.assertTrue(conversation.conversation_id)
        record = self.lookup(conversation)
        self.assertEqual(record["created_at"], conversation.created_at)
        self.assertEqual(record["updated_at"], conversation.updated_at)
        self.assertEqual(record["structured_files"], [])
        self.assertEqual(self.repository.open("client", conversation.conversation_id), conversation)
        self.assertNotEqual(self.repository.open("client").conversation_id, conversation.conversation_id)

    def test_supplied_id_must_exist_for_the_client(self):
        with self.assertRaises(ValueError):
            self.repository.open("client", "missing")
        conversation = self.repository.open("client")
        with self.assertRaises(ValueError):
            self.repository.open("another-client", conversation.conversation_id)

    def test_save_reopen_preserves_all_events_and_continues_sequence(self):
        conversation = self.repository.open("client")
        self.add_turn(conversation, "first")
        self.add_turn(conversation, "second")
        reopened = ConversationRepository(self.root).open("client", conversation.conversation_id)
        self.assertEqual(reopened.context, conversation.context)
        self.assertEqual(reopened._memory_events, conversation._memory_events)
        self.assertEqual(reopened.last_event_seq, 6)
        self.assertEqual([event.event_seq for event in self.repository.load_events("client", conversation.conversation_id, 2, 4)], [2, 3, 4])
        self.add_turn(reopened, "third")
        self.assertEqual(reopened.last_event_seq, 9)
        record = self.lookup(conversation)
        self.assertEqual(record["last_event_seq"], 9)
        metadata = json.loads(self.repository.index_path.read_text())["file_metadata"]
        for path in record["structured_files"]:
            self.assertEqual(metadata[path]["size_bytes"], (self.root / path).stat().st_size)

    def test_byte_rotation_keeps_each_oversized_event_whole(self):
        self.repository = ConversationRepository(self.root, max_file_bytes=1)
        conversation = self.repository.open("client")
        self.add_turn(conversation, "a" * 500)
        paths = self.lookup(conversation)["structured_files"]
        self.assertEqual(len(paths), 3)
        self.assertTrue(all(len((self.root / path).read_text().splitlines()) == 1 for path in paths))
        self.assertEqual(self.repository.open("client", conversation.conversation_id).context, conversation.context)

    def test_token_limit_rotates_before_byte_limit(self):
        self.repository = ConversationRepository(self.root, max_file_bytes=1_000_000, max_file_tokens=1)
        conversation = self.repository.open("client")
        self.add_turn(conversation)
        self.assertEqual(len(self.lookup(conversation)["structured_files"]), 3)

    def test_index_write_failure_can_retry_without_duplicate_events(self):
        conversation = self.repository.open("client")
        conversation.add_event("user_message", {"role": "user", "content": "retry me"}, "turn")
        pending = list(conversation._pending_events)
        with patch.object(self.repository, "_write_index", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                conversation.finish_turn(self.repository)
        self.assertEqual(conversation._pending_events, pending)
        conversation.finish_turn(self.repository)
        self.assertEqual(self.repository.load_events("client", conversation.conversation_id), pending)
        self.assertEqual(conversation._pending_events, [])
        self.assertEqual(len(self.lookup(conversation)["structured_files"]), 1)
        self.repository.save_events(pending)
        self.assertEqual(self.repository.load_events("client", conversation.conversation_id), pending)

    def test_index_failure_on_existing_segment_recovers_appended_events(self):
        conversation = self.repository.open("client")
        self.add_turn(conversation)
        conversation.add_event("user_message", {"role": "user", "content": "later"}, "next")
        with patch.object(self.repository, "_write_index", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                conversation.finish_turn(self.repository)
        conversation.finish_turn(self.repository)
        reopened = self.repository.open("client", conversation.conversation_id)
        self.assertEqual(reopened.last_event_seq, 4)
        self.assertEqual(len(reopened._memory_events), 4)
        self.assertEqual(len(self.lookup(conversation)["structured_files"]), 1)

    def test_conflicting_or_nonconsecutive_sequences_are_rejected(self):
        conversation = self.repository.open("client")
        self.add_turn(conversation)
        saved = self.repository.load_events("client", conversation.conversation_id)
        with self.assertRaises(ValueError):
            self.repository.save_events([replace(saved[0], payload={"content": "conflict"})])
        with self.assertRaises(ValueError):
            self.repository.save_events([replace(saved[-1], event_seq=5)])
        self.assertEqual(self.repository.load_events("client", conversation.conversation_id), saved)

    def test_inactive_summary_is_saved_without_excluding_raw_events(self):
        conversation = self.repository.open("client")
        self.add_turn(conversation)
        original = conversation.context["memory"]["structured"][:]
        summary = self.summary(conversation)
        self.repository.save_summary(conversation, summary)
        self.assertEqual(conversation.context["memory"]["structured"], original)
        self.assertEqual(self.lookup(conversation)["active_summaries"], [])
        summary_files = list((self.root / "client" / "summaries").glob("*.json"))
        self.assertEqual(len(summary_files), 1)
        self.assertEqual(json.loads(summary_files[0].read_text()), asdict(summary))
        self.assertEqual(self.repository.open("client", conversation.conversation_id).context, conversation.context)

    def test_active_summaries_preserve_gaps_and_raw_history(self):
        conversation = self.repository.open("client")
        for text in ("first", "middle", "last"):
            self.add_turn(conversation, text)
        first = self.summary(conversation)
        last = self.summary(conversation, 7, 9)
        self.repository.save_summary(conversation, last, activate=True)
        self.repository.save_summary(conversation, first, activate=True)
        reopened = self.repository.open("client", conversation.conversation_id)
        self.assertEqual(reopened.context, conversation.context)
        self.assertEqual([event.event_seq for event in reopened._memory_events], [4, 5, 6])
        self.assertEqual(reopened.context["memory"]["summaries"], [first, last])
        self.assertEqual(len(self.repository.load_events("client", conversation.conversation_id)), 9)
        self.assertEqual(reopened.last_event_seq, 9)
        with self.assertRaises(ValueError):
            self.repository.save_summary(reopened, self.summary(reopened, 1, 6), activate=True)

    def test_merge_summaries_replaces_only_active_references_and_keeps_files(self):
        conversation = self.repository.open("client")
        for _ in range(3):
            self.add_turn(conversation)
        summaries = [self.summary(conversation, start, start + 2) for start in (1, 4, 7)]
        for summary in summaries:
            self.repository.save_summary(conversation, summary, activate=True)
        old_files = {path: path.read_bytes() for path in self.root.rglob("*.json*") if path != self.repository.index_path}
        merged = self.summary(conversation, 1, 6)
        replaced_ids = [item.summary_id for item in summaries[:2]]
        self.repository.save_summary(conversation, merged, activate=True, replace_summary_ids=replaced_ids)
        self.assertEqual(conversation.context["memory"]["summaries"], [merged, summaries[2]])
        self.assertEqual(conversation._memory_events, [])
        self.assertEqual([item["summary_id"] for item in self.lookup(conversation)["active_summaries"]], [merged.summary_id, summaries[2].summary_id])
        self.assertEqual(self.repository.open("client", conversation.conversation_id).context, conversation.context)
        self.assertTrue(all(path.read_bytes() == data for path, data in old_files.items()))
        self.assertEqual(len(self.repository.load_events("client", conversation.conversation_id)), 9)
        # Retrying an already activated replacement remains harmless.
        self.repository.save_summary(conversation, merged, activate=True, replace_summary_ids=replaced_ids)
        self.assertEqual(len(self.lookup(conversation)["active_summaries"]), 2)

    def test_recompact_one_summary_preserves_pending_current_turn(self):
        conversation = self.repository.open("client")
        self.add_turn(conversation)
        first = self.summary(conversation)
        self.repository.save_summary(conversation, first, activate=True)
        conversation.add_event("user_message", {"role": "user", "content": "Current request"}, "current")
        pending = deepcopy(conversation._pending_events)
        current = deepcopy(conversation.context["current_conversation"])
        compacted = self.summary(conversation)
        self.repository.save_summary(conversation, compacted, activate=True, replace_summary_ids=[first.summary_id])
        self.assertEqual(conversation.context["memory"]["summaries"], [compacted])
        self.assertEqual(conversation._pending_events, pending)
        self.assertEqual(conversation.context["current_conversation"], current)
        self.assertEqual(conversation.last_event_seq, 4)

    def test_invalid_summary_replacements_leave_state_and_index_intact(self):
        conversation = self.repository.open("client")
        for _ in range(3):
            self.add_turn(conversation)
        first, last = self.summary(conversation), self.summary(conversation, 7, 9)
        for summary in (first, last):
            self.repository.save_summary(conversation, summary, activate=True)
        cases = [
            (self.summary(conversation, 1, 6), ["missing"], True),
            (self.summary(conversation, 1, 2), [first.summary_id], True),
            (self.summary(conversation, 1, 9), [first.summary_id], True),
            (self.summary(conversation, 1, 6), [first.summary_id], False),
            (replace(first, through_event_seq=6, revision=2), [first.summary_id], True),
            (self.summary(conversation, 1, 6), [first.summary_id, last.summary_id], True),
        ]
        for summary, replacements, activate in cases:
            before = deepcopy(conversation.__dict__)
            index = self.repository.index_path.read_bytes()
            with self.subTest(replacements=replacements, activate=activate), self.assertRaises(ValueError):
                self.repository.save_summary(conversation, summary, activate=activate, replace_summary_ids=replacements)
            self.assertEqual(conversation.__dict__, before)
            self.assertEqual(self.repository.index_path.read_bytes(), index)

    def test_replacement_cannot_cover_pending_events_or_partial_raw_turn(self):
        conversation = self.repository.open("client")
        self.add_turn(conversation)
        first = self.summary(conversation)
        self.repository.save_summary(conversation, first, activate=True)
        self.add_turn(conversation)
        conversation.add_event("user_message", {"role": "user", "content": "Pending"}, "current")
        for end in (5, 7):
            with self.subTest(end=end), self.assertRaises(ValueError):
                self.repository.save_summary(conversation, self.summary(conversation, 1, end), activate=True, replace_summary_ids=[first.summary_id])
        self.assertEqual(conversation.context["memory"]["summaries"], [first])
        self.assertEqual([item.event_seq for item in conversation._memory_events], [4, 5, 6])

    def test_replacement_index_failure_keeps_context_and_can_retry(self):
        conversation = self.repository.open("client")
        self.add_turn(conversation)
        first = self.summary(conversation)
        self.repository.save_summary(conversation, first, activate=True)
        self.add_turn(conversation)
        merged = self.summary(conversation, 1, 6)
        before = deepcopy(conversation.__dict__)
        with patch.object(self.repository, "_write_index", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                self.repository.save_summary(conversation, merged, activate=True, replace_summary_ids=[first.summary_id])
        self.assertEqual(conversation.__dict__, before)
        self.assertEqual(self.repository.open("client", conversation.conversation_id).context, conversation.context)
        self.repository.save_summary(conversation, merged, activate=True, replace_summary_ids=[first.summary_id])
        self.assertEqual(conversation.context["memory"]["summaries"], [merged])
        self.assertEqual(self.repository.open("client", conversation.conversation_id).context, conversation.context)

    def test_summary_revision_replaces_reference_and_preserves_old_file(self):
        conversation = self.repository.open("client")
        self.add_turn(conversation)
        first = self.summary(conversation)
        self.repository.save_summary(conversation, first, activate=True)
        revised = replace(first, revision=2, text="Updated preference.", updated_at=first.updated_at + 1)
        self.repository.save_summary(conversation, revised, activate=True)
        reopened = self.repository.open("client", conversation.conversation_id)
        self.assertEqual(reopened.context["memory"]["summaries"], [revised])
        self.assertEqual(len(self.lookup(conversation)["active_summaries"]), 1)
        self.assertEqual(self.lookup(conversation)["active_summaries"][0]["revision"], 2)
        self.assertEqual(len(list((self.root / "client" / "summaries").glob("*.json"))), 2)
        self.assertEqual(len(self.repository.load_events("client", conversation.conversation_id)), 3)
        with self.assertRaises(ValueError):
            self.repository.save_summary(conversation, replace(revised, text="Stale same revision"), activate=True)


if __name__ == "__main__":
    unittest.main()
