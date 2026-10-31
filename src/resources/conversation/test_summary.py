from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import Mock, patch

from src.resources.conversation.conversation import Conversation, Summary


class SummaryConversationTests(unittest.TestCase):
    def setUp(self):
        self.conversation = Conversation(client_id="client", conversation_id="chat")
        self.saved = []
        self.repository = Mock()
        self.repository.save_events.side_effect = lambda events: self.saved.append(list(events))

    def turn(self, turn_id):
        c = self.conversation
        c.add_event("user_message", {"role": "user", "content": turn_id}, turn_id)
        c.add_event("model_response", {"role": "assistant", "content": "answer"}, turn_id, 1)
        c.add_event("run_result", {"termination": "answer_returned"}, turn_id, 1)
        c.finish_turn(self.repository)

    def summary(self, start=1, end=3, **changes):
        values = dict(client_id="client", conversation_id="chat", text="Keep the plans.",
                      from_event_seq=start, through_event_seq=end,
                      summary_id="summary", created_at=10, updated_at=10)
        values.update(changes)
        return Summary(**values)

    def reject(self, summary):
        before = deepcopy(self.conversation.__dict__)
        with self.assertRaises(ValueError):
            self.conversation._apply_summary(summary)
        self.assertEqual(self.conversation.__dict__, before)

    def test_sequence_payload_identity_and_turn_promotion(self):
        c = self.conversation
        c.updated_at = 0
        payload = {"role": "user", "content": "first"}
        with patch("src.resources.conversation.conversation.now", return_value=123):
            c.add_event("user_message", payload, "one")
        self.assertEqual(c.updated_at, 123)
        self.assertEqual(c.last_event_seq, 1)
        self.assertIs(c._pending_events[0].payload, payload)
        self.assertIs(c.context["current_conversation"][0], payload)
        c.add_event("run_result", {"termination": "exception"}, "one")
        c.finish_turn(self.repository)
        self.assertEqual([event.event_seq for event in c._memory_events], [1, 2])
        self.assertEqual(c.context["memory"]["structured"], [payload])
        self.assertEqual(c._pending_events, [])
        self.assertEqual(c.context["current_conversation"], [])
        c.add_event("user_message", {"role": "user", "content": "next"}, "two")
        self.assertEqual(c._pending_events[0].event_seq, 3)
        self.assertEqual([event.event_seq for event in self.saved[0]], [1, 2])

    def test_save_failure_keeps_sequence_ids_and_pending_messages(self):
        c = self.conversation
        c.add_event("user_message", {"role": "user", "content": "retain"}, "one")
        event = c._pending_events[0]
        before = deepcopy(c.__dict__)
        failure = Mock()
        failure.save_events.side_effect = OSError("disk full")
        with self.assertRaises(OSError):
            c.finish_turn(failure)
        self.assertEqual(c.__dict__, before)
        c.finish_turn(self.repository)
        self.assertIs(c._memory_events[0], event)
        self.assertEqual((event.event_seq, c.last_event_seq), (1, 1))
        self.assertEqual(self.saved[0][0].item_id, event.item_id)

    def test_summary_rendering_system_first_and_no_context_mutation(self):
        self.turn("one")
        c = self.conversation
        c._apply_summary(self.summary())
        current = {"role": "user", "content": "next"}
        c.add_event("user_message", current, "two")
        before = deepcopy(c.__dict__)
        messages = c.build_messages("Base instructions")
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Base instructions", messages[0]["content"])
        self.assertIn("Events 1-3", messages[0]["content"])
        self.assertEqual(messages[0]["content"].count("Keep the plans."), 1)
        self.assertIs(messages[1], current)
        self.assertEqual(c.build_messages("Base instructions"), messages)
        self.assertEqual(c.__dict__, before)

    def test_middle_range_preserves_gaps_and_sorts_summaries(self):
        for turn_id in ("one", "two", "three"):
            self.turn(turn_id)
        c = self.conversation
        c._apply_summary(self.summary(4, 6, summary_id="middle"))
        self.assertEqual([e.event_seq for e in c._memory_events], [1, 2, 3, 7, 8, 9])
        self.assertEqual(len(c.context["memory"]["structured"]), 4)
        self.assertTrue(all("role" in p for p in c.context["memory"]["structured"]))
        self.assertEqual(c.last_event_seq, 9)
        c._apply_summary(self.summary(1, 3, summary_id="first"))
        self.assertEqual([s.summary_id for s in c.context["memory"]["summaries"]],
                         ["first", "middle"])
        self.assertEqual([e.event_seq for e in c._memory_events], [7, 8, 9])

    def test_invalid_ownership_bounds_and_revision_do_not_mutate(self):
        self.turn("one")
        invalid = [dict(client_id="other"), dict(conversation_id="other"),
                   dict(from_event_seq=0), dict(from_event_seq=4),
                   dict(through_event_seq=99), dict(from_event_seq=1.5),
                   dict(through_event_seq="3"), dict(revision=0),
                   dict(text=""), dict(text=None)]
        for changes in invalid:
            with self.subTest(changes=changes):
                self.reject(self.summary(**changes))

    def test_overlapping_different_summary_is_rejected(self):
        self.turn("one")
        self.turn("two")
        self.conversation._apply_summary(self.summary())
        self.reject(self.summary(1, 6, summary_id="overlap"))

    def test_fixed_range_newer_revision_replaces_without_raw_events(self):
        self.turn("one")
        self.turn("two")
        c = self.conversation
        original = self.summary()
        c._apply_summary(original)
        remaining = list(c._memory_events)
        updated = replace(original, revision=2, text="Updated plans.", updated_at=20)
        c._apply_summary(updated)
        self.assertEqual(c.context["memory"]["summaries"], [updated])
        self.assertEqual(c._memory_events, remaining)
        self.assertEqual(c.last_event_seq, 6)

    def test_stale_or_changed_revision_is_rejected(self):
        self.turn("one")
        self.turn("two")
        original = self.summary(revision=2)
        self.conversation._apply_summary(original)
        invalid = [dict(revision=1), dict(revision=2),
                   dict(revision=3, through_event_seq=6),
                   dict(revision=3, from_event_seq=2),
                   dict(revision=3, created_at=11)]
        for changes in invalid:
            with self.subTest(changes=changes):
                self.reject(replace(original, **changes))

    def test_partial_turn_and_unpersisted_range_are_rejected(self):
        self.turn("one")
        for start, end in ((2, 3), (1, 2)):
            with self.subTest(start=start, end=end):
                self.reject(self.summary(start, end))
        c = self.conversation
        c.add_event("user_message", {"role": "user", "content": "pending"}, "two")
        c.add_event("run_result", {"termination": "exception"}, "two")
        self.reject(self.summary(4, 5))
        self.reject(self.summary(1, 5))

    def test_pending_continuation_of_selected_turn_is_rejected(self):
        self.turn("one")
        self.conversation.add_event("model_response", {"role": "assistant", "content": "late"}, "one")
        self.reject(self.summary())

    def test_unfinished_turn_inside_range_is_rejected(self):
        c = self.conversation
        c.add_event("user_message", {"role": "user", "content": "unfinished"}, "one")
        c.add_event("model_response", {"role": "assistant", "content": "partial"}, "one")
        c.finish_turn(self.repository)
        self.turn("two")
        self.reject(self.summary(1, 5))

    def test_api_history_without_matching_metadata_is_not_wiped(self):
        c = self.conversation
        c.context["memory"]["structured"] = [{"role": "user", "content": "loaded history"}]
        c.last_event_seq = 3
        self.reject(self.summary())


if __name__ == "__main__":
    unittest.main()
