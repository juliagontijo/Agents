import contextlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from evals.metrics.scoring import RunScorer
from evals.metrics.text import rouge_l, token_overlap
from src.resources.conversation.repository import ConversationRepository


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TextMetricsTests(unittest.TestCase):
    def test_overlap_uses_word_counts(self):
        scores = token_overlap("red red blue", "red blue blue")
        for value in scores.values():
            self.assertAlmostEqual(value, 2 / 3)

    def test_partial_answer(self):
        scores = token_overlap("red blue green", "red green")
        self.assertAlmostEqual(scores["token_precision"], 2 / 3)
        self.assertEqual(scores["token_recall"], 1)
        self.assertAlmostEqual(scores["token_f1"], 0.8)
        self.assertAlmostEqual(rouge_l("red blue green", "red green")["rouge_l_f1"], 0.8)

    def test_rouge_detects_reversed_word_order(self):
        self.assertEqual(token_overlap("red blue green", "green blue red")["token_f1"], 1)
        self.assertAlmostEqual(rouge_l("red blue green", "green blue red")["rouge_l_f1"], 1 / 3)

    def test_normalization_and_negation(self):
        response = "- **NOOR:** 11:00–13:00; CAFÉ"
        reference = "Noor: 11:00-13:00; cafe\u0301"
        self.assertEqual(token_overlap(response, reference)["token_f1"], 1)
        self.assertEqual(rouge_l(response, reference)["rouge_l_f1"], 1)
        self.assertLess(token_overlap("not booked", "booked")["token_f1"], 1)

    def test_empty_and_disjoint_answers(self):
        for response, reference, expected in [("", "", 1), ("", "red", 0), ("red", "", 0), ("red", "blue", 0)]:
            with self.subTest(response=response, reference=reference):
                self.assertEqual(token_overlap(response, reference)["token_f1"], expected)
                self.assertEqual(rouge_l(response, reference)["rouge_l_f1"], expected)

    def test_summary_is_a_per_answer_mean(self):
        scorer = RunScorer()
        scorer.score("red", "red")
        scorer.score("red blue", "red")
        summary = scorer.summary()
        self.assertEqual(summary["answers_scored"], 2)
        self.assertEqual(summary["metrics"]["token_precision"], {"mean": 0.75, "count": 2})
        self.assertAlmostEqual(summary["metrics"]["token_f1"]["mean"], 5 / 6)

    def test_selection_and_invalid_inputs(self):
        scorer = RunScorer(["rouge_l"])
        self.assertEqual(scorer.score("red", "red"), {"rouge_l_f1": 1})
        with self.assertRaises(TypeError):
            scorer.score(None, "red")
        self.assertEqual(scorer.summary()["error_count"], 1)
        self.assertEqual(scorer.summary()["answers_scored"], 1)
        with self.assertRaises(ValueError):
            RunScorer(["unknown"])
        self.assertEqual(RunScorer([]).score("red", "red"), {})


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.input = self.root / "answers.jsonl"
        self.output = self.root / "results" / "scored.jsonl"
        self.records = [
            {"eval_id": "source", "turn_id": 1, "response": "red", "expected_answer": "red", "grader_response": {"score": 2}},
            {"eval_id": "source", "turn_id": 2, "response": "blue", "expected_answer": "red"},
        ]
        self.input.write_text("".join(json.dumps(row) + "\n" for row in self.records))

    def run_module(self, module, *arguments):
        return subprocess.run(
            [sys.executable, "-m", module, *map(str, arguments)],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
        )

    def test_standalone_preserves_answers_and_writes_summary(self):
        original = self.input.read_bytes()
        result = self.run_module("evals.metrics.run", "--input", self.input, "--output", self.output)
        self.assertEqual(result.returncode, 0, result.stderr)
        path = next(self.output.parent.glob("scored_*.jsonl"))
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([row["eval_id"] for row in rows], ["source", "source"])
        self.assertEqual(rows[0]["grader_response"], {"score": 2})
        self.assertEqual(rows[0]["response"], self.records[0]["response"])
        self.assertIn(rows[0]["scoring_run_id"], path.name)
        summary = json.loads(path.with_suffix(".summary.json").read_text())
        self.assertEqual(summary["source_eval_ids"], ["source"])
        self.assertEqual(summary["metrics"]["token_f1"], {"mean": 0.5, "count": 2})
        self.assertEqual(self.input.read_bytes(), original)

    def test_global_toml_dispatch_and_limit(self):
        config = self.root / "evals.toml"
        family = PROJECT_ROOT / "evals" / "metrics" / "config.toml"
        config.write_text(
            '[[runs]]\nname = "offline"\ncase = "saved_answers"\n'
            f'config = {json.dumps(str(family))}\n'
            'input = "answers.jsonl"\noutput = "results/scored.jsonl"\n'
            'limit = 1\nmetrics = ["rouge_l"]\n'
        )
        result = self.run_module("evals.run", "--config", config)
        self.assertEqual(result.returncode, 0, result.stderr)
        path = next(self.output.parent.glob("scored_*.jsonl"))
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["metrics"], {"rouge_l_f1": 1})

    def test_score_only_input_fails_before_creating_output(self):
        self.input.write_text('{"eval_id":"old","turn_id":1,"grader_response":{"score":2}}\n')
        result = self.run_module("evals.metrics.run", "--input", self.input, "--output", self.output)
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires string response and expected_answer", result.stderr)
        self.assertFalse(self.output.parent.exists())

    def test_memory_runner_integration_without_model_calls(self):
        agent_module = ModuleType("src.myagent.agent")
        agent_module.TRACE_PATH = self.root / "unused-trace.jsonl"
        calls = []

        def fake_agent(**kwargs):
            calls.append(kwargs)
            return "red", kwargs["conversation"], "trace-id"

        agent_module.run_agent = fake_agent
        model_module = ModuleType("resources.system.system")

        def fake_grader(messages, tools, model, trace):
            self.assertEqual(tools, [])
            self.assertIn("expected_answer", json.loads(messages[1]["content"]))
            message = SimpleNamespace(content='{"score":2,"reason":"Correct"}')
            return SimpleNamespace(choices=[SimpleNamespace(message=message)]), {}

        model_module.call_model = fake_grader
        progress_module = ModuleType("tqdm")

        class Progress(list):
            def set_postfix(self, **kwargs):
                pass

        progress_module.tqdm = lambda items, **kwargs: Progress(items)
        dataset = self.root / "dataset.jsonl"
        dataset.write_text(''.join(json.dumps({"id": i, "prompt": "Recall the colour", "expected_answer": "red"}) + "\n" for i in [1, 2]))
        arguments = ["memory_eval", "--input", str(dataset), "--output", str(self.output), "--model", "mock", "--grader-model", "mock"]
        modules = {"src.myagent.agent": agent_module, "resources.system.system": model_module, "tqdm": progress_module}
        repository = ConversationRepository(self.root / "memory")
        with patch.dict(sys.modules, modules), patch.object(sys, "argv", arguments), patch("src.resources.conversation.repository.ConversationRepository", return_value=repository) as repository_class, patch.object(repository, "open", wraps=repository.open) as open_conversation, contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            runpy.run_module("evals.memory_design.eval_memory_script", run_name="__main__")
        repository_class.assert_called_once()
        open_conversation.assert_called_once_with(client_id=calls[0]["conversation"].client_id)
        path = next(self.output.parent.glob("scored_*.jsonl"))
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["metrics"]["token_f1"], 1)
        self.assertEqual(rows[0]["trace_run_id"], "trace-id")
        self.assertEqual(rows[0]["expected_answer"], "red")
        self.assertTrue(all(call["eval_mode"] for call in calls))
        self.assertIs(calls[0]["conversation"], calls[1]["conversation"])
        self.assertIs(calls[0]["repository"], calls[1]["repository"])
        summary = json.loads(path.with_suffix(".summary.json").read_text())
        self.assertEqual(summary["eval_id"], rows[0]["eval_id"])
        self.assertEqual(summary["metrics"]["rouge_l_f1"], {"mean": 1, "count": 2})


if __name__ == "__main__":
    unittest.main()
