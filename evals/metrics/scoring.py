"""Shared metric selection and per-answer/run scoring for any eval family."""

import statistics

from .text import METRIC_VERSION, NORMALIZATION, rouge_l, token_overlap


METRICS = {
    "token_overlap": token_overlap,
    "rouge_l": rouge_l,
}
DEFAULT_METRICS = tuple(METRICS)


class RunScorer:
    def __init__(self, metrics=DEFAULT_METRICS):
        self.metrics = list(dict.fromkeys(metrics))
        unknown = set(self.metrics) - METRICS.keys()
        if unknown:
            raise ValueError(f"Unknown metrics: {', '.join(sorted(unknown))}")
        self.scores = []
        self.error_count = 0

    def score(self, response: str, expected_answer: str) -> dict[str, float]:
        try:
            if not isinstance(response, str) or not isinstance(expected_answer, str):
                raise TypeError("response and expected_answer must be strings")
            scores = {}
            for name in self.metrics:
                scores.update(METRICS[name](response, expected_answer))
        except (TypeError, ValueError):
            self.error_count += 1
            raise

        self.scores.append(scores.copy())
        return scores

    def summary(self) -> dict:
        # Each answer has equal weight, irrespective of its length.
        keys = self.scores[0].keys() if self.scores else ()
        return {
            "metric_version": METRIC_VERSION,
            "normalization": NORMALIZATION,
            "selected_metrics": self.metrics,
            "answers_scored": len(self.scores),
            "error_count": self.error_count,
            "metrics": {
                key: {
                    "mean": statistics.mean(row[key] for row in self.scores),
                    "count": len(self.scores),
                }
                for key in keys
            },
        }
