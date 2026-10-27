"""Score saved JSONL response/reference pairs without calling any models."""

import argparse
import json
from pathlib import Path
from uuid import uuid4

from .scoring import DEFAULT_METRICS, RunScorer


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")

    try:
        scorer = RunScorer(args.metrics)
        with args.input.open(encoding="utf-8") as file:
            records = [json.loads(line) for line in file if line.strip()]
        if args.limit is not None:
            records = records[:args.limit]

        # Validate before creating output; old score-only results cannot be rescored.
        for index, record in enumerate(records, start=1):
            if not isinstance(record, dict) or not all(
                isinstance(record.get(key), str)
                for key in ("response", "expected_answer")
            ):
                raise ValueError(
                    f"Row {index} requires string response and expected_answer fields; "
                    "score-only results need their answer text recovered first"
                )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    scoring_run_id = uuid4().hex
    path = args.output.with_name(
        f"{args.output.stem}_{scoring_run_id}{args.output.suffix}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        for record in records:
            scores = scorer.score(record["response"], record["expected_answer"])
            file.write(json.dumps({
                **record,
                "scoring_run_id": scoring_run_id,
                "metrics": scores,
            }, ensure_ascii=False) + "\n")

    summary = {
        "scoring_run_id": scoring_run_id,
        "source_eval_ids": sorted({str(r["eval_id"]) for r in records if "eval_id" in r}),
        "input": str(args.input.resolve()),
        **scorer.summary(),
    }
    summary_path = path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Results: {path}\nSummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
