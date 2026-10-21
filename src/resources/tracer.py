from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


# logg raw tracing for evals and observability
@dataclass
class Tracer:
    client_id: str
    conversation_id: str
    model: str
    run_id: str = field(default_factory=lambda: uuid4().hex)
    started_at: str = field(default_factory=timestamp)
    events: list[dict[str, Any]] = field(default_factory=list)
    _started: float = field(default_factory=perf_counter, repr=False)

    def append(self, event_type: str, payload: Any) -> None:
        self.events.append({
            "sequence": len(self.events),
            "type": event_type,
            "recorded_at": timestamp(),
            "elapsed_ms": round(
                (perf_counter() - self._started) * 1000, 3
            ),
            "payload": deepcopy(payload),
        })

    def dump(self, path: Path) -> None:
        record = asdict(self)
        record.pop("_started")
        record["duration_ms"] = round(
            (perf_counter() - self._started) * 1000, 3
        )
        line = json.dumps(record, ensure_ascii=False) + "\n"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as output:
            output.write(line)