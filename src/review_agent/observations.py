from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json


@dataclass(frozen=True)
class Observation:
    observation_id: str
    source: str
    revision: str
    path: str | None
    line_start: int | None
    line_end: int | None
    content_hash: str
    raw_artifact_ref: str
    context_view: str


class ObservationStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.observations_dir = run_dir / "observations"
        self.observations_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = run_dir / "observations.jsonl"
        self._observations: list[Observation] = []

    def record(
        self,
        source: str,
        revision: str,
        path: str | None,
        line_start: int | None,
        line_end: int | None,
        raw_content: str,
        context_view: str,
    ) -> Observation:
        content_hash = _sha256(raw_content)
        observation_id = _observation_id(
            source=source,
            revision=revision,
            path=path,
            line_start=line_start,
            line_end=line_end,
            content_hash=content_hash,
        )
        artifact_ref = f"observations/{observation_id}.txt"
        (self.run_dir / artifact_ref).write_text(raw_content, encoding="utf-8")
        observation = Observation(
            observation_id=observation_id,
            source=source,
            revision=revision,
            path=path,
            line_start=line_start,
            line_end=line_end,
            content_hash=content_hash,
            raw_artifact_ref=artifact_ref,
            context_view=context_view,
        )
        self._observations.append(observation)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(observation), ensure_ascii=False))
            handle.write("\n")
        return observation

    def list_observations(self) -> list[Observation]:
        return list(self._observations)

    def summaries_by_id(self) -> dict[str, str]:
        return {observation.observation_id: observation.context_view for observation in self._observations}


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _observation_id(
    source: str,
    revision: str,
    path: str | None,
    line_start: int | None,
    line_end: int | None,
    content_hash: str,
) -> str:
    seed = "|".join(
        [
            source,
            revision,
            path or "",
            "" if line_start is None else str(line_start),
            "" if line_end is None else str(line_end),
            content_hash,
        ]
    )
    return f"O-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:12]}"
