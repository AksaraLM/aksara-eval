"""Base classes for Aksara-Indo-Bench tasks."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Any
from statistics import mean


@dataclass
class TaskResult:
    name: str
    track: str
    score: float
    n: int
    extra: dict = field(default_factory=dict)


class BenchTask:
    """Subclass this to add a task.

    Required class attributes:
        name:    short task identifier (e.g. "indommlu")
        track:   one of {"knowledge", "culture", "reasoning", "reading", "regional", "safety", "generation"}
        dataset: HF dataset id, or None if loaded locally
    """

    name: str = ""
    track: str = ""
    dataset: str | None = None

    def iter_examples(self) -> Iterable[dict]:
        raise NotImplementedError

    def format_prompt(self, example: dict) -> str:
        raise NotImplementedError

    def extract_answer(self, output: str, example: dict) -> Any:
        return output.strip()

    def score_one(self, pred: Any, example: dict) -> float:
        raise NotImplementedError

    def run(self, model, limit: int | None = None) -> TaskResult:
        scores = []
        for i, ex in enumerate(self.iter_examples()):
            if limit is not None and i >= limit:
                break
            prompt = self.format_prompt(ex)
            raw = model.generate(prompt)
            pred = self.extract_answer(raw, ex)
            scores.append(self.score_one(pred, ex))
        return TaskResult(
            name=self.name,
            track=self.track,
            score=mean(scores) if scores else 0.0,
            n=len(scores),
        )
