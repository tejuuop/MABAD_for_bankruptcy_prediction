"""Training-only numerical severity for the Track A MVP.

This module deliberately uses a small, explicit set of raw XBRL concepts. It
does not select alternate concepts when a configured concept is unavailable.
Each ratio is converted to an adverse-direction empirical percentile using
only the rows passed to ``fit``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import json

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RatioDefinition:
    name: str
    numerator: str
    denominator: str
    adverse_direction: str
    denominator_rule: str


@dataclass(frozen=True)
class TopicDefinition:
    name: str
    weight: float
    ratios: tuple[RatioDefinition, ...]
    narrative_keywords: tuple[str, ...]


def load_topic_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else Path(__file__).with_name("topic_config.json")
    return json.loads(path.read_text(encoding="utf-8"))


def topic_definitions(config: Mapping[str, Any]) -> tuple[TopicDefinition, ...]:
    topics = []
    for name, raw in config["topics"].items():
        ratios = tuple(RatioDefinition(**ratio) for ratio in raw["ratios"])
        weight = float(raw["weight"])
        if weight < 0:
            raise ValueError(f"Topic weight must be nonnegative: {name}")
        topics.append(TopicDefinition(name, weight, ratios, tuple(raw["narrative_keywords"])))
    if not topics or sum(t.weight for t in topics) <= 0:
        raise ValueError("At least one positive-weight topic is required")
    return tuple(topics)


def _as_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise KeyError(f"Configured XBRL field is absent: {column}")
    return pd.to_numeric(frame[column], errors="coerce")


class FinancialTopicScorer:
    """Compute bounded, topic-specific numerical severity without labels."""

    def __init__(self, config: Mapping[str, Any] | None = None):
        self.config = dict(config or load_topic_config())
        self.topics = topic_definitions(self.config)
        self._fitted = False
        self._references: dict[tuple[str, str], np.ndarray] = {}
        self.fit_row_count_: int | None = None
        self.fit_uids_: tuple[str, ...] = ()

    @property
    def required_columns(self) -> set[str]:
        return {
            field
            for topic in self.topics
            for ratio in topic.ratios
            for field in (ratio.numerator, ratio.denominator)
        }

    def _ratio_values(self, frame: pd.DataFrame, ratio: RatioDefinition) -> pd.Series:
        numerator = _as_numeric(frame, ratio.numerator)
        denominator = _as_numeric(frame, ratio.denominator)
        valid_denominator = denominator.gt(0) & np.isfinite(denominator)
        values = numerator.div(denominator).where(valid_denominator)
        return values.replace([np.inf, -np.inf], np.nan)

    def fit(self, train_frame: pd.DataFrame) -> "FinancialTopicScorer":
        missing = self.required_columns - set(train_frame.columns)
        if missing:
            raise KeyError(f"Missing configured financial fields: {sorted(missing)}")
        self._references = {}
        for topic in self.topics:
            for ratio in topic.ratios:
                values = self._ratio_values(train_frame, ratio).dropna().to_numpy(float)
                if len(values):
                    self._references[(topic.name, ratio.name)] = np.sort(values)
        self._fitted = True
        self.fit_row_count_ = len(train_frame)
        self.fit_uids_ = tuple(train_frame["uid"].astype(str)) if "uid" in train_frame else ()
        return self

    @staticmethod
    def _adverse_percentile(values: pd.Series, reference: np.ndarray, direction: str) -> pd.Series:
        finite = values.notna().to_numpy()
        result = np.full(len(values), np.nan, dtype=float)
        if finite.any() and len(reference):
            ranks = np.searchsorted(reference, values.to_numpy(float)[finite], side="right")
            percentile = ranks / (len(reference) + 1.0)
            if direction == "lower":
                percentile = 1.0 - percentile
            elif direction != "higher":
                raise ValueError(f"Unknown adverse direction: {direction}")
            result[finite] = np.clip(percentile, 0.0, 1.0)
        return pd.Series(result, index=values.index)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("FinancialTopicScorer.fit must run before transform")
        missing = self.required_columns - set(frame.columns)
        if missing:
            raise KeyError(f"Missing configured financial fields: {sorted(missing)}")

        out = pd.DataFrame(index=frame.index)
        for topic in self.topics:
            ratio_severities = []
            available_names = []
            for ratio in topic.ratios:
                values = self._ratio_values(frame, ratio)
                reference = self._references.get((topic.name, ratio.name), np.array([]))
                severity = self._adverse_percentile(values, reference, ratio.adverse_direction)
                prefix = f"{topic.name}__{ratio.name}"
                out[f"{prefix}__value"] = values
                out[f"{prefix}__severity"] = severity
                out[f"{prefix}__available"] = severity.notna().astype(int)
                if reference.size:
                    ratio_severities.append(severity)
                    available_names.append(ratio.name)

            severity_frame = pd.concat(ratio_severities, axis=1) if ratio_severities else pd.DataFrame(index=frame.index)
            if not severity_frame.empty:
                topic_available = severity_frame.notna().any(axis=1)
                topic_severity = severity_frame.mean(axis=1, skipna=True).where(topic_available)
            else:
                topic_available = pd.Series(False, index=frame.index)
                topic_severity = pd.Series(np.nan, index=frame.index)
            out[f"{topic.name}__numerical_severity"] = topic_severity
            out[f"{topic.name}__numerical_available"] = topic_available.astype(int)
            out[f"{topic.name}__available_ratio_count"] = severity_frame.notna().sum(axis=1).astype(int)
            out[f"{topic.name}__configured_ratio_count"] = len(topic.ratios)
            out[f"{topic.name}__reference_ratio_count"] = len(available_names)
        return out

    def fit_transform(self, train_frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_frame).transform(train_frame)
