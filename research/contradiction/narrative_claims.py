"""Transparent, deliberately unvalidated MD&A claim-proxy extractor.

The extractor is not a sentiment model and does not infer contradiction from
embedding distance. It only emits a candidate when a topic anchor and an
explicit condition/assessment cue occur in the same sentence. Results are
proxies pending human annotation.
"""

from __future__ import annotations

from dataclasses import dataclass
import html
import re
from typing import Any, Iterable, Mapping

import pandas as pd

from .financial_topic_scores import TopicDefinition, topic_definitions, load_topic_config


@dataclass(frozen=True)
class Claim:
    filing_uid: str
    company_id: str
    uid_date: str | None
    sentence_id: str
    topic: str
    sentence: str
    acknowledgement_severity: float
    claim_applicable: bool
    applicability: str
    negated: bool
    mitigating_explanation: bool
    uncertainty: str
    extraction_method: str


_TAG_RE = re.compile(r"<[^>]+>")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+|(?<=</p>)", re.IGNORECASE)
_SENTENCE_SCAN_RE = re.compile(r"[^.!?\n]+(?:[.!?]|$)")
_PAST_RE = re.compile(r"\b(previously|prior|historically|last year|in the preceding|during \d{4})\b", re.I)
_FUTURE_RE = re.compile(r"\b(expect|expects|expected|anticipate|anticipates|will|may|could|should|plan|plans|forecast|outlook)\b", re.I)
_REASSURANCE_RE = re.compile(
    r"\b(strong|adequate|sufficient|ample|healthy|robust|solid|comfortable|"
    r"no (?:material )?(?:concern|concerns|risk|risks|problem|problems)|"
    r"not (?:a |an |material )?(?:concern|concerns|risk|risks|problem|problems)|"
    r"not expected to (?:have|result in|cause)|remain(?:s)? strong)\b", re.I
)
_CRITICAL_RE = re.compile(
    r"\b(unable to meet|unable to pay|default(?:ed|ing)?|covenant breach|"
    r"substantial doubt|insolvency|insolvent|going concern|severe liquidity|"
    r"material adverse|significant deterioration)\b", re.I
)
_MODERATE_RE = re.compile(
    r"\b(challenge|challenging|pressure|constrain(?:ed|ing)?|tight|declin(?:e|ed|ing)|"
    r"deteriorat(?:e|ed|ing)|weak(?:ness)?|risk|uncertain(?:ty)?|difficult|loss(?:es)?)\b", re.I
)
_MITIGATION_RE = re.compile(
    r"\b(new|available|committed|secured|undrawn|expanded|additional)\b.{0,70}\b"
    r"(credit facility|financing|funding|liquidity|borrowing capacity|credit line)\b|"
    r"\b(credit facility|financing|funding|borrowing capacity|credit line)\b.{0,70}\b"
    r"(supports?|provides?|ensures?|sufficient|available)\b", re.I
)


def clean_text(text: Any) -> str:
    text = html.unescape("" if text is None else str(text))
    text = _TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: Any) -> list[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return [s.strip() for s in _SENTENCE_RE.split(cleaned) if s.strip()]


def iter_relevant_sentences(text: Any, topics: tuple[TopicDefinition, ...]):
    """Yield only sentences containing a configured topic anchor.

    Full MD&A documents can be hundreds of thousands of characters. A
    generator avoids materializing every sentence and keeps evidence scanning
    bounded while preserving the original sentence order.
    """
    cleaned = clean_text(text)
    for index, match in enumerate(_SENTENCE_SCAN_RE.finditer(cleaned)):
        sentence = match.group(0).strip()
        if sentence and any(_topic_anchor(sentence, topic) for topic in topics):
            yield index, sentence


def _topic_anchor(sentence: str, topic: TopicDefinition) -> bool:
    lower = sentence.lower()
    return any(keyword.lower() in lower for keyword in topic.narrative_keywords)


def _claim_cue(sentence: str) -> bool:
    return bool(_REASSURANCE_RE.search(sentence) or _CRITICAL_RE.search(sentence) or _MODERATE_RE.search(sentence))


def _severity(sentence: str) -> float:
    if _CRITICAL_RE.search(sentence):
        return 1.0
    if _MODERATE_RE.search(sentence):
        return 0.55
    if _REASSURANCE_RE.search(sentence):
        return 0.0
    return 0.0


def _applicability(sentence: str) -> tuple[bool, str, str]:
    past = bool(_PAST_RE.search(sentence))
    future = bool(_FUTURE_RE.search(sentence))
    if past and future:
        return False, "uncertain_past_and_future", "past_and_future_context"
    if past:
        return False, "past_only", "past_context"
    if future:
        return False, "future_or_conditional", "future_or_conditional_context"
    return True, "current", "current_condition_proxy"


class NarrativeClaimExtractor:
    """Extract current-period management claims as auditable proxy records."""

    extraction_method = "topic_anchor_plus_explicit_condition_cue_rules_v1"

    def __init__(self, config: Mapping[str, Any] | None = None):
        self.config = dict(config or load_topic_config())
        self.topics = topic_definitions(self.config)

    def extract(self, records: pd.DataFrame) -> pd.DataFrame:
        required = {"uid", "item_7"}
        missing = required - set(records.columns)
        if missing:
            raise KeyError(f"Missing narrative columns: {sorted(missing)}")
        topic_rows: list[dict[str, Any]] = []
        for row in records.itertuples(index=False):
            values = row._asdict()
            uid = str(values["uid"])
            parts = uid.split("__", 1)
            company_id = parts[0]
            uid_date = parts[1] if len(parts) == 2 else None
            for index, sentence in iter_relevant_sentences(values.get("item_7", ""), self.topics):
                for topic in self.topics:
                    if not _topic_anchor(sentence, topic) or not _claim_cue(sentence):
                        continue
                    applicable, applicability, context = _applicability(sentence)
                    negated = bool(re.search(r"\b(no|not|never|without|none)\b", sentence, re.I))
                    mitigation = bool(_MITIGATION_RE.search(sentence))
                    uncertainty = context
                    if mitigation:
                        uncertainty = f"{uncertainty};mitigating_explanation_candidate"
                    topic_rows.append(
                        {
                            "filing_uid": uid,
                            "company_id": company_id,
                            "uid_date": uid_date,
                            "sentence_id": f"{uid}__s{index:04d}",
                            "topic": topic.name,
                            "sentence": sentence,
                            "acknowledgement_severity": float(_severity(sentence)),
                            "claim_applicable": bool(applicable),
                            "applicability": applicability,
                            "negated": bool(negated),
                            "mitigating_explanation": bool(mitigation),
                            "uncertainty": uncertainty,
                            "extraction_method": self.extraction_method,
                            "narrative_status": "unvalidated_proxy",
                            "source_field": "item_7",
                        }
                    )
        columns = [
            "filing_uid", "company_id", "uid_date", "sentence_id", "topic", "sentence",
            "acknowledgement_severity", "claim_applicable", "applicability", "negated",
            "mitigating_explanation", "uncertainty", "extraction_method", "narrative_status",
            "source_field",
        ]
        return pd.DataFrame(topic_rows, columns=columns)

    def extract_auditor_evidence(self, records: pd.DataFrame) -> pd.DataFrame:
        """Return topic-keyword auditor sentences without mixing them into A."""
        required = {"uid", "opinion_text"}
        missing = required - set(records.columns)
        if missing:
            raise KeyError(f"Missing auditor columns: {sorted(missing)}")
        rows: list[dict[str, Any]] = []
        for row in records.itertuples(index=False):
            values = row._asdict()
            uid = str(values["uid"])
            parts = uid.split("__", 1)
            for index, sentence in iter_relevant_sentences(values.get("opinion_text", ""), self.topics):
                for topic in self.topics:
                    if _topic_anchor(sentence, topic):
                        rows.append(
                            {
                                "filing_uid": uid,
                                "company_id": parts[0],
                                "uid_date": parts[1] if len(parts) == 2 else None,
                                "sentence_id": f"{uid}__auditor_s{index:04d}",
                                "topic": topic.name,
                                "sentence": sentence,
                                "source_field": "opinion_text",
                                "extraction_method": "topic_anchor_only_auditor_evidence_v1",
                                "combined_into_management_acknowledgement": False,
                            }
                        )
        return pd.DataFrame(rows, columns=[
            "filing_uid", "company_id", "uid_date", "sentence_id", "topic", "sentence",
            "source_field", "extraction_method", "combined_into_management_acknowledgement",
        ])


def aggregate_current_claims(claims: pd.DataFrame) -> pd.DataFrame:
    """Aggregate applicable claims conservatively using the maximum severity."""
    columns = ["filing_uid", "topic", "narrative_acknowledged_severity", "evidence_sentence_ids", "claim_count", "uncertain_claim_count"]
    if claims.empty:
        return pd.DataFrame(columns=columns)
    current = claims[claims["claim_applicable"]].copy()
    if current.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (uid, topic), group in current.groupby(["filing_uid", "topic"], sort=False):
        rows.append(
            {
                "filing_uid": uid,
                "topic": topic,
                "narrative_acknowledged_severity": float(group["acknowledgement_severity"].max()),
                "evidence_sentence_ids": ";".join(group["sentence_id"].astype(str)),
                "claim_count": int(len(group)),
                "uncertain_claim_count": int((group["uncertainty"].astype(str) != "current_condition_proxy").sum()),
            }
        )
    return pd.DataFrame(rows, columns=columns)
