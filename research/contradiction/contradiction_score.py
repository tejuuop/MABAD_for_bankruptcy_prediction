"""Static Track A contradiction score and inspectable output contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json

import numpy as np
import pandas as pd

from .financial_topic_scores import FinancialTopicScorer, load_topic_config, topic_definitions
from .narrative_claims import NarrativeClaimExtractor, aggregate_current_claims


def _record_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    if "uid" not in frame.columns:
        raise KeyError("Records must include uid")
    result = pd.DataFrame(index=frame.index)
    result["uid"] = frame["uid"].astype(str)
    parts = result["uid"].str.split("__", n=1, expand=True)
    result["company_id"] = parts[0]
    result["uid_date"] = parts[1] if parts.shape[1] > 1 else pd.Series(pd.NA, index=frame.index)
    result["prediction_cutoff"] = pd.to_datetime(result["uid_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    result["prediction_cutoff_provenance"] = "uid_date_order_only_not_verified_publication_date"
    return result


class TrackAScorer:
    """Compute static B/A/S/R components with no temporal or future state."""

    def __init__(self, config: Mapping[str, Any] | None = None):
        self.config = dict(config or load_topic_config())
        if self.config.get("temporal_scoring", {}).get("enabled", False):
            raise ValueError("Temporal scoring must remain disabled in Track A static MVP")
        self.topics = topic_definitions(self.config)
        self.financial = FinancialTopicScorer(self.config)
        self.claim_extractor = NarrativeClaimExtractor(self.config)
        self._fitted = False

    def fit(self, train_records: pd.DataFrame) -> "TrackAScorer":
        self.financial.fit(train_records)
        self._fitted = True
        return self

    def _topic_payload(
        self,
        uid: str,
        topic_name: str,
        numerical: pd.Series,
        claim: pd.Series | None,
    ) -> dict[str, Any]:
        b = numerical.get(f"{topic_name}__numerical_severity", np.nan)
        b = float(b) if pd.notna(b) else None
        numerical_available = bool(numerical.get(f"{topic_name}__numerical_available", 0))
        a = None if claim is None or pd.isna(claim.get("narrative_acknowledged_severity")) else float(claim["narrative_acknowledged_severity"])
        eligible = b is not None and a is not None and numerical_available
        static_gap = max(0.0, b - a) if eligible else None
        reverse_gap = max(0.0, a - b) if eligible else None
        return {
            "numerical_severity": b,
            "narrative_acknowledged_severity": a,
            "static_gap": static_gap,
            "temporal_gap": None,
            "reverse_gap": reverse_gap,
            "topic_score": static_gap,
            "temporal_available": False,
            "numerical_available": numerical_available,
            "narrative_available": a is not None,
            "eligible": eligible,
            "evidence_sentence_ids": [] if claim is None else str(claim.get("evidence_sentence_ids", "")).split(";") if str(claim.get("evidence_sentence_ids", "")) else [],
            "claim_count": 0 if claim is None else int(claim.get("claim_count", 0)),
            "uncertain_claim_count": 0 if claim is None else int(claim.get("uncertain_claim_count", 0)),
            "limitations": [
                "narrative_acknowledgement_is_an_unvalidated_rules_based_proxy",
                "temporal_scoring_disabled_without_verified_filing_publication_date",
            ],
        }

    def transform(self, records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if not self._fitted:
            raise RuntimeError("TrackAScorer.fit must run before transform")
        if "item_7" not in records.columns:
            raise KeyError("Records must include item_7 for narrative extraction")
        numeric = self.financial.transform(records)
        claims = self.claim_extractor.extract(records)
        claim_summary = aggregate_current_claims(claims)
        claim_lookup = {
            (str(row.filing_uid), str(row.topic)): row._asdict()
            for row in claim_summary.itertuples(index=False)
        }

        output = _record_metadata(records)
        evidence = claims.copy()
        auditor_evidence = self.claim_extractor.extract_auditor_evidence(records)
        if not auditor_evidence.empty:
            auditor_evidence = auditor_evidence.copy()
            auditor_evidence["evidence_role"] = "separate_auditor_evidence_not_in_A"
        if not evidence.empty:
            evidence["evidence_role"] = "management_claim_candidate"
        evidence = pd.concat([evidence, auditor_evidence], ignore_index=True, sort=False)

        payloads = []
        for pos, uid in enumerate(output["uid"]):
            row_numeric = numeric.iloc[pos]
            topics_payload = {}
            for topic in self.topics:
                claim = claim_lookup.get((uid, topic.name))
                topics_payload[topic.name] = self._topic_payload(uid, topic.name, row_numeric, claim)
            eligible = [p for p, t in topics_payload.items() if t["eligible"]]
            weight_by_name = {t.name: t.weight for t in self.topics}
            eligible_weight = sum(weight_by_name[n] for n in eligible)
            total_weight = sum(weight_by_name.values())
            contradiction = None
            if eligible_weight > 0:
                contradiction = sum(weight_by_name[n] * topics_payload[n]["topic_score"] for n in eligible) / eligible_weight
            coverage = eligible_weight / total_weight if total_weight else 0.0
            missing_topics = [t.name for t in self.topics if t.name not in eligible]
            payloads.append({
                "contradiction_score": contradiction,
                "coverage": float(coverage),
                "topics": topics_payload,
                "missing_topics": missing_topics,
                "score_version": self.config["score_version"],
                "narrative_status": self.config["narrative_status"],
            })

        output["contradiction_score"] = [p["contradiction_score"] for p in payloads]
        output["coverage"] = [p["coverage"] for p in payloads]
        output["eligible_topic_count"] = [sum(int(t["eligible"]) for t in p["topics"].values()) for p in payloads]
        output["missing_topics"] = [";".join(p["missing_topics"]) for p in payloads]
        output["score_version"] = self.config["score_version"]
        output["narrative_status"] = self.config["narrative_status"]
        output["topic_components_json"] = [json.dumps(p["topics"], sort_keys=True) for p in payloads]
        output["output_contract_json"] = [json.dumps(p, sort_keys=True) for p in payloads]

        # Preserve labels for join audits, but never use them in score construction.
        if "label" in records.columns:
            output["label"] = records["label"].astype(int).to_numpy()
        return output, evidence, numeric

    def fit_transform(self, train_records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self.fit(train_records).transform(train_records)
