#!/usr/bin/env python3
"""
Phase 0 analysis for reliability/completeness research.

This script is intentionally separate from the training pipeline. It does not
train models, mutate saved weights, or overwrite historical result files.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "research" / "phase0" / "outputs"
RANDOM_SEED = 42


class MLP(nn.Module):
    def __init__(self, in_dim: int, hid: int, latent: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, latent))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class LeafEmbed(nn.Module):
    def __init__(self, num_trees: int, num_leaves: int, emb_dim: int):
        super().__init__()
        self.embs = nn.ModuleList(
            [nn.Embedding(num_leaves + 1, emb_dim) for _ in range(num_trees)]
        )
        self.num_trees = num_trees

    def forward(self, idx):
        x = 0.0
        for t, emb in enumerate(self.embs):
            x = x + emb(idx[:, t])
        return x / self.num_trees


@dataclass(frozen=True)
class ArtifactChain:
    name: str
    tri_sad_path: Path
    xgb_path: Path
    text_dim: int
    text_source: str
    historical_label: str
    notes: str


def parse_uid(uid: pd.Series) -> tuple[pd.Series, pd.Series]:
    parts = uid.astype(str).str.split("__", n=1, expand=True)
    return parts[0], pd.to_datetime(parts[1])


def time_split_mask(meta: pd.DataFrame) -> pd.Series:
    positives = meta[meta["label"] == 1].sort_values("date").reset_index(drop=True)
    train_end = int(0.65 * len(positives))
    val_end = int(0.75 * len(positives))
    train_max_date = positives.iloc[train_end]["date"]
    val_max_date = positives.iloc[val_end]["date"]

    split = pd.Series("test", index=meta.index, dtype="object")
    split.loc[meta["date"] <= train_max_date] = "train"
    split.loc[(meta["date"] > train_max_date) & (meta["date"] <= val_max_date)] = "val"
    return split


def bootstrap_ci(
    labels: np.ndarray,
    values: np.ndarray,
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    labels = np.asarray(labels)
    values = np.asarray(values)
    point = float(stat_fn(labels, values))
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        try:
            stats.append(float(stat_fn(labels[idx], values[idx])))
        except ValueError:
            continue
    if not stats:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi)


def mean_diff_pos_minus_neg(labels: np.ndarray, values: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    values = np.asarray(values, dtype=float)
    return values[labels == 1].mean() - values[labels == 0].mean()


def recall_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_pos = labels.sum()
    if n_pos == 0:
        return float("nan")
    top = np.argsort(scores)[::-1][:k]
    return float(labels[top].sum() / n_pos)


def metric_block(labels: np.ndarray, scores: np.ndarray) -> dict:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    out = {"ap": None, "auc": None, "recall_at_50": None, "recall_at_100": None}
    if labels.sum() > 0 and len(np.unique(labels)) == 2:
        out["ap"] = float(average_precision_score(labels, scores))
        out["auc"] = float(roc_auc_score(labels, scores))
        out["recall_at_50"] = recall_at_k(labels, scores, 50)
        out["recall_at_100"] = recall_at_k(labels, scores, 100)
    return out


def build_completeness_table() -> pd.DataFrame:
    rows = []
    for chunk in pd.read_csv(DATA_DIR / "numerical.csv", chunksize=2000):
        feature_cols = [c for c in chunk.columns if c not in ("uid", "label")]
        company_id, date = parse_uid(chunk["uid"])
        nonnull = chunk[feature_cols].notna().sum(axis=1)
        rows.append(
            pd.DataFrame(
                {
                    "uid": chunk["uid"].to_numpy(),
                    "company_id": company_id.to_numpy(),
                    "date": date.to_numpy(),
                    "year": date.dt.year.to_numpy(),
                    "label": chunk["label"].astype(int).to_numpy(),
                    "financial_nonnull_count": nonnull.to_numpy(),
                    "financial_missing_pct": 1.0 - (nonnull.to_numpy() / len(feature_cols)),
                    "financial_available": (nonnull.to_numpy() > 0).astype(int),
                }
            )
        )
    completeness = pd.concat(rows, ignore_index=True)

    text_rows = []
    for chunk in pd.read_csv(
        DATA_DIR / "mda_auditor.csv",
        chunksize=1000,
        usecols=["uid", "item_7", "opinion_text"],
    ):
        mda = chunk["item_7"].fillna("").astype(str)
        auditor = chunk["opinion_text"].fillna("").astype(str)
        text_rows.append(
            pd.DataFrame(
                {
                    "uid": chunk["uid"].to_numpy(),
                    "mda_char_count": mda.str.len().to_numpy(),
                    "auditor_char_count": auditor.str.len().to_numpy(),
                    "mda_available": (mda.str.strip().str.len() > 0).astype(int).to_numpy(),
                    "auditor_available": (auditor.str.strip().str.len() > 0)
                    .astype(int)
                    .to_numpy(),
                }
            )
        )
    text = pd.concat(text_rows, ignore_index=True)
    completeness = completeness.merge(text, on="uid", how="inner", validate="one_to_one")
    completeness["split"] = time_split_mask(completeness)
    return completeness


def summarize_phase0a(completeness: pd.DataFrame) -> dict:
    labels = completeness["label"].to_numpy()
    summary = {
        "n_rows": int(len(completeness)),
        "n_positive": int(completeness["label"].sum()),
        "n_companies": int(completeness["company_id"].nunique()),
        "date_min": str(completeness["date"].min().date()),
        "date_max": str(completeness["date"].max().date()),
        "split_counts": completeness.groupby("split")["label"].agg(["count", "sum"]).to_dict(),
        "signals": {},
    }
    signal_defs = {
        "financial_nonnull_count": completeness["financial_nonnull_count"].to_numpy(),
        "financial_missing_pct": completeness["financial_missing_pct"].to_numpy(),
        "mda_char_count": completeness["mda_char_count"].to_numpy(),
        "auditor_char_count": completeness["auditor_char_count"].to_numpy(),
    }
    for name, values in signal_defs.items():
        pos = values[labels == 1]
        neg = values[labels == 0]
        diff, lo, hi = bootstrap_ci(labels, values, mean_diff_pos_minus_neg)
        signal = {
            "positive_mean": float(pos.mean()),
            "negative_mean": float(neg.mean()),
            "positive_median": float(np.median(pos)),
            "negative_median": float(np.median(neg)),
            "mean_difference_pos_minus_neg": diff,
            "mean_difference_95ci": [lo, hi],
            "as_positive_ranking": metric_block(labels, values),
            "as_negative_ranking": metric_block(labels, -values),
        }
        summary["signals"][name] = signal
    return summary


def write_markdown_summary(summary: dict, output_path: Path) -> None:
    lines = [
        "# Phase 0A Completeness Summary",
        "",
        f"- Rows: {summary['n_rows']}",
        f"- Positives: {summary['n_positive']}",
        f"- Companies: {summary['n_companies']}",
        f"- Date range: {summary['date_min']} to {summary['date_max']}",
        "",
        "## Signal Summary",
        "",
        "| signal | pos mean | neg mean | diff 95% CI | best AP direction | best AP | best Recall@50 | best Recall@100 |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for name, signal in summary["signals"].items():
        pos_ap = signal["as_positive_ranking"]["ap"]
        neg_ap = signal["as_negative_ranking"]["ap"]
        if neg_ap is not None and (pos_ap is None or neg_ap > pos_ap):
            direction = "lower values riskier"
            metrics = signal["as_negative_ranking"]
        else:
            direction = "higher values riskier"
            metrics = signal["as_positive_ranking"]
        ci = signal["mean_difference_95ci"]
        lines.append(
            "| {name} | {pm:.3f} | {nm:.3f} | [{lo:.3f}, {hi:.3f}] | {direction} | {ap:.4f} | {r50:.4f} | {r100:.4f} |".format(
                name=name,
                pm=signal["positive_mean"],
                nm=signal["negative_mean"],
                lo=ci[0],
                hi=ci[1],
                direction=direction,
                ap=metrics["ap"] if metrics["ap"] is not None else float("nan"),
                r50=metrics["recall_at_50"]
                if metrics["recall_at_50"] is not None
                else float("nan"),
                r100=metrics["recall_at_100"]
                if metrics["recall_at_100"] is not None
                else float("nan"),
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase0a() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    completeness = build_completeness_table()
    completeness.to_csv(OUTPUT_DIR / "completeness_by_filing.csv", index=False)

    yearly = (
        completeness.groupby("year")
        .agg(
            filings=("uid", "count"),
            bankruptcies=("label", "sum"),
            financial_nonnull_mean=("financial_nonnull_count", "mean"),
            financial_missing_pct_mean=("financial_missing_pct", "mean"),
            mda_chars_median=("mda_char_count", "median"),
            auditor_chars_median=("auditor_char_count", "median"),
        )
        .reset_index()
    )
    yearly.to_csv(OUTPUT_DIR / "yearly_completeness.csv", index=False)

    summary = summarize_phase0a(completeness)
    (OUTPUT_DIR / "phase0a_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_markdown_summary(summary, OUTPUT_DIR / "phase0a_summary.md")
    return completeness


def load_artifact_chain() -> ArtifactChain:
    chain = ArtifactChain(
        name="active_finbert_mabad",
        tri_sad_path=REPO_ROOT / "models" / "saved_models" / "multisource" / "MABAD" / "tri_sad.pt",
        xgb_path=REPO_ROOT / "models" / "saved_models" / "multisource" / "MABAD" / "xgb_clf.pkl",
        text_dim=768,
        text_source="data/embeddings/finbert_mda.pkl + data/embeddings/finbert_auditor.pkl",
        historical_label="latest saved MABAD artifact; FinBERT-shaped, not the TF-IDF MABAD row",
        notes=(
            "The saved state has 768-dimensional text encoders, so it must be paired "
            "with FinBERT embeddings. It must not be paired with 1000-dimensional TF-IDF vectors."
        ),
    )
    state = torch.load(chain.tri_sad_path, map_location="cpu")
    first_mda_weight = next(iter(state["mda_enc"].values()))
    observed_text_dim = int(first_mda_weight.shape[1])
    if observed_text_dim != chain.text_dim:
        raise RuntimeError(
            f"Artifact mismatch: expected text_dim={chain.text_dim}, got {observed_text_dim}"
        )
    chain_record = {
        "name": chain.name,
        "tri_sad_path": str(chain.tri_sad_path.relative_to(REPO_ROOT)),
        "xgb_path": str(chain.xgb_path.relative_to(REPO_ROOT)),
        "text_dim": chain.text_dim,
        "observed_text_dim": observed_text_dim,
        "text_source": chain.text_source,
        "historical_label": chain.historical_label,
        "notes": chain.notes,
        "leaf_meta": state["leaf_meta"],
        "center_shapes": {
            key: list(value.shape) for key, value in state.items() if key.startswith("c_")
        },
    }
    (OUTPUT_DIR / "artifact_chain_active_finbert_mabad.json").write_text(
        json.dumps(chain_record, indent=2), encoding="utf-8"
    )
    return chain


def load_text_embedding_dicts() -> tuple[dict, dict]:
    import pickle

    with open(DATA_DIR / "embeddings" / "finbert_mda.pkl", "rb") as handle:
        mda = pickle.load(handle)
    with open(DATA_DIR / "embeddings" / "finbert_auditor.pkl", "rb") as handle:
        auditor = pickle.load(handle)
    return mda, auditor


def run_phase0b(completeness: pd.DataFrame) -> pd.DataFrame:
    chain = load_artifact_chain()
    state = torch.load(chain.tri_sad_path, map_location="cpu")
    xgb_clf = joblib.load(chain.xgb_path)
    mda_dict, auditor_dict = load_text_embedding_dicts()

    leaf_meta = state["leaf_meta"]
    latent_dim = int(state["c_tab"].shape[0])
    leaf_embed = LeafEmbed(leaf_meta["num_trees"], leaf_meta["num_leaves"], latent_dim)
    mda_enc = MLP(chain.text_dim, 512, latent_dim)
    aud_enc = MLP(chain.text_dim, 512, latent_dim)
    leaf_embed.load_state_dict(state["leaf_embed"])
    mda_enc.load_state_dict(state["mda_enc"])
    aud_enc.load_state_dict(state["aud_enc"])
    leaf_embed.eval()
    mda_enc.eval()
    aud_enc.eval()
    c_tab = state["c_tab"]
    c_mda = state["c_mda"]
    c_aud = state["c_aud"]

    meta = completeness[completeness["split"].isin(["val", "test"])].copy()
    meta = meta.sort_values("date").reset_index(drop=True)
    needed_uids = set(meta["uid"])

    rows = []
    feature_cols = None
    for chunk in pd.read_csv(DATA_DIR / "numerical.csv", chunksize=2000):
        chunk = chunk[chunk["uid"].isin(needed_uids)].copy()
        if chunk.empty:
            continue
        if feature_cols is None:
            feature_cols = [c for c in chunk.columns if c not in ("uid", "label")]
        chunk = chunk.set_index("uid")
        chunk_meta = meta.set_index("uid").loc[chunk.index].reset_index()
        x_tab = chunk[feature_cols]
        leaf_idx = xgb_clf.apply(x_tab).astype(np.int64)
        x_mda = np.stack([mda_dict[uid] for uid in chunk.index]).astype(np.float32)
        x_aud = np.stack([auditor_dict[uid] for uid in chunk.index]).astype(np.float32)
        with torch.no_grad():
            z_tab = F.normalize(leaf_embed(torch.from_numpy(leaf_idx)), dim=-1)
            z_mda = mda_enc(torch.from_numpy(x_mda))
            z_aud = aud_enc(torch.from_numpy(x_aud))
            d_tab = torch.norm(z_tab - c_tab, dim=1).numpy()
            d_mda = torch.norm(z_mda - c_mda, dim=1).numpy()
            d_aud = torch.norm(z_aud - c_aud, dim=1).numpy()
        dist = pd.DataFrame(
            {
                "uid": chunk.index.to_numpy(),
                "d_financial": d_tab,
                "d_mda": d_mda,
                "d_auditor": d_aud,
            }
        )
        rows.append(chunk_meta.merge(dist, on="uid", how="left"))

    out = pd.concat(rows, ignore_index=True)
    distance_cols = ["d_financial", "d_mda", "d_auditor"]
    out["d_max"] = out[distance_cols].max(axis=1)
    out["dominant_modality"] = out[distance_cols].idxmax(axis=1).str.replace("d_", "", regex=False)
    out["gap_financial_mda"] = (out["d_financial"] - out["d_mda"]).abs()
    out["gap_financial_auditor"] = (out["d_financial"] - out["d_auditor"]).abs()
    out["gap_mda_auditor"] = (out["d_mda"] - out["d_auditor"]).abs()
    out["distance_max_minus_min"] = out[distance_cols].max(axis=1) - out[distance_cols].min(axis=1)
    out["dominant_margin"] = out[distance_cols].apply(
        lambda row: float(np.sort(row.to_numpy())[-1] - np.sort(row.to_numpy())[-2]), axis=1
    )
    out.to_csv(OUTPUT_DIR / "active_finbert_mabad_valtest_modality_distances.csv", index=False)
    return out


def run_phase0c(distances: pd.DataFrame) -> dict:
    diagnostics = {"overall": {}, "by_split": {}, "dominant_modality": {}}
    raw_distance_cols = ["d_financial", "d_mda", "d_auditor"]
    normed = distances.copy()
    for col in raw_distance_cols:
        val = distances[distances["split"] == "val"][col]
        normed[f"{col}_norm"] = (distances[col] - val.mean()) / (val.std(ddof=0) + 1e-8)
    norm_cols = [f"{c}_norm" for c in raw_distance_cols]
    normed["disagreement_std"] = normed[norm_cols].std(axis=1)
    normed["disagreement_max_minus_min"] = normed[norm_cols].max(axis=1) - normed[norm_cols].min(axis=1)
    normed["dominant_margin_norm"] = normed[norm_cols].apply(
        lambda row: float(np.sort(row.to_numpy())[-1] - np.sort(row.to_numpy())[-2]), axis=1
    )
    normed["pred_threshold_0_5"] = (normed["d_max"] > 0.5).astype(int)
    normed["is_error_threshold_0_5"] = (
        normed["pred_threshold_0_5"] != normed["label"].astype(int)
    ).astype(int)
    normed["error_type_threshold_0_5"] = np.select(
        [
            (normed["label"] == 1) & (normed["pred_threshold_0_5"] == 1),
            (normed["label"] == 0) & (normed["pred_threshold_0_5"] == 1),
            (normed["label"] == 1) & (normed["pred_threshold_0_5"] == 0),
        ],
        ["TP", "FP", "FN"],
        default="TN",
    )
    normed.to_csv(OUTPUT_DIR / "active_finbert_mabad_valtest_diagnostics.csv", index=False)

    for col in raw_distance_cols + ["d_max", "disagreement_std", "disagreement_max_minus_min", "dominant_margin_norm"]:
        diagnostics["overall"][col] = metric_block(normed["label"].to_numpy(), normed[col].to_numpy())
        diagnostics["overall"][col]["mean_diff_pos_minus_neg"] = mean_diff_pos_minus_neg(
            normed["label"].to_numpy(), normed[col].to_numpy()
        )

    for split, group in normed.groupby("split"):
        diagnostics["by_split"][split] = {}
        for col in ["d_max", "disagreement_std", "disagreement_max_minus_min"]:
            diagnostics["by_split"][split][col] = metric_block(
                group["label"].to_numpy(), group[col].to_numpy()
            )

    dom = normed.groupby("dominant_modality")["label"].agg(["count", "sum", "mean"]).reset_index()
    diagnostics["dominant_modality"] = dom.to_dict(orient="records")

    err = (
        normed.groupby(["split", "error_type_threshold_0_5"])
        .agg(
            count=("uid", "count"),
            disagreement_std_mean=("disagreement_std", "mean"),
            disagreement_range_mean=("disagreement_max_minus_min", "mean"),
            financial_missing_pct_mean=("financial_missing_pct", "mean"),
            mda_chars_median=("mda_char_count", "median"),
            auditor_chars_median=("auditor_char_count", "median"),
        )
        .reset_index()
    )
    diagnostics["threshold_0_5_error_groups"] = err.to_dict(orient="records")
    diagnostics["error_prediction_by_disagreement"] = {
        "disagreement_std": metric_block(
            normed["is_error_threshold_0_5"].to_numpy(),
            normed["disagreement_std"].to_numpy(),
        ),
        "disagreement_max_minus_min": metric_block(
            normed["is_error_threshold_0_5"].to_numpy(),
            normed["disagreement_max_minus_min"].to_numpy(),
        ),
    }

    dominant_completeness = (
        normed.groupby("dominant_modality")
        .agg(
            count=("uid", "count"),
            positive_rate=("label", "mean"),
            financial_nonnull_mean=("financial_nonnull_count", "mean"),
            financial_missing_pct_mean=("financial_missing_pct", "mean"),
            mda_chars_median=("mda_char_count", "median"),
            auditor_chars_median=("auditor_char_count", "median"),
        )
        .reset_index()
    )
    diagnostics["dominant_modality_completeness"] = dominant_completeness.to_dict(
        orient="records"
    )
    (OUTPUT_DIR / "phase0c_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    write_phase0_decision(diagnostics)
    return diagnostics


def write_phase0_decision(diagnostics: dict) -> None:
    overall = diagnostics["overall"]
    d_max = overall["d_max"]
    d_mda = overall["d_mda"]
    disagreement = overall["disagreement_std"]
    lines = [
        "# Phase 0 Decision",
        "",
        "Decision: D. Useful diagnostic signal, but insufficient evidence yet for a final new scorer.",
        "",
        "Rationale:",
        "",
        f"- Saved-model inference recovered modality distances without retraining. `d_max` gives AP {d_max['ap']:.4f}, AUC {d_max['auc']:.4f}, Recall@50 {d_max['recall_at_50']:.4f}, and Recall@100 {d_max['recall_at_100']:.4f} on validation+test.",
        f"- The MD&A distance is the strongest individual modality distance: AP {d_mda['ap']:.4f}, AUC {d_mda['auc']:.4f}.",
        f"- Disagreement is a real diagnostic signal but weaker than the core distances: disagreement standard deviation gives AP {disagreement['ap']:.4f}, AUC {disagreement['auc']:.4f}.",
        "- Completeness/missingness alone is not a useful bankruptcy ranking signal, although financial completeness differs slightly between bankrupt and non-bankrupt filings.",
        "- The next step should be a validation-only post-hoc reliability experiment, not a new architecture.",
        "",
        "Recommended next experiment:",
        "",
        "Fit a tiny validation-only post-hoc scorer using only transparent features: `d_financial`, `d_mda`, `d_auditor`, `d_max`, normalized disagreement range, financial missing percentage, MD&A length, auditor length, and dominant modality. Evaluate once on test against the historical MABAD/XGBoost rows.",
    ]
    (OUTPUT_DIR / "phase0_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["0a", "0b", "0c", "all"], default="all")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    completeness_path = OUTPUT_DIR / "completeness_by_filing.csv"
    if args.phase in ("0a", "all"):
        completeness = run_phase0a()
    else:
        completeness = pd.read_csv(completeness_path, parse_dates=["date"])

    if args.phase in ("0b", "all"):
        distances = run_phase0b(completeness)
    elif args.phase == "0c":
        distances = pd.read_csv(
            OUTPUT_DIR / "active_finbert_mabad_valtest_modality_distances.csv",
            parse_dates=["date"],
        )
    else:
        distances = None

    if args.phase in ("0c", "all") and distances is not None:
        run_phase0c(distances)


if __name__ == "__main__":
    main()
