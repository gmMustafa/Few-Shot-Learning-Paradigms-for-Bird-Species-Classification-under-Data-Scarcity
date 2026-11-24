"""
Data loader for FSL thesis dashboard.
Loads and normalises all result data from the thesis experiment outputs.
"""

import os
import json
import glob
import pandas as pd
import numpy as np
import streamlit as st

# ── Path resolution ──────────────────────────────────────────────────────────
_LOADER_DIR    = os.path.dirname(os.path.abspath(__file__))         # dashboard/data/
_DASHBOARD_DIR = os.path.dirname(_LOADER_DIR)                       # dashboard/
_THESIS_DIR    = os.path.dirname(_DASHBOARD_DIR)                    # repository root
_RESULTS_ROOT = os.path.join(_THESIS_DIR, "results")
_MASTER_CSV   = os.path.join(_RESULTS_ROOT, "all_results_master.csv")

# ── Canonical method ordering for consistent chart grouping ──────────────────
METHOD_ORDER = [
    "NCC (R50)",
    "NCC (DeiT-S)",
    "ProtoNet",
    "ProtoNet (DeiT-S)",
    "ProtoNet+TTA",
    "MatchingNet",
    "MatchingNet+FCE",
    "RelationNet",
]

BACKBONE_COLORS = {"ResNet-50": "#003865", "DeiT-S": "#FF6B35"}
BACKBONE_COLOR  = BACKBONE_COLORS   # alias used by pages
METHOD_COLORS   = {
    "NCC (R50)":        "#003865",
    "NCC (DeiT-S)":     "#FF6B35",
    "ProtoNet":         "#1e6bb8",
    "ProtoNet (DeiT-S)":"#e85d20",
    "ProtoNet+TTA":     "#0a9396",
    "MatchingNet":      "#94d2bd",
    "MatchingNet+FCE":  "#e9d8a6",
    "RelationNet":      "#ae2012",
}

SUPERVISED_METHODS = {"Supervised (full)", "Supervised (base)", "Zero-Shot", "5-shot Fine-tune"}


# ── Master results ────────────────────────────────────────────────────────────
@st.cache_data
def load_master() -> pd.DataFrame:
    """Load and normalise the master results CSV into a clean DataFrame."""
    df = pd.read_csv(_MASTER_CSV, dtype=str)

    # Normalise column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    # Rename to stable names
    rename = {
        "acc_": "accuracy",
        "acc": "accuracy",
        "_ci95": "ci",
        "ci95": "ci",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "auc_ovr": "auc",
        "k_shot": "k_shot",
        "episodic": "episodic",
    }
    # Map the messy auto-generated names
    col_map = {}
    for c in df.columns:
        if "acc" in c and "%" in c.replace("_", " "):
            col_map[c] = "accuracy"
        elif c == "acc__":
            col_map[c] = "accuracy"
        elif "ci95" in c or c == "_ci95":
            col_map[c] = "ci"
        elif c == "auc_ovr":
            col_map[c] = "auc"
    df = df.rename(columns=col_map)

    # Re-do with explicit column list based on known CSV header
    # Dataset,Split,Method,Backbone,Distance,K-Shot,Acc (%),+/-CI95,F1,Precision,Recall,AUC-OVR,Episodic
    df.columns = [c.strip() for c in pd.read_csv(_MASTER_CSV, nrows=0).columns]
    df = pd.read_csv(_MASTER_CSV)

    df = df.rename(columns={
        "Acc (%)":  "accuracy",
        "+/-CI95":  "ci",
        "F1":       "f1",
        "Precision":"precision",
        "Recall":   "recall",
        "AUC-OVR":  "auc",
        "K-Shot":   "k_shot",
        "Episodic": "episodic",
        "Dataset":  "dataset",
        "Split":    "split",
        "Method":   "method",
        "Backbone": "backbone",
        "Distance": "distance",
    })

    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    df["ci"]       = pd.to_numeric(df["ci"],       errors="coerce")
    df["f1"]       = pd.to_numeric(df["f1"],       errors="coerce")
    df["precision"]= pd.to_numeric(df["precision"],errors="coerce")
    df["recall"]   = pd.to_numeric(df["recall"],   errors="coerce")
    df["auc"]      = pd.to_numeric(df["auc"],      errors="coerce")
    df["k_shot"]   = pd.to_numeric(df["k_shot"],   errors="coerce")
    df["episodic"] = df["episodic"].astype(str).str.lower().map(
        {"true": True, "false": False}
    ).fillna(False)

    # Canonicalise distance names
    df["distance"] = df["distance"].fillna("--").str.strip()

    # Add helper columns
    df["method_label"] = df["method"]
    df["k_shot_int"] = df["k_shot"].dropna().astype(float).apply(
        lambda x: int(x) if not np.isnan(x) else x
    )
    df["k_shot_int"] = pd.to_numeric(df["k_shot_int"], errors="coerce")

    # Wilson CI bounds
    df["ci_low"]  = (df["accuracy"] - df["ci"]).clip(lower=0)
    df["ci_high"] = (df["accuracy"] + df["ci"]).clip(upper=100)

    return df


@st.cache_data
def get_fsl_results() -> pd.DataFrame:
    """Return only FSL rows (exclude supervised/zero-shot references) with valid accuracy."""
    df = load_master()
    mask = (
        ~df["method"].isin(SUPERVISED_METHODS)
        & df["k_shot"].notna()
        & df["accuracy"].notna()
        # Exclude the Zero-Shot and Fine-tune marker rows that misuse K-Shot column
        & ~(df["method"].isin(["Zero-Shot", "5-shot Fine-tune"]))
    )
    return df[mask].copy()


def filter_results(
    df: pd.DataFrame = None,
    method=None,
    backbone=None,
    split=None,
    shot=None,
    dataset=None,
    distance=None,
) -> pd.DataFrame:
    """Filter a DataFrame (defaults to master FSL results) by any combination of fields."""
    if df is None:
        df = get_fsl_results()
    if method is not None:
        methods = [method] if isinstance(method, str) else method
        df = df[df["method"].isin(methods)]
    if backbone is not None:
        df = df[df["backbone"] == backbone]
    if split is not None:
        df = df[df["split"] == split]
    if shot is not None:
        df = df[df["k_shot"] == float(shot)]
    if dataset is not None:
        df = df[df["dataset"] == dataset]
    if distance is not None and distance != "All":
        df = df[df["distance"] == distance]
    return df


def get_methods(df=None):
    if df is None:
        df = get_fsl_results()
    present = set(df["method"].unique())
    return [m for m in METHOD_ORDER if m in present]


def get_splits(dataset="Birds-20"):
    df = get_fsl_results()
    return sorted(df[df["dataset"] == dataset]["split"].unique().tolist())


def get_shots():
    df = get_fsl_results()
    shots = sorted(df["k_shot"].dropna().unique())
    return [int(s) for s in shots if s in (1, 3, 5)]


# ── Per-class accuracy (from individual summary JSON files) ──────────────────
@st.cache_data
def load_perclass_data() -> pd.DataFrame:
    """
    Recursively scan all summary.json files and extract per-class accuracy.
    Returns a DataFrame with columns:
      [dataset, split, backbone, method, distance, k_shot, species, accuracy]
    """
    records = []
    for root, _dirs, files in os.walk(_RESULTS_ROOT):
        if "summary.json" not in files:
            continue
        path = os.path.join(root, "summary.json")
        try:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue

        per_class = d.get("per_class_accuracy_pct")
        if not per_class or not isinstance(per_class, dict):
            continue

        # Infer context from JSON fields and path
        norm_path = path.replace("\\", "/").lower()
        backbone  = "DeiT-S" if "/vit/" in norm_path else "ResNet-50"
        dataset   = "CUB-200" if "/cub/" in norm_path else "Birds-20"
        k_shot    = d.get("k_shot")
        distance  = (d.get("distance") or "euclidean").lower()
        distance  = {"euclidean": "Euclidean", "manhattan": "Manhattan",
                     "cosine": "Cosine"}.get(distance, distance.capitalize())

        # Split inference — check longer prefixes FIRST to avoid substring collisions
        # (e.g. "base15" is a prefix of "base150")
        if "base150" in norm_path:
            split = "B150/N50"
        elif "base100" in norm_path:
            split = "B100/N50"
        elif "base15" in norm_path:
            split = "B15/N5"
        elif "base10" in norm_path:
            split = "B10/N10"
        else:
            split = "Unknown"

        # Method inference
        if "matching_net_fce" in norm_path:
            method = "MatchingNet+FCE"
        elif "matching_net" in norm_path and "fce" not in norm_path:
            method = "MatchingNet"
        elif "relation_net" in norm_path:
            method = "RelationNet"
        elif "proto_net" in norm_path or "protonet" in norm_path:
            tta = "tta" in norm_path
            if backbone == "DeiT-S":
                method = "ProtoNet (DeiT-S)"
            elif tta:
                method = "ProtoNet+TTA"
            else:
                method = "ProtoNet"
        elif "ncc" in norm_path or ("baseline" in norm_path and "novel_ncc" in norm_path):
            method = "NCC (DeiT-S)" if backbone == "DeiT-S" else "NCC (R50)"
        else:
            continue  # unknown — skip

        for species, acc in per_class.items():
            records.append({
                "dataset":  dataset,
                "split":    split,
                "backbone": backbone,
                "method":   method,
                "distance": distance,
                "k_shot":   int(k_shot) if k_shot is not None else None,
                "species":  species.replace("_", " ").title(),
                "accuracy": float(acc),
            })

    if not records:
        return pd.DataFrame(columns=["dataset","split","backbone","method",
                                     "distance","k_shot","species","accuracy"])
    return pd.DataFrame(records)


@st.cache_data
def get_perclass_birds20(split: str = "B15/N5", shot: int = 5,
                         distance: str = "Euclidean") -> pd.DataFrame:
    """Return per-class accuracy for Birds-20 novel species in wide (pivot) form."""
    df = load_perclass_data()
    sub = df[
        (df["dataset"] == "Birds-20") &
        (df["split"] == split) &
        (df["k_shot"] == shot) &
        (df["distance"] == distance)
    ]
    if sub.empty:
        return pd.DataFrame()
    pivot = sub.pivot_table(
        index="method", columns="species", values="accuracy", aggfunc="mean"
    )
    # Order rows by METHOD_ORDER
    present = [m for m in METHOD_ORDER if m in pivot.index]
    return pivot.loc[present] if present else pivot


@st.cache_data
def get_perclass_cub(split: str = "B100/N50", shot: int = 5,
                     distance: str = "Euclidean", top_n: int = 10) -> pd.DataFrame:
    """Return per-class accuracy for CUB novel species (top N hardest + easiest)."""
    df = load_perclass_data()
    sub = df[
        (df["dataset"] == "CUB-200") &
        (df["split"] == split) &
        (df["k_shot"] == shot) &
        (df["distance"] == distance)
    ]
    if sub.empty:
        return pd.DataFrame()

    mean_per_species = sub.groupby("species")["accuracy"].mean().sort_values()
    hardest = mean_per_species.head(top_n).index.tolist()
    easiest = mean_per_species.tail(top_n).index.tolist()
    keep = hardest + [s for s in easiest if s not in hardest]

    sub2 = sub[sub["species"].isin(keep)]
    pivot = sub2.pivot_table(
        index="method", columns="species", values="accuracy", aggfunc="mean"
    )
    # Sort columns: hardest first
    cols = [s for s in hardest + easiest if s in pivot.columns]
    return pivot[cols]
