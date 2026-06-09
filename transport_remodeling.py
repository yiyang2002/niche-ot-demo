"""
Optimal-transport remodeling model for healthy-to-disease niche libraries.

This module contains helper functions for the exploratory transport notebook.
It consumes existing niche-library cluster-level metric tables and fits an
unbalanced source-linked transport model:

    healthy niche clusters -> disease niche clusters

The model is intended as a downstream interpretability layer. It does not claim
causal or temporal transition; it decomposes disease niche structure relative to
a healthy reference library.

The notebook should define the organ-specific canonical cell-type map and pass
that map into `load_cluster_metrics(...)`. This file intentionally does not
import `utils_plot.py` or own the analysis run.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import linprog


DEFAULT_BRANCH_MIN_TOTAL_MASS = 0.01
DEFAULT_BRANCH_MIN_SOURCE_FRACTION = 0.10


@dataclass
class TransportOptions:
    """Notebook-facing option container used by the helper functions."""

    mass_mode: str = "n_cells"
    healthy_h5ad: Optional[str] = None
    disease_h5ad: Optional[str] = None
    cluster_key: Optional[str] = None
    healthy_cluster_key: Optional[str] = None
    disease_cluster_key: Optional[str] = None
    slide_key: str = "slide"
    lambda_expand: float = 0.35
    lambda_reduce: float = 0.35
    lambda_residual: float = 0.80
    branch_min_total_mass: float = DEFAULT_BRANCH_MIN_TOTAL_MASS
    branch_min_source_fraction: float = DEFAULT_BRANCH_MIN_SOURCE_FRACTION
    unknown_fraction_threshold: float = 0.30
    pseudocount_mass: float = 1e-6
    plan_mass_tol: float = 1e-10
    mass_zero_tol: float = 1e-9
    top_n_heatmap: int = 25


@dataclass
class TransportSolution:
    transport: np.ndarray
    expansion: np.ndarray
    reduction: np.ndarray
    residual: np.ndarray
    objective: float
    solver_status: str
    solver_message: str


def get_option(options: object, name: str, default: object = None) -> object:
    """Read a parameter from a dataclass, SimpleNamespace, dict, or plain object."""

    if isinstance(options, Mapping):
        return options.get(name, default)
    return getattr(options, name, default)


def canonicalize_cell_name(cell_name: object, canonical_map: Optional[Mapping[str, str]] = None) -> str:
    """Map one cell-type label to a notebook-specified canonical label.

    The map is intentionally supplied by the notebook so the user controls the
    biological vocabulary. Matching is exact after stripping whitespace, with a
    case-insensitive fallback for convenience. Labels absent from the map are
    retained unchanged.
    """

    label = str(cell_name).strip()
    if not canonical_map:
        return label
    lookup = {str(k).strip(): str(v).strip() for k, v in canonical_map.items()}
    if label in lookup:
        return lookup[label]
    lower_lookup = {k.lower(): v for k, v in lookup.items()}
    return lower_lookup.get(label.lower(), label)


def canonicalize_comp_col(column: str, canonical_map: Optional[Mapping[str, str]] = None) -> str:
    """Canonicalize a `Comp_*` column name using a notebook-defined cell map."""

    if not str(column).startswith("Comp_"):
        return str(column)
    raw_cell = str(column).replace("Comp_", "", 1)
    return f"Comp_{canonicalize_cell_name(raw_cell, canonical_map)}"


def harmonize_comp_columns(
    df: pd.DataFrame,
    canonical_map: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """Rename and merge `Comp_*` columns according to a supplied canonical map.

    If multiple raw cell labels map to the same canonical cell label, their
    composition columns are summed. Non-composition columns are preserved.
    """

    out = df.loc[:, [c for c in df.columns if not str(c).startswith("Comp_")]].copy()
    comp_cols = [c for c in df.columns if str(c).startswith("Comp_")]
    for col in comp_cols:
        canonical_col = canonicalize_comp_col(col, canonical_map)
        values = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        if canonical_col in out.columns:
            out[canonical_col] = pd.to_numeric(out[canonical_col], errors="coerce").fillna(0.0) + values
        else:
            out[canonical_col] = values
    return out


def replace_cell_names_in_text(text: object, canonical_map: Optional[Mapping[str, str]] = None) -> object:
    """Replace raw cell labels in a readable text field using the same map."""

    if not isinstance(text, str) or not canonical_map:
        return text
    updated = text
    pairs = sorted(
        ((str(k).strip(), str(v).strip()) for k, v in canonical_map.items()),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for raw, canonical in pairs:
        if not raw or raw == canonical:
            continue
        updated = re.sub(re.escape(raw), canonical, updated, flags=re.IGNORECASE)
    return updated


def harmonize_text_fields(
    df: pd.DataFrame,
    canonical_map: Optional[Mapping[str, str]] = None,
    fields: Sequence[str] = ("Top_Enriched_Cells (log2FC)", "Niche_Signature"),
) -> pd.DataFrame:
    """Canonicalize optional descriptive text columns for output readability."""

    out = df.copy()
    for field in fields:
        if field in out.columns:
            out[field] = out[field].map(lambda x: replace_cell_names_in_text(x, canonical_map))
    return out


def load_cluster_metrics(
    path: Union[str, Path],
    canonical_map: Optional[Mapping[str, str]] = None,
    harmonize_text: bool = True,
) -> pd.DataFrame:
    """Read one `cluster_level_metrics.csv` and apply explicit harmonization."""

    df = pd.read_csv(path)
    df = harmonize_comp_columns(df, canonical_map=canonical_map)
    if harmonize_text:
        df = harmonize_text_fields(df, canonical_map=canonical_map)
    if "cluster_id" not in df.columns:
        raise ValueError(f"{path} is missing required column 'cluster_id'.")
    if "N_Cells" not in df.columns:
        raise ValueError(f"{path} is missing required column 'N_Cells'.")

    df = df.copy()
    df["cluster_id"] = df["cluster_id"].astype(str)
    df["N_Cells"] = pd.to_numeric(df["N_Cells"], errors="coerce").fillna(0.0)
    if "N_Slides" in df.columns:
        df["N_Slides"] = pd.to_numeric(df["N_Slides"], errors="coerce").fillna(0.0)
    else:
        df["N_Slides"] = np.nan

    if not any(str(c).startswith("Comp_") for c in df.columns):
        raise ValueError(f"{path} has no Comp_* columns.")
    return df


def read_cluster_metrics(
    path: Union[str, Path],
    canonical_map: Optional[Mapping[str, str]] = None,
    harmonize_text: bool = True,
) -> pd.DataFrame:
    """Backward-compatible alias for `load_cluster_metrics`."""

    return load_cluster_metrics(path, canonical_map=canonical_map, harmonize_text=harmonize_text)


def aligned_composition_matrices(
    healthy_df: pd.DataFrame,
    disease_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    comp_cols = sorted(
        set(c for c in healthy_df.columns if str(c).startswith("Comp_"))
        | set(c for c in disease_df.columns if str(c).startswith("Comp_"))
    )
    healthy_comp = healthy_df.reindex(columns=comp_cols, fill_value=0.0).to_numpy(dtype=float)
    disease_comp = disease_df.reindex(columns=comp_cols, fill_value=0.0).to_numpy(dtype=float)
    return row_normalize(healthy_comp), row_normalize(disease_comp), comp_cols


def row_normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=float)
    sums = mat.sum(axis=1, keepdims=True)
    sums[sums <= 0] = 1.0
    return mat / sums


def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=float)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms <= 0] = 1.0
    return mat / norms


def cosine_distance_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = l2_normalize_rows(left)
    right_norm = l2_normalize_rows(right)
    sim = left_norm @ right_norm.T
    sim = np.clip(sim, -1.0, 1.0)
    return np.clip(1.0 - sim, 0.0, 2.0)


def scale_cost_matrix(cost: np.ndarray) -> Tuple[np.ndarray, float]:
    finite = cost[np.isfinite(cost)]
    positive = finite[finite > 0]
    if positive.size == 0:
        return np.zeros_like(cost, dtype=float), 1.0
    scale = float(np.percentile(positive, 95))
    if scale <= 0:
        scale = float(positive.max())
    if scale <= 0:
        return np.zeros_like(cost, dtype=float), 1.0
    return np.clip(cost / scale, 0.0, 1.0), scale


def compute_n_cells_mass(df: pd.DataFrame, label: str) -> np.ndarray:
    counts = pd.to_numeric(df["N_Cells"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    total = counts.sum()
    if total <= 0:
        raise ValueError(f"{label} N_Cells sum is zero; cannot compute masses.")
    return counts / total


def read_h5ad(path: str):
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("Reading h5ad inputs requires anndata.") from exc
    return ad.read_h5ad(path)


def compute_slide_mean_mass(
    h5ad_path: str,
    cluster_ids: Sequence[str],
    cluster_key: str,
    slide_key: str,
    label: str,
) -> np.ndarray:
    adata = read_h5ad(h5ad_path)
    for key in [cluster_key, slide_key]:
        if key not in adata.obs:
            raise ValueError(f"{label} h5ad is missing obs[{key!r}].")

    obs = adata.obs.loc[:, [cluster_key, slide_key]].copy()
    obs[cluster_key] = obs[cluster_key].astype(str)
    obs[slide_key] = obs[slide_key].astype(str)
    counts = pd.crosstab(obs[slide_key], obs[cluster_key])
    fractions = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    fractions = fractions.reindex(columns=list(cluster_ids), fill_value=0.0)
    mass = fractions.mean(axis=0).to_numpy(dtype=float)
    total = mass.sum()
    if total <= 0:
        raise ValueError(f"{label} slide-mean cluster fractions sum to zero.")
    return mass / total


def compute_masses(
    healthy_df: pd.DataFrame,
    disease_df: pd.DataFrame,
    options: object,
) -> Tuple[np.ndarray, np.ndarray]:
    mass_mode = str(get_option(options, "mass_mode", "n_cells"))
    if mass_mode == "n_cells":
        return compute_n_cells_mass(healthy_df, "healthy"), compute_n_cells_mass(disease_df, "disease")

    healthy_h5ad = get_option(options, "healthy_h5ad")
    disease_h5ad = get_option(options, "disease_h5ad")
    shared_cluster_key = get_option(options, "cluster_key", None)
    healthy_cluster_key = str(get_option(options, "healthy_cluster_key", None) or shared_cluster_key or "leiden")
    disease_cluster_key = str(get_option(options, "disease_cluster_key", None) or shared_cluster_key or "leiden")
    slide_key = str(get_option(options, "slide_key", "slide"))

    if mass_mode != "slide_mean_fraction":
        raise ValueError(f"Unsupported mass_mode={mass_mode!r}. Use 'n_cells' or 'slide_mean_fraction'.")
    if not healthy_h5ad or not disease_h5ad:
        raise ValueError("mass_mode='slide_mean_fraction' requires healthy_h5ad and disease_h5ad.")

    healthy_ids = healthy_df["cluster_id"].astype(str).tolist()
    disease_ids = disease_df["cluster_id"].astype(str).tolist()
    return (
        compute_slide_mean_mass(str(healthy_h5ad), healthy_ids, healthy_cluster_key, slide_key, "healthy"),
        compute_slide_mean_mass(str(disease_h5ad), disease_ids, disease_cluster_key, slide_key, "disease"),
    )


def build_cost_matrices(
    healthy_comp: np.ndarray,
    disease_comp: np.ndarray,
    healthy_df: pd.DataFrame,
    disease_df: pd.DataFrame,
    options: object,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, object]]:
    comp_cost_raw = cosine_distance_matrix(healthy_comp, disease_comp)
    comp_cost_scaled, comp_scale = scale_cost_matrix(comp_cost_raw)
    total_cost = comp_cost_scaled

    healthy_ids = healthy_df["cluster_id"].astype(str).tolist()
    disease_ids = disease_df["cluster_id"].astype(str).tolist()
    records = []
    for i, h in enumerate(healthy_ids):
        for j, d in enumerate(disease_ids):
            records.append(
                {
                    "healthy_cluster_id": h,
                    "disease_cluster_id": d,
                    "composition_cost_raw": comp_cost_raw[i, j],
                    "composition_cost_scaled": comp_cost_scaled[i, j],
                    "total_cost": total_cost[i, j],
                    "composition_similarity": 1.0 - comp_cost_raw[i, j],
                }
            )
    cost_long = pd.DataFrame.from_records(records)
    metadata = {
        "composition_cost_scale_p95": comp_scale,
        "cost_definition": "scaled_composition_cosine_distance",
    }
    return total_cost, cost_long, metadata


def solve_unbalanced_transport(
    cost: np.ndarray,
    healthy_mass: np.ndarray,
    disease_mass: np.ndarray,
    lambda_expand: float,
    lambda_reduce: float,
    lambda_residual: float,
) -> TransportSolution:
    n_h, n_d = cost.shape
    n_t = n_h * n_d
    e_offset = n_t
    r_offset = e_offset + n_h
    u_offset = r_offset + n_h
    n_vars = u_offset + n_d

    c = np.concatenate(
        [
            cost.reshape(-1),
            np.full(n_h, lambda_expand, dtype=float),
            np.full(n_h, lambda_reduce, dtype=float),
            np.full(n_d, lambda_residual, dtype=float),
        ]
    )

    a_eq = np.zeros((n_d + n_h, n_vars), dtype=float)
    b_eq = np.concatenate([disease_mass, healthy_mass])

    for j in range(n_d):
        row = j
        for i in range(n_h):
            a_eq[row, i * n_d + j] = 1.0
        a_eq[row, u_offset + j] = 1.0

    for i in range(n_h):
        row = n_d + i
        for j in range(n_d):
            a_eq[row, i * n_d + j] = 1.0
        a_eq[row, e_offset + i] = -1.0
        a_eq[row, r_offset + i] = 1.0

    result = linprog(
        c,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Transport LP failed: {result.message}")

    x = result.x
    return TransportSolution(
        transport=x[:n_t].reshape(n_h, n_d),
        expansion=x[e_offset:r_offset],
        reduction=x[r_offset:u_offset],
        residual=x[u_offset:],
        objective=float(result.fun),
        solver_status=str(result.status),
        solver_message=str(result.message),
    )


def objective_components(
    solution: TransportSolution,
    cost: np.ndarray,
    lambda_expand: float,
    lambda_reduce: float,
    lambda_residual: float,
) -> Dict[str, float]:
    """Break the solved objective into interpretable cost and slack terms."""

    transport_cost = float((solution.transport * cost).sum())
    expansion_penalty = float(lambda_expand * solution.expansion.sum())
    reduction_penalty = float(lambda_reduce * solution.reduction.sum())
    residual_penalty = float(lambda_residual * solution.residual.sum())
    return {
        "transport_cost": transport_cost,
        "expansion_mass": float(solution.expansion.sum()),
        "reduction_mass": float(solution.reduction.sum()),
        "residual_mass": float(solution.residual.sum()),
        "expansion_penalty": expansion_penalty,
        "reduction_penalty": reduction_penalty,
        "residual_penalty": residual_penalty,
        "objective_recomputed": transport_cost + expansion_penalty + reduction_penalty + residual_penalty,
        "objective_solver": float(solution.objective),
    }


def shannon_entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    total = values.sum()
    if total <= 0:
        return 0.0
    probs = values[values > 0] / total
    return float(-(probs * np.log(probs)).sum())


def effective_count_from_entropy(entropy: float) -> float:
    return float(math.exp(entropy))


def log2_shift(numerator: float, denominator: float, pseudocount: float) -> float:
    return float(math.log2((numerator + pseudocount) / (denominator + pseudocount)))


def rank_desc(values: pd.Series) -> pd.Series:
    return values.rank(ascending=False, method="min", na_option="bottom").astype(int)


def interpret_shift(value: float, up_thr: float = 0.75, down_thr: float = -0.75) -> str:
    if value >= up_thr:
        return "Expanded in disease"
    if value <= down_thr:
        return "Reduced in disease"
    return "Roughly shared"


def summarize_current_best_match(
    cost_long: pd.DataFrame,
    healthy_df: pd.DataFrame,
    disease_df: pd.DataFrame,
    healthy_mass: np.ndarray,
    disease_mass: np.ndarray,
    pseudocount: float,
) -> pd.DataFrame:
    best_idx = cost_long.sort_values(
        ["disease_cluster_id", "composition_cost_raw", "healthy_cluster_id"]
    ).groupby("disease_cluster_id", as_index=False).head(1)
    disease_mass_map = dict(zip(disease_df["cluster_id"].astype(str), disease_mass))
    best_idx["disease_mass"] = best_idx["disease_cluster_id"].map(disease_mass_map).astype(float)
    family_mass = (
        best_idx.groupby("healthy_cluster_id", as_index=False)
        .agg(
            current_disease_family_mass=("disease_mass", "sum"),
            current_n_disease_clusters=("disease_cluster_id", "nunique"),
            current_mean_best_match_similarity=("composition_similarity", "mean"),
            current_top_disease_clusters=(
                "disease_cluster_id",
                lambda x: "|".join(map(str, x.astype(str).tolist())),
            ),
        )
    )
    healthy = pd.DataFrame(
        {
            "healthy_cluster_id": healthy_df["cluster_id"].astype(str),
            "healthy_mass": healthy_mass,
        }
    )
    out = healthy.merge(family_mass, how="left", on="healthy_cluster_id")
    out["current_disease_family_mass"] = out["current_disease_family_mass"].fillna(0.0)
    out["current_n_disease_clusters"] = out["current_n_disease_clusters"].fillna(0).astype(int)
    out["current_log2_abundance_shift"] = [
        log2_shift(d, h, pseudocount)
        for d, h in zip(out["current_disease_family_mass"], out["healthy_mass"])
    ]
    out["current_abundance_interpretation"] = out["current_log2_abundance_shift"].map(interpret_shift)
    out["current_expansion_rank"] = rank_desc(out["current_log2_abundance_shift"])
    return out


def source_metadata(row: pd.Series, prefix: str) -> Dict[str, object]:
    return {
        f"{prefix}_N_Cells": row.get("N_Cells", np.nan),
        f"{prefix}_N_Slides": row.get("N_Slides", np.nan),
        f"{prefix}_signature": row.get("Niche_Signature", ""),
        f"{prefix}_top_enriched": row.get("Top_Enriched_Cells (log2FC)", ""),
    }


def summarize_sources(
    solution: TransportSolution,
    total_cost: np.ndarray,
    healthy_df: pd.DataFrame,
    disease_df: pd.DataFrame,
    healthy_mass: np.ndarray,
    options: object,
    current_summary: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    rows = []
    healthy_ids = healthy_df["cluster_id"].astype(str).tolist()
    disease_ids = disease_df["cluster_id"].astype(str).tolist()
    row_sum = solution.transport.sum(axis=1)
    branch_min_total_mass = float(get_option(options, "branch_min_total_mass", DEFAULT_BRANCH_MIN_TOTAL_MASS))
    branch_min_source_fraction = float(
        get_option(options, "branch_min_source_fraction", DEFAULT_BRANCH_MIN_SOURCE_FRACTION)
    )
    pseudocount_mass = float(get_option(options, "pseudocount_mass", 1e-6))
    plan_mass_tol = float(get_option(options, "plan_mass_tol", 1e-10))
    mass_zero_tol = float(get_option(options, "mass_zero_tol", 1e-9))

    for i, h_id in enumerate(healthy_ids):
        transport_row = solution.transport[i, :]
        transported_mass = float(row_sum[i])
        net_shift_mass = transported_mass - float(healthy_mass[i])
        if abs(net_shift_mass) <= mass_zero_tol:
            net_shift_mass = 0.0
        expansion_mass = float(solution.expansion[i])
        reduction_mass = float(solution.reduction[i])
        if expansion_mass <= mass_zero_tol:
            expansion_mass = 0.0
        if reduction_mass <= mass_zero_tol:
            reduction_mass = 0.0
        mean_cost = (
            float((transport_row * total_cost[i, :]).sum() / transported_mass)
            if transported_mass > 0
            else np.nan
        )
        branch_threshold = max(
            branch_min_total_mass,
            branch_min_source_fraction * transported_mass,
        )
        branch_mask = transport_row >= branch_threshold if transported_mass > 0 else np.zeros_like(transport_row, dtype=bool)
        entropy = shannon_entropy(transport_row)
        top_targets = []
        for j in np.argsort(-transport_row):
            if transport_row[j] <= plan_mass_tol:
                continue
            top_targets.append(f"{disease_ids[j]}:{transport_row[j]:.6g}:cost={total_cost[i, j]:.4f}")
            if len(top_targets) >= 5:
                break

        row = {
            "healthy_cluster_id": h_id,
            "healthy_mass": float(healthy_mass[i]),
            "transported_out_mass": transported_mass,
            "net_shift_mass": net_shift_mass,
            "log2_transport_shift": log2_shift(transported_mass, float(healthy_mass[i]), pseudocount_mass),
            "transport_shift_interpretation": interpret_shift(
                log2_shift(transported_mass, float(healthy_mass[i]), pseudocount_mass)
            ),
            "expansion_mass": expansion_mass,
            "reduction_mass": reduction_mass,
            "mean_transport_cost": mean_cost,
            "cost_weighted_mass": float((transport_row * total_cost[i, :]).sum()),
            "n_nonzero_disease_targets": int((transport_row > plan_mass_tol).sum()),
            "n_disease_branches": int(branch_mask.sum()),
            "split_entropy": entropy,
            "effective_n_branches": effective_count_from_entropy(entropy),
            "top_disease_targets": "|".join(top_targets),
        }
        row.update(source_metadata(healthy_df.iloc[i], "healthy"))
        rows.append(row)

    summary = pd.DataFrame(rows)
    if current_summary is not None:
        current_cols = [
            "healthy_cluster_id",
            "current_disease_family_mass",
            "current_log2_abundance_shift",
            "current_abundance_interpretation",
            "current_expansion_rank",
            "current_n_disease_clusters",
            "current_mean_best_match_similarity",
            "current_top_disease_clusters",
        ]
        available_current_cols = [c for c in current_cols if c in current_summary.columns]
        summary = summary.merge(current_summary[available_current_cols], how="left", on="healthy_cluster_id")
    summary["transported_mass_rank"] = rank_desc(summary["transported_out_mass"])
    summary["net_expansion_rank"] = rank_desc(summary["net_shift_mass"])
    summary["cost_weighted_mass_rank"] = rank_desc(summary["cost_weighted_mass"])
    summary["fragmentation_rank"] = rank_desc(summary["split_entropy"])
    summary["absolute_shift_rank"] = rank_desc(summary["net_shift_mass"].abs())
    summary["event_labels"] = summary.apply(label_source_event, axis=1)

    rank_cols = [
        "transported_mass_rank",
        "net_expansion_rank",
        "cost_weighted_mass_rank",
        "fragmentation_rank",
        "absolute_shift_rank",
    ]
    summary["best_transport_event_rank"] = summary[rank_cols].min(axis=1)
    summary["best_transport_event_axis"] = summary[rank_cols].idxmin(axis=1).str.replace("_rank", "", regex=False)
    return summary.sort_values(["best_transport_event_rank", "transported_mass_rank", "healthy_cluster_id"])


def label_source_event(row: pd.Series) -> str:
    labels = []
    if row["expansion_mass"] > 1e-9:
        labels.append("expansion")
    if row["reduction_mass"] > 1e-9:
        labels.append("reduction")
    if row["n_disease_branches"] >= 2:
        labels.append("fragmentation")
    if pd.notna(row["mean_transport_cost"]) and row["mean_transport_cost"] >= 0.50 and row["transported_out_mass"] > 0:
        labels.append("high_cost")
    return "|".join(labels) if labels else "low_signal"


def summarize_targets(
    solution: TransportSolution,
    total_cost: np.ndarray,
    healthy_df: pd.DataFrame,
    disease_df: pd.DataFrame,
    disease_mass: np.ndarray,
    options: object,
) -> pd.DataFrame:
    rows = []
    healthy_ids = healthy_df["cluster_id"].astype(str).tolist()
    disease_ids = disease_df["cluster_id"].astype(str).tolist()
    col_sum = solution.transport.sum(axis=0)
    pseudocount_mass = float(get_option(options, "pseudocount_mass", 1e-6))
    unknown_fraction_threshold = float(get_option(options, "unknown_fraction_threshold", 0.30))

    unknown_col = "Comp_unknown"
    disease_unknown = (
        pd.to_numeric(disease_df[unknown_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if unknown_col in disease_df.columns
        else np.zeros(len(disease_df), dtype=float)
    )

    for j, d_id in enumerate(disease_ids):
        incoming = solution.transport[:, j]
        explained = float(col_sum[j])
        residual = float(solution.residual[j])
        disease_total = float(disease_mass[j])
        if incoming.sum() > 0:
            dominant_i = int(np.argmax(incoming))
            dominant_source = healthy_ids[dominant_i]
            dominant_fraction = float(incoming[dominant_i] / max(disease_total, pseudocount_mass))
            mean_cost = float((incoming * total_cost[:, j]).sum() / incoming.sum())
        else:
            dominant_source = ""
            dominant_fraction = 0.0
            mean_cost = np.nan
        entropy = shannon_entropy(incoming)
        row = {
            "disease_cluster_id": d_id,
            "disease_mass": disease_total,
            "explained_transport_mass": explained,
            "residual_mass": residual,
            "residual_fraction": residual / disease_total if disease_total > 0 else np.nan,
            "dominant_healthy_source": dominant_source,
            "dominant_source_fraction_of_target": dominant_fraction,
            "mean_incoming_cost": mean_cost,
            "mixed_source_entropy": entropy,
            "effective_n_sources": effective_count_from_entropy(entropy),
            "unknown_fraction": float(disease_unknown[j]),
            "high_unknown_flag": bool(disease_unknown[j] >= unknown_fraction_threshold),
        }
        row.update(source_metadata(disease_df.iloc[j], "disease"))
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["residual_rank"] = rank_desc(summary["residual_mass"])
    summary["incoming_cost_rank"] = rank_desc(summary["mean_incoming_cost"].fillna(-np.inf))
    summary["mixed_source_rank"] = rank_desc(summary["mixed_source_entropy"])
    return summary.sort_values(["residual_rank", "disease_cluster_id"])


def build_plan_df(
    solution: TransportSolution,
    cost_long: pd.DataFrame,
    healthy_df: pd.DataFrame,
    disease_df: pd.DataFrame,
    healthy_mass: np.ndarray,
    disease_mass: np.ndarray,
    total_cost: np.ndarray,
    options: object,
) -> pd.DataFrame:
    healthy_ids = healthy_df["cluster_id"].astype(str).tolist()
    disease_ids = disease_df["cluster_id"].astype(str).tolist()
    cost_lookup = cost_long.set_index(["healthy_cluster_id", "disease_cluster_id"])
    rows = []
    plan_mass_tol = float(get_option(options, "plan_mass_tol", 1e-10))
    for i, h_id in enumerate(healthy_ids):
        source_total = solution.transport[i, :].sum()
        for j, d_id in enumerate(disease_ids):
            mass = float(solution.transport[i, j])
            if mass <= plan_mass_tol:
                continue
            cost_row = cost_lookup.loc[(h_id, d_id)]
            row = {
                "healthy_cluster_id": h_id,
                "disease_cluster_id": d_id,
                "transport_mass": mass,
                "transport_fraction_of_healthy_source": mass / source_total if source_total > 0 else np.nan,
                "transport_fraction_of_disease_target": mass / disease_mass[j] if disease_mass[j] > 0 else np.nan,
                "total_cost": float(total_cost[i, j]),
                "composition_cost_raw": cost_row["composition_cost_raw"],
                "composition_similarity": cost_row["composition_similarity"],
                "healthy_mass": float(healthy_mass[i]),
                "disease_mass": float(disease_mass[j]),
            }
            row.update(source_metadata(healthy_df.iloc[i], "healthy"))
            row.update(source_metadata(disease_df.iloc[j], "disease"))
            rows.append(row)
    columns = [
        "healthy_cluster_id",
        "disease_cluster_id",
        "transport_mass",
        "transport_fraction_of_healthy_source",
        "transport_fraction_of_disease_target",
        "total_cost",
        "composition_cost_raw",
        "composition_similarity",
        "healthy_mass",
        "disease_mass",
        "healthy_N_Cells",
        "healthy_N_Slides",
        "healthy_signature",
        "healthy_top_enriched",
        "disease_N_Cells",
        "disease_N_Slides",
        "disease_signature",
        "disease_top_enriched",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).reindex(columns=columns).sort_values(
        ["transport_mass", "healthy_cluster_id"],
        ascending=[False, True],
    )


def build_bridge_table(source_summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "healthy_cluster_id",
        "healthy_mass",
        "current_disease_family_mass",
        "current_log2_abundance_shift",
        "current_expansion_rank",
        "transported_out_mass",
        "log2_transport_shift",
        "transported_mass_rank",
        "net_expansion_rank",
        "cost_weighted_mass",
        "cost_weighted_mass_rank",
        "split_entropy",
        "fragmentation_rank",
        "mean_transport_cost",
        "event_labels",
        "best_transport_event_rank",
        "best_transport_event_axis",
        "top_disease_targets",
        "current_top_disease_clusters",
    ]
    available = [c for c in cols if c in source_summary.columns]
    return source_summary.loc[:, available].sort_values(
        ["current_expansion_rank", "best_transport_event_rank", "healthy_cluster_id"]
    )


def run_sensitivity_grid(
    cost: np.ndarray,
    healthy_mass: np.ndarray,
    disease_mass: np.ndarray,
    healthy_df: pd.DataFrame,
    shift_values: Optional[Sequence[float]] = None,
    residual_values: Optional[Sequence[float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    source_ids = healthy_df["cluster_id"].astype(str).tolist()
    rows = []
    top_rows = []
    if shift_values is None:
        shift_values = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35]
    if residual_values is None:
        residual_values = [0.60, 0.80, 1.00]
    for lam_er in shift_values:
        for lam_res in residual_values:
            sol = solve_unbalanced_transport(cost, healthy_mass, disease_mass, lam_er, lam_er, lam_res)
            row_sum = sol.transport.sum(axis=1)
            cost_weighted = (sol.transport * cost).sum(axis=1)
            split_entropy = np.array([shannon_entropy(sol.transport[i, :]) for i in range(sol.transport.shape[0])])
            frame = pd.DataFrame(
                {
                    "healthy_cluster_id": source_ids,
                    "lambda_expand": lam_er,
                    "lambda_reduce": lam_er,
                    "lambda_residual": lam_res,
                    "transported_out_mass": row_sum,
                    "net_shift_mass": row_sum - healthy_mass,
                    "cost_weighted_mass": cost_weighted,
                    "split_entropy": split_entropy,
                    "total_transport_cost": float((sol.transport * cost).sum()),
                    "total_expansion_mass": float(sol.expansion.sum()),
                    "total_reduction_mass": float(sol.reduction.sum()),
                    "total_residual_mass": float(sol.residual.sum()),
                    "objective": float(sol.objective),
                }
            )
            frame["transported_mass_rank"] = rank_desc(frame["transported_out_mass"])
            frame["net_expansion_rank"] = rank_desc(frame["net_shift_mass"])
            frame["cost_weighted_mass_rank"] = rank_desc(frame["cost_weighted_mass"])
            frame["fragmentation_rank"] = rank_desc(frame["split_entropy"])
            rows.append(frame)
            for axis in ["transported_mass_rank", "net_expansion_rank", "cost_weighted_mass_rank", "fragmentation_rank"]:
                top = frame.sort_values([axis, "healthy_cluster_id"]).iloc[0]
                top_rows.append(
                    {
                        "lambda_expand": lam_er,
                        "lambda_reduce": lam_er,
                        "lambda_residual": lam_res,
                        "axis": axis.replace("_rank", ""),
                        "top_healthy_cluster_id": top["healthy_cluster_id"],
                    }
                )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(top_rows)


def plot_transport_heatmap(
    solution: TransportSolution,
    healthy_df: pd.DataFrame,
    disease_df: pd.DataFrame,
    out_path: Path,
    top_n: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib is unavailable; skipping heatmap.")
        return

    transport = solution.transport
    source_order = np.argsort(-transport.sum(axis=1))[:top_n]
    target_order = np.argsort(-transport.sum(axis=0))[:top_n]
    if source_order.size == 0 or target_order.size == 0:
        return

    sub = transport[np.ix_(source_order, target_order)]
    fig_w = max(8, 0.35 * len(target_order))
    fig_h = max(6, 0.35 * len(source_order))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(sub, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(target_order)))
    ax.set_yticks(np.arange(len(source_order)))
    ax.set_xticklabels(disease_df.iloc[target_order]["cluster_id"].astype(str).tolist(), rotation=90)
    ax.set_yticklabels(healthy_df.iloc[source_order]["cluster_id"].astype(str).tolist())
    ax.set_xlabel("Disease target cluster")
    ax.set_ylabel("Healthy source cluster")
    ax.set_title("Healthy-to-disease transport mass")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Transport mass")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_json(path: Path, data: Dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
