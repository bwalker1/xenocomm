from __future__ import annotations

import re
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from . import _common
from ._common import (
    _aligned_matrix,
    _casefold_index,
    _gene_id,
    _resource_genes,
    _sample_rows,
)


def _receiver_group(cell_type: str) -> str:
    suffix = re.sub(r"[^a-z0-9]+", "_", cell_type.casefold()).strip("_")
    return f"mouse_{suffix}"


def _build_grouped_adata(
    human: ad.AnnData,
    mouse: ad.AnnData,
    sample: str,
    source_cell_type: str,
    receiver_cell_types: tuple[str, ...],
    resource_genes: list[str],
    maximum: int,
    seed: int,
    human_feature_map: dict[str, str] | None = None,
    mouse_feature_map: dict[str, str] | None = None,
) -> ad.AnnData:
    human_index = _casefold_index(human.var_names, "human data")
    mouse_index = _casefold_index(mouse.var_names, "mouse data")
    genes = _resource_genes(
        dict.fromkeys(resource_genes),
        human_index,
        mouse_index,
        human_feature_map,
        mouse_feature_map,
    )
    if not genes:
        raise ValueError("No resource genes can be aligned to the PDX matrices")
    blocks: list[ad.AnnData] = []

    human_mask = (human.obs["sample"].astype(str).to_numpy() == sample) & (
        human.obs["cell_type"].astype(str).to_numpy() == source_cell_type
    )
    human_candidates = np.flatnonzero(human_mask)
    if human_candidates.size == 0:
        raise ValueError(f"No {source_cell_type} cells for PDX sample {sample}")
    human_rows = _sample_rows(
        human_candidates,
        maximum,
        seed,
        sample,
        "human",
        source_cell_type,
    )
    human_obs = pd.DataFrame(
        {
            "benchmark_group": "human_tumor",
            "sample": sample,
            "species": "human",
            "cell_type": source_cell_type,
        },
        index=[f"human_{index}" for index in human_rows],
    )
    blocks.append(
        ad.AnnData(
            X=_aligned_matrix(
                human, human_rows, genes, human_feature_map, "human data"
            ),
            obs=human_obs,
            var=pd.DataFrame(index=pd.Index(genes)),
        )
    )

    mouse_sample = mouse.obs["sample"].astype(str).to_numpy() == sample
    mouse_types = mouse.obs["cell_type"].astype(str).to_numpy()
    for cell_type in receiver_cell_types:
        candidates = np.flatnonzero(mouse_sample & (mouse_types == cell_type))
        if candidates.size == 0:
            raise ValueError(f"No {cell_type} cells for PDX sample {sample}")
        rows = _sample_rows(
            candidates,
            maximum,
            seed,
            sample,
            "mouse",
            cell_type,
        )
        group = _receiver_group(cell_type)
        obs = pd.DataFrame(
            {
                "benchmark_group": group,
                "sample": sample,
                "species": "mouse",
                "cell_type": cell_type,
            },
            index=[f"mouse_{index}" for index in rows],
        )
        blocks.append(
            ad.AnnData(
                X=_aligned_matrix(mouse, rows, genes, mouse_feature_map, "mouse data"),
                obs=obs,
                var=pd.DataFrame(index=pd.Index(genes)),
            )
        )

    combined = ad.concat(blocks, join="inner", merge="same")
    combined.X = sparse.csr_matrix(combined.X, dtype=np.float32)
    combined.uns["log1p"] = {"base": None}
    return combined


def _build_pooled_grouped_adata(
    human: ad.AnnData,
    mouse: ad.AnnData,
    samples: list[str],
    source_cell_type: str,
    receiver_cell_types: tuple[str, ...],
    resource_genes: list[str],
    maximum: int,
    seed: int,
    human_feature_map: dict[str, str] | None = None,
    mouse_feature_map: dict[str, str] | None = None,
) -> ad.AnnData:
    blocks: list[ad.AnnData] = []
    for sample in samples:
        block = _build_grouped_adata(
            human,
            mouse,
            sample,
            source_cell_type,
            receiver_cell_types,
            resource_genes,
            maximum,
            seed,
            human_feature_map,
            mouse_feature_map,
        )
        blocks.append(block)
    combined = ad.concat(blocks, join="inner", merge="same")
    combined.X = sparse.csr_matrix(combined.X, dtype=np.float32)
    combined.uns["log1p"] = {"base": None}
    return combined


def _cellchat_subunits(value: Any) -> tuple[str, ...]:
    if pd.isna(value):
        return ()
    return tuple(part for part in re.split(r"\s*[,;]\s*", str(value)) if part)


def _annotate_cellchat_database(
    database: pd.DataFrame,
    mouse_to_human: dict[str, str],
    present_human: set[str],
    present_mouse: set[str],
) -> pd.DataFrame:
    return _common._annotate_cellchat_database(
        database,
        mouse_to_human,
        present_human,
        present_mouse,
        splitter=_cellchat_subunits,
        missing_source_reason="source_ligand_missing_ortholog",
    )


def _annotate_liana_resource(
    resource: pd.DataFrame,
    human_to_mouse: dict[str, str],
    present_human: set[str],
    present_mouse: set[str],
) -> pd.DataFrame:
    result = resource.copy()
    ligands = result["ligand"].astype(str)
    annotations = _common._liana_annotations(
        ligands,
        result["receptor"],
        human_to_mouse,
        present_mouse,
        ligands.map(lambda value: _gene_id(value) in present_human),
    )
    # PDX rankings require a mapped mouse ligand ID.
    annotations["directional_eligible"] &= annotations["ligand_id"].ne("")
    result = result.assign(**annotations)
    return result


def _cellchat_interactions(
    result: pd.DataFrame,
    database: pd.DataFrame,
    sample: str,
    seed: int,
    source_cell_type: str,
    target_lookup: dict[str, str],
    significance_threshold: float,
) -> pd.DataFrame:
    database = database.drop_duplicates("interaction_name")
    merged = result.merge(
        database,
        on="interaction_name",
        how="left",
        validate="many_to_one",
        suffixes=("", "_database"),
    )
    merged = merged.loc[
        (merged["source"].astype(str) == "human_tumor")
        & merged["target"].astype(str).isin(target_lookup)
    ].copy()
    return _common._cellchat_table(
        merged,
        sample=sample,
        seed=seed,
        source_cell_type=source_cell_type,
        target_cell_type=merged["target"].map(target_lookup),
        significance_threshold=significance_threshold,
    )


def _liana_interactions(
    result: pd.DataFrame,
    sample: str,
    seed: int,
    source_cell_type: str,
    target_lookup: dict[str, str],
    significance_threshold: float,
    human_to_mouse: dict[str, str],
    present_human: set[str],
    present_mouse: set[str],
) -> pd.DataFrame:
    frame = result.loc[
        (result["source"].astype(str) == "human_tumor")
        & result["target"].astype(str).isin(target_lookup)
    ].copy()
    ligands = frame["ligand_complex"].fillna("").astype(str)
    annotations = _common._liana_annotations(
        ligands,
        frame["receptor_complex"].fillna("").astype(str),
        human_to_mouse,
        present_mouse,
        ligands.map(lambda value: _gene_id(value) in present_human),
    )
    # PDX rankings require a mapped mouse ligand ID.
    annotations["directional_eligible"] &= annotations["ligand_id"].ne("")
    return _common._liana_table(
        frame,
        annotations,
        sample=sample,
        seed=seed,
        source_cell_type=source_cell_type,
        target_cell_type=frame["target"].map(target_lookup),
        significance_threshold=significance_threshold,
    )


def _aggregate_method_interactions(
    interactions: pd.DataFrame,
    native_ligands: dict[str, set[str]],
    ligand_labels: dict[str, str],
    target_cell_types: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    replicate_columns = ["method", "biological_sample", "replicate_seed", "replicate"]
    replicates = interactions.loc[:, replicate_columns].drop_duplicates()
    for replicate in replicates.itertuples(index=False):
        method_frame = interactions.loc[
            (interactions["method"] == replicate.method)
            & (interactions["replicate"] == replicate.replicate)
            & interactions["eligible_ligand"]
        ]
        universe = sorted(native_ligands[replicate.method])
        for target in [*target_cell_types, "ALL_STROMA"]:
            target_frame = (
                method_frame
                if target == "ALL_STROMA"
                else method_frame.loc[method_frame["target_cell_type"] == target]
            )
            grouped = target_frame.groupby("ligand_id", sort=False)
            base = pd.DataFrame({"ligand_id": universe})
            counts = grouped.agg(
                n_interactions=("raw_score", "size"),
                n_significant=("significant", "sum"),
            ).reset_index()
            significant = target_frame.loc[target_frame["significant"]]
            scores = significant.groupby("ligand_id", sort=False)["raw_score"].sum()
            score_frame = scores.rename("score").reset_index()
            result = base.merge(counts, on="ligand_id", how="left").merge(
                score_frame, on="ligand_id", how="left"
            )
            result[["n_interactions", "n_significant", "score"]] = result[
                ["n_interactions", "n_significant", "score"]
            ].fillna(0)
            result.insert(0, "method", replicate.method)
            result.insert(1, "biological_sample", replicate.biological_sample)
            result.insert(2, "replicate_seed", replicate.replicate_seed)
            result.insert(3, "replicate", replicate.replicate)
            result.insert(4, "aggregation", "significant_sum")
            result.insert(5, "target_cell_type", target)
            result["ligand_label"] = result["ligand_id"].map(ligand_labels)
            result["detected"] = result["n_significant"] > 0
            result["eligible_ligand"] = True
            result["n_interactions"] = result["n_interactions"].astype(int)
            result["n_significant"] = result["n_significant"].astype(int)
            rows.append(result)
    return pd.concat(rows, ignore_index=True)


def _rank_scores(
    frame: pd.DataFrame,
    universe: set[str],
    universe_name: str,
) -> pd.DataFrame:
    result = frame.loc[frame["ligand_id"].isin(universe)].copy()
    if len(result) != len(universe) or result["ligand_id"].duplicated().any():
        raise ValueError("Each ranking must contain its ligand universe exactly once")
    if not np.isfinite(result["score"].to_numpy(dtype=float)).all():
        raise ValueError("Ranking scores must be finite")
    result.sort_values(
        ["score", "ligand_id"],
        ascending=[False, True],
        kind="mergesort",
        inplace=True,
    )
    result.reset_index(drop=True, inplace=True)
    result["rank"] = np.arange(1, len(result) + 1, dtype=int)
    result["average_rank"] = result["score"].rank(method="average", ascending=False)
    denominator = max(len(result) - 1, 1)
    result["percentile"] = 1.0 - (result["average_rank"] - 1.0) / denominator
    result["universe"] = universe_name
    return result


def _build_replicate_rankings(
    target_scores: pd.DataFrame,
    native_ligands: dict[str, set[str]],
    shared_ligands: set[str],
) -> pd.DataFrame:
    pooled = target_scores.loc[target_scores["target_cell_type"] == "ALL_STROMA"]
    keys = ["method", "replicate", "aggregation"]
    rankings: list[pd.DataFrame] = []
    for key, frame in pooled.groupby(keys, sort=False):
        method = str(key[0])
        for universe_name, universe in (
            ("native", native_ligands[method]),
            ("shared", shared_ligands),
        ):
            rankings.append(_rank_scores(frame, universe, universe_name))
    keep = [
        "method",
        "biological_sample",
        "replicate_seed",
        "replicate",
        "aggregation",
        "universe",
        "ligand_id",
        "ligand_label",
        "score",
        "rank",
        "average_rank",
        "percentile",
        "detected",
        "n_interactions",
        "n_significant",
    ]
    return pd.concat(rankings, ignore_index=True).loc[:, keep]


def _build_consensus_rankings(replicate_rankings: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "aggregation", "universe", "ligand_id", "ligand_label"]
    result = (
        replicate_rankings.groupby(keys, as_index=False, observed=True)
        .agg(
            score_mean=("score", "mean"),
            score_sd=("score", "std"),
            percentile_mean=("percentile", "mean"),
            percentile_sd=("percentile", "std"),
            detected_fraction=("detected", "mean"),
            replicate_count=("replicate", "nunique"),
        )
        .fillna({"score_sd": 0.0, "percentile_sd": 0.0})
    )
    ranked: list[pd.DataFrame] = []
    for _, frame in result.groupby(["method", "aggregation", "universe"], sort=False):
        frame = frame.sort_values(
            ["percentile_mean", "ligand_id"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        frame["rank"] = np.arange(1, len(frame) + 1, dtype=int)
        frame["average_rank"] = frame["percentile_mean"].rank(
            method="average", ascending=False
        )
        ranked.append(frame)
    return pd.concat(ranked, ignore_index=True)
