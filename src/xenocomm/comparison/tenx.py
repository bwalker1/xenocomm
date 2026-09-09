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
    _all_present,
    _casefold_index,
    _gene_id,
    _resource_genes,
    _sample_rows,
    _underscore_subunits,
)

SOURCE_GROUP = "human_HEK293T"
TARGET_GROUP = "mouse_NIH3T3"
SOURCE_CELL_TYPE = "HEK293T"
TARGET_CELL_TYPE = "NIH3T3"
BIOLOGICAL_SAMPLE = "10x_hgmm"


def _canonical_genes(values: list[str]) -> list[str]:
    genes: dict[str, str] = {}
    for value in values:
        gene = str(value)
        if gene:
            genes.setdefault(gene.casefold(), gene)
    return list(genes.values())


def _build_grouped_adata(
    human: ad.AnnData,
    mouse: ad.AnnData,
    resource_genes: list[str],
    maximum: int,
    seed: int,
    human_feature_map: dict[str, str] | None = None,
    mouse_feature_map: dict[str, str] | None = None,
) -> ad.AnnData:
    human_index = _casefold_index(human.var_names, "human 10x data")
    mouse_index = _casefold_index(mouse.var_names, "mouse 10x data")
    genes = _resource_genes(
        _canonical_genes(resource_genes),
        human_index,
        mouse_index,
        human_feature_map,
        mouse_feature_map,
    )
    if not genes:
        raise ValueError("No resource genes occur in either 10x species matrix")

    human_rows = _sample_rows(
        np.arange(human.n_obs), maximum, seed, BIOLOGICAL_SAMPLE, SOURCE_GROUP
    )
    mouse_rows = _sample_rows(
        np.arange(mouse.n_obs), maximum, seed, BIOLOGICAL_SAMPLE, TARGET_GROUP
    )
    matrix = sparse.vstack(
        [
            _aligned_matrix(
                human, human_rows, genes, human_feature_map, "10x expression data"
            ),
            _aligned_matrix(
                mouse, mouse_rows, genes, mouse_feature_map, "10x expression data"
            ),
        ],
        format="csr",
        dtype=np.float32,
    )
    obs = pd.DataFrame(
        {
            "benchmark_group": [SOURCE_GROUP] * len(human_rows)
            + [TARGET_GROUP] * len(mouse_rows),
            "species": ["human"] * len(human_rows) + ["mouse"] * len(mouse_rows),
            "cell_type": [SOURCE_CELL_TYPE] * len(human_rows)
            + [TARGET_CELL_TYPE] * len(mouse_rows),
        },
        index=[f"human_{index}" for index in human_rows]
        + [f"mouse_{index}" for index in mouse_rows],
    )
    combined = ad.AnnData(
        X=matrix,
        obs=obs,
        var=pd.DataFrame(index=pd.Index(genes)),
    )
    combined.uns["log1p"] = {"base": None}
    return combined


def _cellchat_subunits(value: Any) -> tuple[str, ...]:
    if pd.isna(value):
        return ()
    return tuple(part for part in re.split(r"\s*[,;]\s*", str(value).strip()) if part)


def _annotate_cellchat_database(
    database: pd.DataFrame,
    present_human: set[str],
    present_mouse: set[str],
    mouse_to_human: dict[str, str],
) -> pd.DataFrame:
    return _common._annotate_cellchat_database(
        database,
        mouse_to_human,
        present_human,
        present_mouse,
        splitter=_cellchat_subunits,
        missing_source_reason="source_ligand_missing",
    )


def _annotate_liana_resource(
    resource: pd.DataFrame,
    present_human: set[str],
    present_mouse: set[str],
    human_to_mouse: dict[str, str],
) -> pd.DataFrame:
    result = resource.loc[:, ["ligand", "receptor"]].astype(str).drop_duplicates()
    ligands = result["ligand"].astype(str)
    annotations = _common._liana_annotations(
        ligands,
        result["receptor"],
        human_to_mouse,
        present_mouse,
        ligands.map(
            lambda value: _all_present(value, present_human, _underscore_subunits)
        ),
    )
    result = result.assign(**annotations)
    result["directional_exclusion_reason"] = _common._liana_exclusion(annotations)
    return result


def _cellchat_interactions(
    result: pd.DataFrame,
    database: pd.DataFrame,
    seed: int,
    significance_threshold: float,
) -> pd.DataFrame:
    merged = result.merge(
        database,
        on="interaction_name",
        how="left",
        validate="many_to_one",
        suffixes=("", "_database"),
    )
    merged = merged.loc[
        (merged["source"].astype(str) == SOURCE_GROUP)
        & (merged["target"].astype(str) == TARGET_GROUP)
    ].copy()
    merged["ligand_id"] = merged["ligand_symbol"].fillna("").astype(str).map(_gene_id)
    return _common._cellchat_table(
        merged,
        sample=BIOLOGICAL_SAMPLE,
        seed=seed,
        source_cell_type=SOURCE_CELL_TYPE,
        target_cell_type=TARGET_CELL_TYPE,
        significance_threshold=significance_threshold,
    )


def _liana_interactions(
    result: pd.DataFrame,
    seed: int,
    significance_threshold: float,
    present_human: set[str],
    present_mouse: set[str],
    human_to_mouse: dict[str, str],
) -> pd.DataFrame:
    frame = result.loc[
        (result["source"].astype(str) == SOURCE_GROUP)
        & (result["target"].astype(str) == TARGET_GROUP)
    ].copy()
    ligands = frame["ligand_complex"].fillna("").astype(str)
    annotations = _common._liana_annotations(
        ligands,
        frame["receptor_complex"].fillna("").astype(str),
        human_to_mouse,
        present_mouse,
        ligands.map(
            lambda value: _all_present(value, present_human, _underscore_subunits)
        ),
    )
    return _common._liana_table(
        frame,
        annotations,
        sample=BIOLOGICAL_SAMPLE,
        seed=seed,
        source_cell_type=SOURCE_CELL_TYPE,
        target_cell_type=TARGET_CELL_TYPE,
        significance_threshold=significance_threshold,
    )
