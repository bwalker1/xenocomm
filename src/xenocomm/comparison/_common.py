"""Shared expression alignment and comparator result tables."""

from __future__ import annotations

import hashlib
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def _gene_id(value: str) -> str:
    return str(value).casefold().upper()


def _casefold_index(values: pd.Index, label: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for index, value in enumerate(values.astype(str)):
        key = value.casefold()
        if key in result:
            raise ValueError(f"{label} has duplicate case-insensitive gene names")
        result[key] = index
    return result


def _stable_seed(seed: int, *parts: str) -> int:
    payload = ":".join((str(seed), *parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _sample_rows(
    candidates: np.ndarray,
    maximum: int,
    seed: int,
    *parts: str,
) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.int64)
    if candidates.size <= maximum:
        return np.sort(candidates)
    rng = np.random.default_rng(_stable_seed(seed, *parts))
    return np.sort(rng.choice(candidates, size=maximum, replace=False))


def _resource_subunits(resource: pd.DataFrame) -> list[str]:
    values = {
        subunit
        for column in ("ligand", "receptor")
        for entity in resource[column].astype(str)
        for subunit in entity.split("_")
        if subunit
    }
    return sorted(values, key=lambda value: (value.casefold(), value))


def _aligned_matrix(
    adata: ad.AnnData,
    rows: np.ndarray,
    genes: list[str],
    feature_map: dict[str, str] | None,
    label: str,
) -> sparse.csr_matrix:
    lookup = _casefold_index(adata.var_names, label)
    source_columns = []
    destination_columns = []
    for destination, gene in enumerate(genes):
        feature = feature_map.get(gene.casefold()) if feature_map is not None else gene
        source = lookup.get(feature.casefold()) if feature is not None else None
        if source is not None:
            source_columns.append(source)
            destination_columns.append(destination)
    if not source_columns:
        return sparse.csr_matrix((len(rows), len(genes)), dtype=np.float32)
    block = sparse.coo_matrix(adata.X[rows][:, source_columns], dtype=np.float32)
    columns = np.asarray(destination_columns, dtype=np.int64)[block.col]
    return sparse.csr_matrix(
        (block.data, (block.row, columns)),
        shape=(len(rows), len(genes)),
        dtype=np.float32,
    )


def _underscore_subunits(value: Any) -> tuple[str, ...]:
    if pd.isna(value):
        return ()
    return tuple(part for part in str(value).split("_") if part)


def _translated_complete(
    value: Any,
    translation: dict[str, str],
    present: set[str],
    splitter,
) -> bool:
    subunits = splitter(value)
    translated = [translation.get(subunit.casefold()) for subunit in subunits]
    return bool(subunits) and all(
        feature is not None and _gene_id(feature) in present for feature in translated
    )


def _all_present(
    value: Any,
    present: set[str],
    splitter,
) -> bool:
    subunits = splitter(value)
    return bool(subunits) and all(_gene_id(subunit) in present for subunit in subunits)


def _resource_genes(
    genes, human_index, mouse_index, human_feature_map, mouse_feature_map
):
    aligned = []
    for gene in genes:
        key = gene.casefold()
        human_feature = (
            human_feature_map.get(key) if human_feature_map is not None else gene
        )
        mouse_feature = (
            mouse_feature_map.get(key) if mouse_feature_map is not None else gene
        )
        if (human_feature is not None and human_feature.casefold() in human_index) or (
            mouse_feature is not None and mouse_feature.casefold() in mouse_index
        ):
            aligned.append(gene)
    return aligned


def _annotate_cellchat_database(
    database: pd.DataFrame,
    mouse_to_human: dict[str, str],
    present_human: set[str],
    present_mouse: set[str],
    *,
    splitter,
    missing_source_reason: str,
) -> pd.DataFrame:
    result = database.drop_duplicates("interaction_name").copy()
    result["ligand_id"] = result["ligand_symbol"].fillna("").map(_gene_id)
    result["source_ligand_present"] = result["ligand_symbol"].map(
        lambda value: _translated_complete(
            value, mouse_to_human, present_human, splitter
        )
    )
    result["target_receptor_complete"] = result["receptor_symbol"].map(
        lambda value: _all_present(value, present_mouse, splitter)
    )
    result["directional_eligible"] = (
        result["eligible_ligand"].fillna(False).astype(bool)
        & result["source_ligand_present"]
        & result["target_receptor_complete"]
    )
    exclusion = result["exclusion_reason"].fillna("").astype(str)
    exclusion = exclusion.mask(
        result["eligible_ligand"].fillna(False) & ~result["source_ligand_present"],
        missing_source_reason,
    )
    exclusion = exclusion.mask(
        result["eligible_ligand"].fillna(False)
        & result["source_ligand_present"]
        & ~result["target_receptor_complete"],
        "target_receptor_incomplete",
    )
    result["directional_exclusion_reason"] = exclusion.mask(
        result["directional_eligible"], ""
    )
    return result


def _cellchat_table(
    merged: pd.DataFrame,
    *,
    sample: str,
    seed: int,
    source_cell_type: str,
    target_cell_type: str | pd.Series,
    significance_threshold: float,
) -> pd.DataFrame:
    ligand_symbol = merged["ligand_symbol"].fillna("").astype(str)
    ligand_id = merged["ligand_id"].fillna("").astype(str)
    eligible = merged["directional_eligible"].fillna(False).astype(bool)
    exclusion = merged["directional_exclusion_reason"].fillna(
        "unresolved_resource_entity"
    )
    probability = merged["prob"].astype(float)
    p_value = merged["pval"].astype(float)
    return pd.DataFrame(
        {
            "method": "cellchat",
            "biological_sample": sample,
            "replicate_seed": seed,
            "replicate": f"{sample}__{seed}",
            "source_cell_type": source_cell_type,
            "target_cell_type": target_cell_type,
            "interaction_id": merged["interaction_name"].astype(str),
            "ligand_entity": merged["ligand_entity"].fillna("").astype(str),
            "ligand_symbol": ligand_symbol,
            "ligand_id": ligand_id,
            "receptor_entity": merged["receptor_entity"].fillna("").astype(str),
            "receptor_symbol": merged["receptor_symbol"].fillna("").astype(str),
            "pathway": merged["pathway"].fillna("").astype(str),
            "resource_name": "CellChatDB.mouse",
            "raw_score_name": "prob",
            "raw_score": probability.astype(float),
            "p_value": p_value.astype(float),
            "significant": (p_value <= significance_threshold).astype(bool),
            "ligand_is_complex": (
                merged["ligand_entity"].fillna("").astype(str) != ligand_symbol
            ),
            "ligand_is_monomeric_entity": merged["eligible_ligand"]
            .fillna(False)
            .astype(bool),
            "eligible_ligand": eligible,
            "exclusion_reason": exclusion.mask(eligible, ""),
        }
    )


def _liana_annotations(
    ligands, receptors, human_to_mouse, present_mouse, source_present
):
    monomer = ~ligands.str.contains("_", regex=False)
    target_complete = receptors.map(
        lambda value: _translated_complete(
            value, human_to_mouse, present_mouse, _underscore_subunits
        )
    )
    return {
        "ligand_id": ligands.map(
            lambda value: _gene_id(human_to_mouse.get(value.casefold(), ""))
        ),
        "ligand_is_monomeric": monomer,
        "source_ligand_present": source_present,
        "target_receptor_complete": target_complete,
        "directional_eligible": monomer & source_present & target_complete,
    }


def _liana_exclusion(annotations):
    monomer = annotations["ligand_is_monomeric"]
    source_present = annotations["source_ligand_present"]
    target_complete = annotations["target_receptor_complete"]
    exclusion = pd.Series("", index=monomer.index, dtype=object)
    exclusion = exclusion.mask(~monomer, "complex_ligand")
    exclusion = exclusion.mask(monomer & ~source_present, "source_ligand_missing")
    return exclusion.mask(
        monomer & source_present & ~target_complete, "target_receptor_incomplete"
    )


def _liana_table(
    frame: pd.DataFrame,
    annotations: dict[str, pd.Series],
    *,
    sample: str,
    seed: int,
    source_cell_type: str,
    target_cell_type: str | pd.Series,
    significance_threshold: float,
) -> pd.DataFrame:
    ligand_entity = frame["ligand_complex"].fillna("").astype(str)
    receptor_entity = frame["receptor_complex"].fillna("").astype(str)
    monomer = annotations["ligand_is_monomeric"]
    ligand_id = annotations["ligand_id"]
    eligible = annotations["directional_eligible"]
    exclusion = _liana_exclusion(annotations)
    score = frame["lr_means"].astype(float)
    p_value = frame["cellphone_pvals"].astype(float)
    return pd.DataFrame(
        {
            "method": "liana_cellphonedb",
            "biological_sample": sample,
            "replicate_seed": seed,
            "replicate": f"{sample}__{seed}",
            "source_cell_type": source_cell_type,
            "target_cell_type": target_cell_type,
            "interaction_id": ligand_entity + "^" + receptor_entity,
            "ligand_entity": ligand_entity,
            "ligand_symbol": ligand_entity.where(monomer, ""),
            "ligand_id": ligand_id,
            "receptor_entity": receptor_entity,
            "receptor_symbol": receptor_entity,
            "pathway": "",
            "resource_name": "LIANA CellPhoneDB",
            "raw_score_name": "lr_means",
            "raw_score": score.astype(float),
            "p_value": p_value.astype(float),
            "significant": (p_value <= significance_threshold).astype(bool),
            "ligand_is_complex": ~monomer,
            "ligand_is_monomeric_entity": monomer,
            "eligible_ligand": eligible,
            "exclusion_reason": exclusion,
        }
    )
