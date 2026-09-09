from collections import defaultdict

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from ._download import ensure_database
from ._orthology import resolve_ortholog_pairs


def _gene_lookup(values: pd.Index) -> dict[str, str]:
    return {value.casefold(): value for value in values.astype(str)}


def _canonical_edges(
    frame: pd.DataFrame,
    from_lookup: dict[str, str],
    to_lookup: dict[str, str],
) -> list[tuple[str, str]]:
    edges = set()
    for source, target in frame[["from", "to"]].itertuples(index=False, name=None):
        source_key = str(source).casefold()
        target_key = str(target).casefold()
        if source_key in from_lookup and target_key in to_lookup:
            edges.add((from_lookup[source_key], to_lookup[target_key]))
    return sorted(edges, key=lambda edge: (edge[0].casefold(), edge[1].casefold()))


def _expression_means(adata: ad.AnnData) -> pd.Series:
    values = np.asarray(adata.X.mean(axis=0)).ravel().astype(float)
    return pd.Series(values, index=adata.var_names.astype(str))


def _dispersion_values(adata: ad.AnnData) -> pd.Series:
    if "dispersions_norm" not in adata.var:
        statistics = sc.pp.highly_variable_genes(adata, inplace=False)
        adata.var["dispersions_norm"] = statistics["dispersions_norm"].to_numpy()
    values = pd.to_numeric(adata.var["dispersions_norm"], errors="coerce")
    return pd.Series(values.to_numpy(dtype=float), index=adata.var_names.astype(str))


def _target_order(target: str, dispersions: pd.Series) -> tuple[float, str, str]:
    return -float(dispersions[target]), target.casefold(), target


def _select_targets(
    rtg: list[tuple[str, str]],
    receptors: set[str],
    targets: set[str],
    dispersions: pd.Series,
    max_targets: int,
    min_targets_per_receptor: int = 3,
) -> set[str]:
    receptor_to_targets: dict[str, set[str]] = defaultdict(set)
    for receptor, target in rtg:
        if receptor in receptors and target in targets:
            receptor_to_targets[receptor].add(target)

    selected: set[str] = set()
    for receptor in sorted(receptors, key=lambda value: (value.casefold(), value)):
        ranked = sorted(
            receptor_to_targets[receptor],
            key=lambda target: _target_order(target, dispersions),
        )
        selected.update(ranked[: min(min_targets_per_receptor, len(ranked))])

    if len(selected) > max_targets:
        if min_targets_per_receptor <= 1:
            raise ValueError(
                f"Cannot preserve receptor connectivity with max_targets={max_targets}"
            )
        return _select_targets(
            rtg,
            receptors,
            targets,
            dispersions,
            max_targets,
            min_targets_per_receptor - 1,
        )

    candidates = {
        target
        for receptor, target in rtg
        if receptor in receptors and target in targets and target not in selected
    }
    selected.update(
        sorted(candidates, key=lambda target: _target_order(target, dispersions))[
            : max_targets - len(selected)
        ]
    )
    return selected


def prepare_network(
    adata_mouse: ad.AnnData,
    adata_human: ad.AnnData,
    *,
    receptor_mean_cutoff: float = 0.05,
    dispersion_cutoff: float = -10.0,
    max_targets: int = 2000,
) -> dict[str, object]:
    database = ensure_database()
    pairs = resolve_ortholog_pairs(
        adata_human,
        adata_mouse,
        database / "human_mouse_orthologs.parquet",
    )
    mouse_lookup = _gene_lookup(adata_mouse.var_names)
    aligned_ligands = set(pairs["mouse_feature"].str.casefold())
    ligand_lookup = {
        key: mouse_lookup[key] for key in sorted(aligned_ligands & set(mouse_lookup))
    }
    lr_frame = pd.read_parquet(database / "mouse_lr.parquet", columns=["from", "to"])
    rtg_frame = pd.read_parquet(database / "mouse_rtg.parquet", columns=["from", "to"])
    lr = _canonical_edges(lr_frame, ligand_lookup, mouse_lookup)
    rtg = _canonical_edges(rtg_frame, mouse_lookup, mouse_lookup)
    means = _expression_means(adata_mouse)
    dispersions = _dispersion_values(adata_mouse)

    receptors = sorted(
        {receptor for _, receptor in lr if means[receptor] > receptor_mean_cutoff}
    )
    receptor_set = set(receptors)
    ligands = sorted(
        {
            ligand
            for ligand, receptor in lr
            if receptor in receptor_set and dispersions[ligand] > dispersion_cutoff
        }
    )
    ligand_set = set(ligands)
    lr = [edge for edge in lr if edge[0] in ligand_set and edge[1] in receptor_set]
    rtg = [
        (receptor, target)
        for receptor, target in rtg
        if receptor in receptor_set
        and target not in receptor_set
        and target not in ligand_set
        and dispersions[target] > dispersion_cutoff
    ]

    targets = sorted({target for _, target in rtg})
    receptors = sorted(
        {receptor for receptor, _ in rtg} & {receptor for _, receptor in lr}
    )
    receptor_set = set(receptors)
    ligands = sorted({ligand for ligand, receptor in lr if receptor in receptor_set})
    ligand_set = set(ligands)
    if len(targets) > max_targets:
        targets = sorted(
            _select_targets(
                rtg,
                receptor_set,
                set(targets),
                dispersions,
                max_targets,
            )
        )
    lr = [edge for edge in lr if edge[0] in ligand_set and edge[1] in receptor_set]
    target_set = set(targets)
    rtg = [
        edge for edge in rtg if edge[0] in receptor_set and edge[1] in target_set
    ]

    ligand_index = {gene: index for index, gene in enumerate(ligands)}
    receptor_index = {gene: index for index, gene in enumerate(receptors)}
    target_index = {gene: index for index, gene in enumerate(targets)}
    ligand_receptor_matrix = np.zeros((len(ligands), len(receptors)), dtype=np.uint8)
    for ligand, receptor in lr:
        ligand_receptor_matrix[ligand_index[ligand], receptor_index[receptor]] = 1
    receptor_target_matrix = np.zeros(
        (len(receptors), len(targets)), dtype=np.uint8
    )
    for receptor, target in rtg:
        receptor_target_matrix[receptor_index[receptor], target_index[target]] = 1

    human_by_mouse = dict(
        zip(pairs["mouse_feature"].str.casefold(), pairs["human_feature"], strict=True)
    )
    human_ligands = tuple(str(human_by_mouse[ligand.casefold()]) for ligand in ligands)
    return {
        "ligands": tuple(ligands),
        "human_ligands": human_ligands,
        "receptors": tuple(receptors),
        "targets": tuple(targets),
        "ligand_receptor_matrix": ligand_receptor_matrix,
        "receptor_target_matrix": receptor_target_matrix,
    }
