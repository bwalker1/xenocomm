import re
from pathlib import Path

import anndata as ad
import pandas as pd


def _ids(value) -> list[str]:
    values = value if isinstance(value, (list, tuple, set)) else str(value).split(";")
    return [re.sub(r"\.\d+$", "", str(item).strip()) for item in values]


def _features(adata: ad.AnnData):
    id_column = "gene_id" if "gene_id" in adata.var else "gene_ids"
    names = [str(name) for name in adata.var_names]
    by_id: dict[str, set[str]] = {}
    for name, value in zip(names, adata.var[id_column], strict=True):
        for stable_id in _ids(value):
            by_id.setdefault(stable_id, set()).add(name)
    return (
        {
            stable_id: next(iter(found))
            for stable_id, found in by_id.items()
            if len(found) == 1
        },
        {name.casefold(): name for name in names},
    )


def resolve_ortholog_pairs(
    adata_human: ad.AnnData,
    adata_mouse: ad.AnnData,
    path: str | Path,
) -> pd.DataFrame:
    orthologs = pd.read_parquet(path)
    orthologs = orthologs[
        orthologs["homology_type"].eq("ortholog_one2one")
        & orthologs["confidence"].eq(1)
    ].copy()
    orthologs["human_gene_id"] = orthologs["human_gene_id"].map(
        lambda value: re.sub(r"\.\d+$", "", str(value))
    )
    orthologs["mouse_gene_id"] = orthologs["mouse_gene_id"].map(
        lambda value: re.sub(r"\.\d+$", "", str(value))
    )
    human_by_id, human_by_name = _features(adata_human)
    mouse_by_id, mouse_by_name = _features(adata_mouse)
    rows = [
        (mouse_by_id[row.mouse_gene_id], human_by_id[row.human_gene_id])
        for row in orthologs.itertuples(index=False)
        if row.mouse_gene_id in mouse_by_id and row.human_gene_id in human_by_id
    ]
    pairs = pd.DataFrame(
        rows, columns=["mouse_feature", "human_feature"]
    ).drop_duplicates()
    if not pairs.empty:
        mouse_key = pairs["mouse_feature"].str.casefold()
        human_key = pairs["human_feature"].str.casefold()
        pairs = pairs[
            mouse_key.map(mouse_key.value_counts()).eq(1)
            & human_key.map(human_key.value_counts()).eq(1)
        ]

    used_mouse = set(pairs["mouse_feature"].str.casefold())
    used_human = set(pairs["human_feature"].str.casefold())
    exact = [
        (mouse_by_name[key], human_by_name[key])
        for key in sorted(set(mouse_by_name) & set(human_by_name))
        if key not in used_mouse and key not in used_human
    ]
    if exact:
        pairs = pd.concat(
            [pairs, pd.DataFrame(exact, columns=pairs.columns)], ignore_index=True
        )
    return pairs.sort_values(
        ["mouse_feature", "human_feature"],
        key=lambda values: values.str.casefold(),
        kind="mergesort",
    ).reset_index(drop=True)
