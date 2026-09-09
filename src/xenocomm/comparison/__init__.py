from types import SimpleNamespace

import pandas as pd

from . import pdx, tenx
from ._common import _gene_id, _resource_subunits
from .worker import _cellchat_isolated

METHOD_LABELS = {
    "xenocomm": "XenoComm",
    "cellchat": "CellChat",
    "liana_cellphonedb": "LIANA",
}
RECEIVERS = (
    "Endothelial",
    "Fibroblast (cancer)",
    "Fibroblast (fibronectin)",
    "Hematopoietic",
    "Mural (muscle)",
)


def cellchat_resources(r_home, r_libraries):
    return _cellchat_isolated(
        "database",
        {
            "cellchat_r_home": r_home,
            "cellchat_r_libs_user": r_libraries,
        },
    )


def prepare_comparison(
    dataset, human, mouse, ligands, ortholog_path, cellchat_database, liana_resource
):
    from xenocomm._orthology import resolve_ortholog_pairs

    if dataset not in ("PDX", "10x"):
        raise ValueError("Dataset must be PDX or 10x")
    pairs = resolve_ortholog_pairs(human, mouse, ortholog_path)
    mouse_to_human = dict(
        zip(pairs.mouse_feature.str.casefold(), pairs.human_feature, strict=True)
    )
    human_to_mouse = dict(
        zip(pairs.human_feature.str.casefold(), pairs.mouse_feature, strict=True)
    )
    present_human = set(human.var_names.map(_gene_id))
    present_mouse = set(mouse.var_names.map(_gene_id))
    if dataset == "PDX":
        database = pdx._annotate_cellchat_database(
            cellchat_database, mouse_to_human, present_human, present_mouse
        )
        resource = pdx._annotate_liana_resource(
            liana_resource, human_to_mouse, present_human, present_mouse
        )
    else:
        database = tenx._annotate_cellchat_database(
            cellchat_database, present_human, present_mouse, mouse_to_human
        )
        resource = tenx._annotate_liana_resource(
            liana_resource, present_human, present_mouse, human_to_mouse
        )
    ligands = ligands.copy()
    ligands["ligand_id"] = ligands.ligand.map(_gene_id)
    native = {
        "xenocomm": set(ligands.ligand_id),
        "cellchat": set(database.loc[database.directional_eligible, "ligand_id"]),
        "liana_cellphonedb": set(
            resource.loc[resource.directional_eligible, "ligand_id"]
        ),
    }
    shared = set.intersection(*native.values())
    if not shared:
        raise ValueError("The methods have no shared ligand universe")
    labels = dict(zip(ligands.ligand_id, ligands.ligand, strict=True))
    for row in database.loc[database.eligible_ligand].itertuples():
        labels.setdefault(row.ligand_id, row.ligand_symbol)
    for row in resource.loc[resource.ligand_is_monomeric].itertuples():
        labels.setdefault(
            row.ligand_id, human_to_mouse.get(row.ligand.casefold(), row.ligand)
        )
    return SimpleNamespace(
        dataset=dataset,
        human=human,
        mouse=mouse,
        ligands=ligands,
        mouse_to_human=mouse_to_human,
        human_to_mouse=human_to_mouse,
        present_human=present_human,
        present_mouse=present_mouse,
        database=database,
        resource=resource,
        native=native,
        shared=shared,
        labels=labels,
    )


def run_comparators(
    study,
    seed,
    cellchat_genes,
    *,
    r_home,
    r_libraries,
    cell_cap,
    nboot=100,
    n_perms=1000,
    expr_prop=0.1,
    min_cells=20,
    significance_threshold=0.05,
):
    import liana as li

    if study.dataset == "PDX":
        samples = sorted(set(study.human.obs["sample"].astype(str)))
        if set(samples) != set(study.mouse.obs["sample"].astype(str)):
            raise ValueError("Human and mouse PDX samples must match")
        source = "human_tumor"
        targets = {pdx._receiver_group(label): label for label in RECEIVERS}

        def grouped(genes, **mapping):
            return pdx._build_pooled_grouped_adata(
                study.human,
                study.mouse,
                samples,
                "Tumor",
                RECEIVERS,
                genes,
                cell_cap,
                seed,
                **mapping,
            )
    else:
        source = tenx.SOURCE_GROUP
        targets = {tenx.TARGET_GROUP: tenx.TARGET_CELL_TYPE}

        def grouped(genes, **mapping):
            return tenx._build_grouped_adata(
                study.human,
                study.mouse,
                genes,
                cell_cap,
                seed,
                **mapping,
            )

    cellchat_adata = grouped(cellchat_genes, human_feature_map=study.mouse_to_human)
    liana_adata = grouped(
        _resource_subunits(study.resource), mouse_feature_map=study.human_to_mouse
    )
    if not cellchat_adata.obs.equals(liana_adata.obs):
        raise ValueError("CellChat and LIANA must receive the same cells and groups")
    cellchat_result = _cellchat_isolated(
        "run",
        {
            "adata": cellchat_adata,
            "cellchat_r_home": r_home,
            "cellchat_r_libs_user": r_libraries,
            "kwargs": {
                "nboot": nboot,
                "seed": seed,
                "source": source,
                "targets": list(targets),
            },
        },
    )
    liana_result = li.mt.cellphonedb(
        liana_adata,
        groupby="benchmark_group",
        resource=study.resource[["ligand", "receptor"]],
        expr_prop=expr_prop,
        min_cells=min_cells,
        groupby_pairs=pd.DataFrame({"source": source, "target": list(targets)}),
        return_all_lrs=True,
        use_raw=False,
        n_perms=n_perms,
        seed=seed,
        n_jobs=1,
        inplace=False,
        verbose=False,
    )
    if study.dataset == "PDX":
        cellchat = pdx._cellchat_interactions(
            cellchat_result,
            study.database,
            "ALL_PDX",
            seed,
            "Tumor",
            targets,
            significance_threshold,
        )
        liana = pdx._liana_interactions(
            liana_result,
            "ALL_PDX",
            seed,
            "Tumor",
            targets,
            significance_threshold,
            study.human_to_mouse,
            study.present_human,
            study.present_mouse,
        )
    else:
        cellchat = tenx._cellchat_interactions(
            cellchat_result, study.database, seed, significance_threshold
        )
        liana = tenx._liana_interactions(
            liana_result,
            seed,
            significance_threshold,
            study.present_human,
            study.present_mouse,
            study.human_to_mouse,
        )
    return pd.concat([cellchat, liana], ignore_index=True)


def rank_ligands(study, interactions, seeds):
    sample = "ALL_PDX" if study.dataset == "PDX" else tenx.BIOLOGICAL_SAMPLE
    # Empty method results still contribute a complete zero-score native universe.
    placeholders = pd.DataFrame(
        [
            {
                "method": method,
                "biological_sample": sample,
                "replicate_seed": seed,
                "replicate": f"{sample}__{seed}",
                "eligible_ligand": False,
            }
            for seed in seeds
            for method in ("cellchat", "liana_cellphonedb")
        ]
    )
    scores = pdx._aggregate_method_interactions(
        pd.concat([interactions, placeholders], ignore_index=True),
        study.native,
        study.labels,
        RECEIVERS if study.dataset == "PDX" else (tenx.TARGET_CELL_TYPE,),
    )
    xenocomm = []
    for seed in seeds:
        xenocomm.append(
            pd.DataFrame(
                {
                    "method": "xenocomm",
                    "biological_sample": sample,
                    "replicate_seed": seed,
                    "replicate": f"{sample}__{seed}",
                    "aggregation": "significant_sum",
                    "target_cell_type": "ALL_STROMA",
                    "ligand_id": study.ligands.ligand_id,
                    "ligand_label": study.ligands.ligand,
                    "score": study.ligands.delta_h_mean,
                    "detected": study.ligands.called.astype(bool),
                    "n_interactions": 1,
                    "n_significant": study.ligands.called.astype(int),
                }
            )
        )
    scores = pd.concat([scores, *xenocomm], ignore_index=True)
    replicate = pdx._build_replicate_rankings(scores, study.native, study.shared)
    consensus = pdx._build_consensus_rankings(replicate)
    return scores, replicate, consensus
