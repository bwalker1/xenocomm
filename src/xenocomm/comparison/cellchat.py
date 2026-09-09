import anndata2ri
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import scipy.sparse as sp
from anndata import AnnData
from rpy2.robjects import default_converter, pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

_DATABASE_INTERACTIONS_R = r"""
function() {
    database <- CellChat::CellChatDB.mouse
    interactions <- database$interaction
    data.frame(
        interaction_name = interactions$interaction_name,
        ligand_entity = interactions$ligand,
        ligand_symbol = interactions$ligand.symbol,
        receptor_entity = interactions$receptor,
        receptor_symbol = interactions$receptor.symbol,
        pathway = interactions$pathway_name,
        annotation = interactions$annotation,
        evidence = interactions$evidence,
        ligand_is_complex = interactions$ligand %in% rownames(database$complex),
        stringsAsFactors = FALSE
    )
}
"""

_RUN_CELLCHAT_R = r"""
function(s, nboot, seed, source, targets) {
    empty_result <- function() {
        data.frame(
            source = character(), target = character(),
            interaction_name = character(), prob = double(), pval = double()
        )
    }
    assay(s, "logcounts") <- assay(s, "X")
    cellchat <- createCellChat(object = s, group.by = "benchmark_group")
    cellchat@DB <- CellChat::CellChatDB.mouse
    cellchat <- subsetData(cellchat)
    cellchat <- identifyOverExpressedGenes(cellchat, do.fast = FALSE)
    if (length(cellchat@var.features[["features"]]) == 0) {
        return(empty_result())
    }
    cellchat <- identifyOverExpressedInteractions(cellchat)
    if (nrow(cellchat@LR$LRsig) == 0) {
        return(empty_result())
    }
    cellchat <- computeCommunProb(
        cellchat,
        type = "truncatedMean",
        trim = 0.05,
        raw.use = TRUE,
        distance.use = FALSE,
        population.size = FALSE,
        nboot = as.integer(nboot),
        seed.use = as.integer(seed)
    )
    if (!any(cellchat@net$prob > 0)) {
        return(empty_result())
    }
    result <- subsetCommunication(
        cellchat,
        sources.use = source,
        targets.use = targets,
        thresh = 1 + .Machine$double.eps
    )
    result[, c("source", "target", "interaction_name", "prob", "pval")]
}
"""


class CellChat:
    def __init__(self):
        self.converter = default_converter + anndata2ri.converter + pandas2ri.converter
        importr("CellChat")

    def database_genes(self) -> list[str]:
        values = robjects.r(
            "unique(CellChat:::extractGene(CellChat::CellChatDB.mouse))"
        )
        return list(values)

    def database_interactions(self) -> pd.DataFrame:
        with localconverter(self.converter):
            result = robjects.r(_DATABASE_INTERACTIONS_R)()
        string_columns = result.columns.drop("ligand_is_complex")
        result[string_columns] = result[string_columns].astype("string")
        component_count = (
            result["ligand_symbol"]
            .fillna("")
            .map(lambda value: len([part for part in value.split(",") if part.strip()]))
        )
        result["ligand_component_count"] = component_count.astype("int64")
        reason = pd.Series("eligible", index=result.index, dtype="string")
        reason.loc[result["ligand_is_complex"]] = "complex_ligand"
        non_protein = result["annotation"].eq("Non-protein Signaling")
        reason.loc[non_protein] = "non_protein_signaling"
        reason.loc[
            component_count.ne(1) & ~result["ligand_is_complex"] & ~non_protein
        ] = "missing_or_multigene_ligand_symbol"
        result["eligible_ligand"] = reason.eq("eligible")
        result["exclusion_reason"] = reason
        return result.reset_index(drop=True)

    def run(
        self,
        adata: AnnData,
        *,
        nboot: int,
        seed: int,
        source: str,
        targets: list[str],
    ) -> pd.DataFrame:
        genes = [gene for gene in self.database_genes() if gene in adata.var_names]
        matrix = adata[:, genes].X
        if sp.issparse(matrix):
            matrix = matrix.toarray()
        prepared = AnnData(
            X=np.asarray(matrix, dtype=np.float32),
            obs=adata.obs[["benchmark_group"]].astype(str).copy(),
            var=pd.DataFrame(index=pd.Index(genes)),
        )
        target_groups = robjects.StrVector(targets)
        with localconverter(self.converter):
            result = robjects.r(_RUN_CELLCHAT_R)(
                prepared, nboot, seed, source, target_groups
            )
        return result.astype(
            {
                "source": "string",
                "target": "string",
                "interaction_name": "string",
                "prob": "float64",
                "pval": "float64",
            }
        ).reset_index(drop=True)
