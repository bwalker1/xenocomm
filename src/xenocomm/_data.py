import hashlib
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_lr_rtg_cache: dict = {}


def load_lr_rtg_pairs(
    adata_human,
    adata_mouse,
    max_targets=None,
    cutoff=-10,
    ligands=None,
    receptors=None,
    targets=None,
):
    """
    Load and filter ligand-receptor and receptor-target-gene pairs.

    Parameters
    ----------
    adata_human : AnnData
        Human gene expression data
    adata_mouse : AnnData
        Mouse gene expression data (must have dispersions_norm in .var)
    max_targets : int, optional
        Maximum number of target genes
    cutoff : float
        Dispersion cutoff for gene filtering
    ligands : list, optional
        Pre-determined ligand list (skip filtering if provided)
    receptors : list, optional
        Pre-determined receptor list (skip filtering if provided)
    targets : list, optional
        Pre-determined target list (skip filtering if provided)

    Returns
    -------
    lr : list of tuples
        Filtered ligand-receptor pairs
    rtg : list of tuples
        Filtered receptor-target-gene pairs
    ligands : list
        Final ligand gene names
    receptors : list
        Final receptor gene names
    targets : list
        Final target gene names
    """
    print("load_lr_rtg_pairs")

    from ._download import ensure_database

    db_dir = ensure_database()
    cache_dir = Path.cwd() / ".xenocomm"
    cache_dir.mkdir(exist_ok=True)

    if ligands is not None and receptors is not None and targets is not None:
        use_cache = False
    else:
        human_genes = sorted(adata_human.var_names)
        mouse_genes = sorted(adata_mouse.var_names)
        genes_str = f"{','.join(human_genes)}|{','.join(mouse_genes)}"
        params_str = f"|max_targets={max_targets}|cutoff={cutoff}"
        hash_key = hashlib.sha256((genes_str + params_str).encode()).hexdigest()
        cache_file = cache_dir / f"lr_rtg_{hash_key}.pkl"
        use_cache = True

    if use_cache and hash_key in _lr_rtg_cache:
        return _lr_rtg_cache[hash_key]

    if use_cache and cache_file.exists():
        print(f"Loading cached LR/RTG pairs from {cache_file}")
        with open(cache_file, "rb") as f:
            result = pickle.load(f)
        _lr_rtg_cache[hash_key] = result
        return result

    print("Loading LR/RTG pairs from database...")
    lr_path = db_dir / "mouse_lr.parquet"
    rtg_path = db_dir / "mouse_rtg.parquet"

    print("  Loading parquet files...")
    LR = pd.read_parquet(lr_path)
    RTG = pd.read_parquet(rtg_path)
    print(f"    Loaded: {len(LR)} LR pairs, {len(RTG)} RTG pairs")

    print("  Filtering to genes present in data...")
    human_gene_set = set(adata_human.var_names)
    mouse_gene_set = set(adata_mouse.var_names)

    lr_filtered = LR[
        LR["from"].isin(human_gene_set)
        & LR["from"].isin(mouse_gene_set)
        & LR["to"].isin(mouse_gene_set)
    ]
    rtg_filtered = RTG[
        RTG["from"].isin(mouse_gene_set) & RTG["to"].isin(mouse_gene_set)
    ]

    lr = [tuple(row) for row in lr_filtered.values]
    rtg = [tuple(row) for row in rtg_filtered.values]

    print(f"    Filtered: {len(lr)} LR pairs, {len(rtg)} RTG pairs")

    if ligands is None or receptors is None or targets is None:
        print("  Applying dispersion and expression filtering...")
        d = adata_mouse.var["dispersions_norm"]
        m = pd.Series(
            np.array(adata_mouse.X.mean(axis=0)).squeeze(),
            index=adata_mouse.var_names,
        )

        receptors_filtered = sorted(
            {v for _, v in lr if v in adata_mouse.var_names and m[v] > 0.05}
        )

        ligands_filtered = sorted(
            {
                v
                for v, r in lr
                if v in adata_mouse.var_names
                and r in receptors_filtered
                and d[v] > cutoff
            }
        )

        lr = [
            (a, b) for a, b in lr if a in ligands_filtered and b in receptors_filtered
        ]
        rtg = [
            (r, tg)
            for r, tg in rtg
            if r in receptors_filtered
            and tg in adata_mouse.var_names
            and tg not in receptors_filtered
            and tg not in ligands_filtered
            and d[tg] > cutoff
        ]

        targets_filtered = sorted({tg for r, tg in rtg})
        target_receptors = {r for r, tg in rtg}
        lr_receptors = {r for l, r in lr}
        receptors_filtered = sorted(target_receptors & lr_receptors)
        ligands_filtered = sorted({l for l, r in lr if r in receptors_filtered})

        n_targets = len(targets_filtered)

        if max_targets is not None and n_targets > max_targets:
            print(f"  Limiting targets from {n_targets} to {max_targets}...")
            targets_set = _filter_targets_preserving_connectivity(
                rtg=rtg,
                receptors=set(receptors_filtered),
                targets=set(targets_filtered),
                dispersions=d,
                max_targets=max_targets,
                min_targets_per_receptor=3,
            )
            targets_filtered = sorted(targets_set)

        ligands = sorted(ligands_filtered)
        receptors = sorted(receptors_filtered)
        targets = sorted(targets_filtered)

    lr = [(a, b) for a, b in lr if a in ligands and b in receptors]
    rtg = [(r, t) for r, t in rtg if r in receptors and t in targets]

    print(
        f"  Final: {len(ligands)} ligands, {len(receptors)} receptors, {len(targets)} targets"
    )
    print(f"         {len(lr)} LR pairs, {len(rtg)} RTG pairs")

    result = (lr, rtg, ligands, receptors, targets)

    if use_cache:
        print(f"Caching results to {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
        _lr_rtg_cache[hash_key] = result

    return result


def _filter_targets_preserving_connectivity(
    rtg,
    receptors,
    targets,
    dispersions,
    max_targets=2000,
    min_targets_per_receptor=3,
):
    receptor_to_targets = defaultdict(set)
    for r, tg in rtg:
        if r in receptors and tg in targets:
            receptor_to_targets[r].add(tg)

    selected_targets = set()
    for receptor in sorted(receptors):
        receptor_targets = receptor_to_targets[receptor]
        sorted_targets = sorted(receptor_targets, key=lambda t: -dispersions[t])
        n_to_allocate = min(min_targets_per_receptor, len(sorted_targets))
        selected_targets.update(sorted_targets[:n_to_allocate])

    if len(selected_targets) > max_targets:
        if min_targets_per_receptor <= 1:
            raise ValueError(
                f"Cannot satisfy max_targets={max_targets}: even with "
                f"min_targets_per_receptor=1, {len(selected_targets)} targets are needed"
            )
        return _filter_targets_preserving_connectivity(
            rtg,
            receptors,
            targets,
            dispersions,
            max_targets,
            min_targets_per_receptor=min_targets_per_receptor - 1,
        )

    remaining_budget = max_targets - len(selected_targets)
    if remaining_budget > 0:
        candidate_targets = set()
        for r, tg in rtg:
            if r in receptors and tg in targets and tg not in selected_targets:
                candidate_targets.add(tg)

        ranked_candidates = sorted(candidate_targets, key=lambda t: -dispersions[t])
        selected_targets.update(ranked_candidates[:remaining_budget])

    for receptor in receptors:
        receptor_selected = [
            tg for tg in receptor_to_targets[receptor] if tg in selected_targets
        ]
        if len(receptor_selected) == 0:
            raise ValueError(f"Receptor {receptor} has no targets after filtering")

    assert len(selected_targets) == max_targets

    return selected_targets
