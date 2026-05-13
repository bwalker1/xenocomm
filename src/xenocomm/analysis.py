import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp


def _get_sample(samples: Any, key: str) -> np.ndarray:
    """Get sample array from either dict or namedtuple format."""
    if isinstance(samples, dict):
        return samples[key]
    return getattr(samples, key).numpy()


def get_receptor_sensitivity(var_dict: dict[str, np.ndarray]) -> np.ndarray:
    """
    Extract receptor sensitivity from var_dict, handling both naming conventions.

    Old: gamma=(n_receptors,), beta=(n_receptors, n_targets)
    New: beta=(n_receptors,) in log-space, gamma=(n_receptors, n_targets)

    Returns:
        1D array of shape (n_receptors,)
    """
    beta_val = var_dict.get("beta")
    gamma_val = var_dict.get("gamma")

    if beta_val is not None and beta_val.ndim == 1:
        return np.exp(beta_val)
    elif gamma_val is not None and gamma_val.ndim == 1:
        return np.exp(gamma_val)
    else:
        raise ValueError("Cannot determine receptor sensitivity from var_dict")


def get_receptor_target_weights(var_dict: dict[str, np.ndarray]) -> np.ndarray:
    """
    Extract receptor-target weights matrix from var_dict, handling both naming conventions.

    Old: gamma=(n_receptors,), beta=(n_receptors, n_targets)
    New: beta=(n_receptors,) in log-space, gamma=(n_receptors, n_targets)

    Returns:
        2D array of shape (n_receptors, n_targets)
    """
    beta_val = var_dict.get("beta")
    gamma_val = var_dict.get("gamma")

    if gamma_val is not None and gamma_val.ndim == 2:
        return gamma_val**2
    elif beta_val is not None and beta_val.ndim == 2:
        return beta_val**2
    else:
        raise ValueError("Cannot determine receptor-target weights from var_dict")


def compute_receptor_species_fractions(
    samples: Any,
    mean_ligand: np.ndarray,
    ligand_receptor_matrix: np.ndarray,
    aggregate: str = "mean",
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-receptor fraction of binding input from each species.

    Returns:
        (mouse_fraction, human_fraction), each shape [n_receptors]
    """
    alpha_mouse = _get_sample(samples, "alpha_mouse")
    alpha_human = _get_sample(samples, "alpha_human")

    mouse_ligand = alpha_mouse * mean_ligand[0, :]
    human_ligand = alpha_human * mean_ligand[1, :]

    mouse_receptor = mouse_ligand @ ligand_receptor_matrix
    human_receptor = human_ligand @ ligand_receptor_matrix

    total = mouse_receptor + human_receptor
    total_safe = np.maximum(total, epsilon)
    mouse_frac = mouse_receptor / total_safe
    human_frac = human_receptor / total_safe

    if aggregate == "mean":
        return np.mean(mouse_frac, axis=0), np.mean(human_frac, axis=0)
    elif aggregate == "median":
        return np.median(mouse_frac, axis=0), np.median(human_frac, axis=0)
    return mouse_frac, human_frac


def compute_receptor_binding(
    samples: Any, aggregate: str | None = "mean"
) -> np.ndarray:
    binding = _get_sample(samples, "receptor_binding")
    if aggregate == "mean":
        return np.mean(binding, axis=0).squeeze()
    elif aggregate == "median":
        return np.median(binding, axis=0).squeeze()
    return binding


def compute_ligand_binding(
    var_dict: dict, ligand_receptor_matrix: Any, mean_ligand: Any, samples: Any
) -> np.ndarray:
    receptor_sensitivity = get_receptor_sensitivity(var_dict)
    m = ligand_receptor_matrix * receptor_sensitivity

    a = np.mean(_get_sample(samples, "alpha_human"), axis=0)
    w = a * mean_ligand
    total_ligand = np.sum(w, axis=0)

    ligand_binding = np.sum(total_ligand[:, None] * m, axis=1)
    return ligand_binding.flatten()


def create_binding_dataframe(
    ligands: list,
    ligand_binding: np.ndarray,
    mean_ligand: Any,
    total_expression: np.ndarray,
    sort_by: str = "binding",
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "ligand": ligands,
            "binding": ligand_binding,
            "expression": total_expression,
            "ratio": ligand_binding / total_expression,
            "mouse_expression": mean_ligand[0, :],
            "human_expression": mean_ligand[1, :],
            "ratio2": mean_ligand[1, :] / mean_ligand[0, :],
        },
        index=ligands,
    )

    df.sort_values(by=sort_by, ascending=False, inplace=True)
    return df


def add_expression_fractions(
    df: pd.DataFrame, min_expression: float = 0.0
) -> pd.DataFrame:
    total_expr = df["mouse_expression"] + df["human_expression"]
    if min_expression > 0:
        mask = total_expr >= min_expression
        df = df[mask].copy()
        total_expr = total_expr[mask]
    df["mouse_expression_fraction"] = df["mouse_expression"] / total_expr
    df["human_expression_fraction"] = df["human_expression"] / total_expr
    df["mouse_binding"] = df["mouse_expression_fraction"] * df["binding"]
    df["human_binding"] = df["human_expression_fraction"] * df["binding"]
    return df


def extract_target_genes(
    var_dict: dict, gene: str, receptors: list, targets: list, threshold: float = 0.001
) -> list[str]:
    weights = get_receptor_target_weights(var_dict)
    gene_idx = receptors.index(gene)
    targets_idx = np.argsort(-weights[gene_idx, :])
    vv = weights[gene_idx, targets_idx]
    target_genes = [str(targets[i]) for i in targets_idx if vv[i] > threshold]
    return target_genes


def create_target_dataframe(
    var_dict: dict,
    receptors: list,
    targets: list,
    threshold: float = 0.001,
    sort_by: str = "total_weight",
) -> pd.DataFrame:
    """Create DataFrame with target gene significance across all receptors."""
    weights = get_receptor_target_weights(var_dict)
    total_weight = np.sum(weights, axis=0)
    max_weight = np.max(weights, axis=0)

    df = pd.DataFrame(
        {
            "target": targets,
            "total_weight": total_weight,
            "max_weight": max_weight,
            "n_receptors": np.sum(weights > threshold, axis=0),
        }
    )
    df.sort_values(by=sort_by, ascending=False, inplace=True)
    return df


def create_receptor_dataframe(
    receptors: list, samples: Any, sort_by: str = "binding"
) -> pd.DataFrame:
    receptor_binding = compute_receptor_binding(samples)
    df = pd.DataFrame(
        {"receptor": receptors, "binding": receptor_binding}, index=receptors
    )
    df.sort_values(by=sort_by, ascending=False, inplace=True)
    return df


def create_receptor_fraction_dataframe(
    receptors: list,
    samples: Any,
    mean_ligand: np.ndarray,
    ligand_receptor_matrix: np.ndarray,
    sort_by: str = "human_fraction",
) -> pd.DataFrame:
    mouse_frac, human_frac = compute_receptor_species_fractions(
        samples, mean_ligand, ligand_receptor_matrix
    )
    df = pd.DataFrame(
        {
            "receptor": receptors,
            "mouse_fraction": mouse_frac,
            "human_fraction": human_frac,
        },
        index=receptors,
    )
    df.sort_values(by=sort_by, ascending=False, inplace=True)
    return df


def hill_function(x: np.ndarray) -> np.ndarray:
    """Hill function H(x) = x / (1 + x)"""
    return x / (1 + x)


def compute_species_decomposed_binding(
    samples: Any,
    mean_ligand: np.ndarray,
    ligand_receptor_matrix: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pre-Hill binding separately for mouse and human.

    Returns:
        (mouse_binding, human_binding, total_binding) each [n_samples, n_receptors]
    """
    alpha_mouse = _get_sample(samples, "alpha_mouse")
    alpha_human = _get_sample(samples, "alpha_human")
    receptor_rate = _get_sample(samples, "receptor_rate")

    mouse_ligand = alpha_mouse * mean_ligand[0, :]
    human_ligand = alpha_human * mean_ligand[1, :]

    mouse_receptor_base = mouse_ligand @ ligand_receptor_matrix
    human_receptor_base = human_ligand @ ligand_receptor_matrix

    mouse_binding = mouse_receptor_base * beta * receptor_rate
    human_binding = human_receptor_base * beta * receptor_rate
    total_binding = mouse_binding + human_binding

    return mouse_binding, human_binding, total_binding


def compute_global_marginal_activation(
    samples: Any,
    mean_ligand: np.ndarray,
    ligand_receptor_matrix: np.ndarray,
    beta: np.ndarray,
    aggregate: str | None = "mean",
) -> dict[str, Any]:
    """
    Compute global marginal: H(mouse+human) - H(mouse) summed over receptors.

    Returns dict with:
      - delta_global: total activation gain from human ligands
      - delta_per_receptor: [n_receptors] marginal per receptor
      - activation_mouse: H(mouse) per receptor
      - activation_total: H(mouse+human) per receptor
    """
    mouse_binding, _, total_binding = compute_species_decomposed_binding(
        samples, mean_ligand, ligand_receptor_matrix, beta
    )

    activation_mouse = hill_function(mouse_binding)
    activation_total = hill_function(total_binding)
    delta_per_receptor = activation_total - activation_mouse
    delta_global = np.sum(delta_per_receptor, axis=1)
    total_activation = np.sum(activation_total, axis=1)
    delta_global_relative = delta_global / (total_activation + 1e-8)

    if aggregate == "mean":
        return {
            "delta_global": np.mean(delta_global),
            "delta_global_std": np.std(delta_global),
            "delta_global_relative": np.mean(delta_global_relative),
            "delta_per_receptor": np.mean(delta_per_receptor, axis=0),
            "activation_mouse": np.mean(activation_mouse, axis=0),
            "activation_total": np.mean(activation_total, axis=0),
        }
    elif aggregate == "median":
        return {
            "delta_global": np.median(delta_global),
            "delta_global_std": np.std(delta_global),
            "delta_global_relative": np.median(delta_global_relative),
            "delta_per_receptor": np.median(delta_per_receptor, axis=0),
            "activation_mouse": np.median(activation_mouse, axis=0),
            "activation_total": np.median(activation_total, axis=0),
        }
    return {
        "delta_global": delta_global,
        "delta_global_relative": delta_global_relative,
        "delta_per_receptor": delta_per_receptor,
        "activation_mouse": activation_mouse,
        "activation_total": activation_total,
    }


def compute_ligand_counterfactual(
    samples: Any,
    mean_ligand: np.ndarray,
    ligand_receptor_matrix: np.ndarray,
    beta: np.ndarray,
    aggregate: str | None = "mean",
    species: str = "human",
) -> np.ndarray:
    """
    Per-ligand counterfactual: for each ligand l, compute
    Δ_l = Σ_r [H(total) - H(total - species_l)]

    Parameters:
        species: "human" or "mouse" - which species contribution to remove

    Returns: [n_ligands] or [n_samples, n_ligands] counterfactual contribution
    """
    alpha_human = _get_sample(samples, "alpha_human")
    alpha_mouse = _get_sample(samples, "alpha_mouse")
    receptor_rate = _get_sample(samples, "receptor_rate")

    _, _, total_binding = compute_species_decomposed_binding(
        samples, mean_ligand, ligand_receptor_matrix, beta
    )

    activation_total = hill_function(total_binding)
    n_samples, n_ligands = alpha_human.shape

    ligand_contribution = np.zeros((n_samples, n_ligands))
    for l in range(n_ligands):
        if species == "human":
            species_l_contrib = (
                alpha_human[:, l : l + 1]
                * mean_ligand[1, l]
                * ligand_receptor_matrix[l : l + 1, :]
                * beta
                * receptor_rate
            )
        else:  # mouse
            species_l_contrib = (
                alpha_mouse[:, l : l + 1]
                * mean_ligand[0, l]
                * ligand_receptor_matrix[l : l + 1, :]
                * beta
                * receptor_rate
            )
        binding_without_l = total_binding - species_l_contrib
        activation_without_l = hill_function(binding_without_l)
        ligand_contribution[:, l] = np.sum(
            activation_total - activation_without_l, axis=1
        )

    if aggregate == "mean":
        return np.mean(ligand_contribution, axis=0)
    elif aggregate == "median":
        return np.median(ligand_contribution, axis=0)
    return ligand_contribution


def compute_ligand_receptor_counterfactual(
    samples: Any,
    mean_ligand: np.ndarray,
    ligand_receptor_matrix: np.ndarray,
    beta: np.ndarray,
    aggregate: str | None = "mean",
    species: str = "human",
) -> np.ndarray:
    """
    Per-ligand, per-receptor counterfactual: for each ligand l and receptor r, compute
    Δ_{l,r} = H(total_r) - H(total_r - species_l_r)

    Parameters:
        species: "human" or "mouse" - which species contribution to remove
        aggregate: "mean", "median", or None

    Returns: [n_ligands, n_receptors] or [n_samples, n_ligands, n_receptors]
    """
    alpha_human = _get_sample(samples, "alpha_human")
    alpha_mouse = _get_sample(samples, "alpha_mouse")
    receptor_rate = _get_sample(samples, "receptor_rate")

    _, _, total_binding = compute_species_decomposed_binding(
        samples, mean_ligand, ligand_receptor_matrix, beta
    )

    activation_total = hill_function(total_binding)
    n_samples, n_ligands = alpha_human.shape
    n_receptors = total_binding.shape[1]

    ligand_receptor_contribution = np.zeros((n_samples, n_ligands, n_receptors))
    for l in range(n_ligands):
        if species == "human":
            species_l_contrib = (
                alpha_human[:, l : l + 1]
                * mean_ligand[1, l]
                * ligand_receptor_matrix[l : l + 1, :]
                * beta
                * receptor_rate
            )
        else:
            species_l_contrib = (
                alpha_mouse[:, l : l + 1]
                * mean_ligand[0, l]
                * ligand_receptor_matrix[l : l + 1, :]
                * beta
                * receptor_rate
            )
        binding_without_l = total_binding - species_l_contrib
        activation_without_l = hill_function(binding_without_l)
        ligand_receptor_contribution[:, l, :] = activation_total - activation_without_l

    if aggregate == "mean":
        return np.mean(ligand_receptor_contribution, axis=0)
    elif aggregate == "median":
        return np.median(ligand_receptor_contribution, axis=0)
    return ligand_receptor_contribution


def compute_enrichment_probability(
    samples: Any,
    mean_ligand: np.ndarray,
    ligand_receptor_matrix: np.ndarray,
    beta: np.ndarray,
    threshold: float = 0.1,
) -> np.ndarray:
    """Compute P(counterfactual contribution > threshold) for each ligand."""
    """
    Compute P(counterfactual contribution > threshold) for each ligand.

    Returns: [n_ligands] posterior probability of enrichment
    """
    ligand_contribution = compute_ligand_counterfactual(
        samples,
        mean_ligand,
        ligand_receptor_matrix,
        beta,
        aggregate=None,
        species="human",
    )
    return np.mean(ligand_contribution > threshold, axis=0)


def compute_human_bias_statistics(
    samples: Any,
    mean_ligand: np.ndarray,
    ligand_receptor_matrix: np.ndarray,
    beta: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """
    Compute statistics for identifying human-biased ligands using posterior samples.

    For each posterior sample, computes human_fraction = Δ_human / (Δ_human + Δ_mouse).
    Then derives:
      - prob_human_dominant: P(human_fraction > threshold) across samples
      - human_fraction_mean: Mean human_fraction
      - human_fraction_std: Std of human_fraction
      - human_fraction_ci_lower: 2.5th percentile (lower bound of 95% CI)
      - human_fraction_ci_upper: 97.5th percentile (upper bound of 95% CI)

    Returns: dict with arrays of shape [n_ligands]
    """
    cf_human = compute_ligand_counterfactual(
        samples,
        mean_ligand,
        ligand_receptor_matrix,
        beta,
        aggregate=None,
        species="human",
    )
    cf_mouse = compute_ligand_counterfactual(
        samples,
        mean_ligand,
        ligand_receptor_matrix,
        beta,
        aggregate=None,
        species="mouse",
    )

    cf_total = cf_human + cf_mouse
    human_fraction_samples = cf_human / (cf_total + 1e-8)

    return {
        "prob_human_dominant": np.mean(human_fraction_samples > threshold, axis=0),
        "human_fraction_mean": np.mean(human_fraction_samples, axis=0),
        "human_fraction_std": np.std(human_fraction_samples, axis=0),
        "human_fraction_ci_lower": np.percentile(human_fraction_samples, 2.5, axis=0),
        "human_fraction_ci_upper": np.percentile(human_fraction_samples, 97.5, axis=0),
    }


def create_marginal_enrichment_dataframe(
    ligands: list,
    samples: Any,
    mean_ligand: np.ndarray,
    ligand_receptor_matrix: np.ndarray,
    beta: np.ndarray,
    threshold: float = 0.1,
    human_dominant_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Create comprehensive DataFrame with marginal enrichment measures.

    Columns:
      - ligand: ligand name
      - counterfactual_human: Δ_human_l (activation loss from removing human ligand l)
      - counterfactual_mouse: Δ_mouse_l (activation loss from removing mouse ligand l)
      - counterfactual_total: Δ_human_l + Δ_mouse_l
      - human_fraction: Δ_human_l / (Δ_human_l + Δ_mouse_l)
      - human_fraction_ci_lower: 2.5th percentile of human_fraction
      - prob_human_dominant: P(human_fraction > human_dominant_threshold)
      - counterfactual_std: uncertainty across posterior samples
      - prob_enriched: P(Δ_l > threshold)
      - human_expression, mouse_expression
      - expression_ratio
    """
    counterfactual_human = compute_ligand_counterfactual(
        samples,
        mean_ligand,
        ligand_receptor_matrix,
        beta,
        aggregate="mean",
        species="human",
    )
    counterfactual_mouse = compute_ligand_counterfactual(
        samples,
        mean_ligand,
        ligand_receptor_matrix,
        beta,
        aggregate="mean",
        species="mouse",
    )
    full_samples = compute_ligand_counterfactual(
        samples,
        mean_ligand,
        ligand_receptor_matrix,
        beta,
        aggregate=None,
        species="human",
    )
    prob_enriched = compute_enrichment_probability(
        samples, mean_ligand, ligand_receptor_matrix, beta, threshold
    )
    bias_stats = compute_human_bias_statistics(
        samples, mean_ligand, ligand_receptor_matrix, beta, human_dominant_threshold
    )

    counterfactual_total = counterfactual_human + counterfactual_mouse
    human_fraction = counterfactual_human / (counterfactual_total + 1e-8)

    df = pd.DataFrame(
        {
            "ligand": ligands,
            "counterfactual_human": counterfactual_human,
            "counterfactual_mouse": counterfactual_mouse,
            "counterfactual_total": counterfactual_total,
            "human_fraction": human_fraction,
            "human_fraction_ci_lower": bias_stats["human_fraction_ci_lower"],
            "prob_human_dominant": bias_stats["prob_human_dominant"],
            "counterfactual_std": np.std(full_samples, axis=0),
            "prob_enriched": prob_enriched,
            "human_expression": mean_ligand[1, :],
            "mouse_expression": mean_ligand[0, :],
        }
    )
    df["expression_ratio"] = df["human_expression"] / (df["mouse_expression"] + 1e-8)
    df.sort_values("counterfactual_human", ascending=False, inplace=True)

    return df


def find_var(adata, key: str) -> int | None:
    """Case-insensitive var lookup. Returns the index, or None if missing."""
    if key in adata.var_names:
        return list(adata.var_names).index(key)
    upper_map = {v.upper(): i for i, v in enumerate(adata.var_names)}
    return upper_map.get(key.upper())


def pct_expressed(adata, gene: str) -> float:
    """Fraction of cells with non-zero expression of `gene` (case-insensitive)."""
    idx = find_var(adata, gene)
    if idx is None:
        return 0.0
    col = adata.X[:, idx]
    if sp.issparse(col):
        n_nonzero = int(col.astype(bool).sum())
    else:
        n_nonzero = int(np.count_nonzero(np.asarray(col)))
    return float(n_nonzero / adata.n_obs)


def gene_expression_array(adata, gene: str) -> np.ndarray:
    """1D dense array of per-cell expression for `gene` (case-insensitive)."""
    idx = find_var(adata, gene)
    if idx is None:
        return np.zeros(adata.n_obs, dtype=np.float32)
    col = adata.X[:, idx]
    if sp.issparse(col):
        return np.asarray(col.toarray()).ravel()
    return np.asarray(col).ravel()


def species_bias_df(model, samples, var_dict) -> pd.DataFrame:
    """Per-ligand binding + species bias.

    Columns: ligand, binding, mouse_binding, human_binding,
    mouse_expression, human_expression, mouse_expression_fraction,
    human_expression_fraction, species_bias (= 2 * human_fraction - 1).
    """
    ligand_binding = compute_ligand_binding(
        var_dict,
        model.ligand_receptor_matrix_np,
        model.mean_ligand_np,
        samples,
    )
    total_expression = np.sum(model.mean_ligand_np, axis=0)
    df = create_binding_dataframe(
        model.ligands,
        ligand_binding,
        model.mean_ligand_np,
        total_expression,
    )
    total_expr = df["mouse_expression"] + df["human_expression"]
    min_expr = float(np.percentile(total_expr, 25))
    df = add_expression_fractions(df, min_expression=min_expr)
    df["species_bias"] = df["human_expression_fraction"] * 2 - 1
    return df.reset_index(drop=True)


def enrichment_df(model, samples, var_dict) -> pd.DataFrame:
    """Per-ligand enrichment scores.

    Columns: ligand, enrichment (counterfactual_human), prob_enriched,
    human_fraction, human_expression, mouse_expression.
    """
    beta = get_receptor_sensitivity(var_dict)
    df = create_marginal_enrichment_dataframe(
        model.ligands,
        samples,
        model.mean_ligand_np,
        model.ligand_receptor_matrix_np,
        beta,
    )
    return pd.DataFrame(
        {
            "ligand": df["ligand"].values,
            "enrichment": df["counterfactual_human"].values,
            "prob_enriched": df["prob_enriched"].values,
            "human_fraction": df["human_fraction"].values,
            "human_expression": df["human_expression"].values,
            "mouse_expression": df["mouse_expression"].values,
        }
    )


def receptor_marginal_df(model, samples, var_dict) -> pd.DataFrame:
    """Per-receptor marginal activation (delta from human + baseline mouse).

    Columns: receptor, delta, mouse.
    """
    beta = get_receptor_sensitivity(var_dict)
    result = compute_global_marginal_activation(
        samples,
        model.mean_ligand_np,
        model.ligand_receptor_matrix_np,
        beta,
    )
    return pd.DataFrame(
        {
            "receptor": list(model.receptors),
            "delta": [float(v) for v in result["delta_per_receptor"]],
            "mouse": [float(v) for v in result["activation_mouse"]],
        }
    )


def targets_marginal_df(model, samples, var_dict, top_n: int = 10) -> pd.DataFrame:
    """Top-N downstream targets by total weighted activation.

    Columns: target, value.
    """
    beta = get_receptor_sensitivity(var_dict)
    result = compute_global_marginal_activation(
        samples,
        model.mean_ligand_np,
        model.ligand_receptor_matrix_np,
        beta,
    )
    weights = get_receptor_target_weights(var_dict)
    total_per_target = (
        result["delta_per_receptor"] + result["activation_mouse"]
    ) @ weights
    order = np.argsort(-total_per_target)[:top_n]
    return pd.DataFrame(
        {
            "target": [model.targets[i] for i in order],
            "value": [float(total_per_target[i]) for i in order],
        }
    )


def species_dotplot_data(
    model,
    samples,
    var_dict,
    adata_mouse,
    adata_human,
    *,
    ligands: list[str] | None = None,
    prob_threshold: float = 0.9,
    human_fraction_threshold: float = 0.8,
    exclude_ligands: tuple[str, ...] = ("Farp2", "Fstl5", "Il1rapl1"),
) -> dict[str, list]:
    """Long-format dot-plot bindings for a ligand set across species.

    If `ligands` is provided, uses that list (in the given order).
    Otherwise filters to ligands with prob_enriched >= prob_threshold
    and human_fraction > human_fraction_threshold, minus excluded
    names, and sorts by enrichment desc.

    Returns four parallel lists matching the SpeciesDotPlot contract
    (genes, species, size, color).
    """
    if ligands is not None:
        selected = [
            name
            for name in ligands
            if find_var(adata_mouse, name) is not None
            or find_var(adata_human, name) is not None
            or name in model.ligands
        ]
    else:
        enrich = enrichment_df(model, samples, var_dict)
        excluded = {name.lower() for name in exclude_ligands}
        mask = (
            (enrich["prob_enriched"] >= prob_threshold)
            & (enrich["human_fraction"] > human_fraction_threshold)
            & (~enrich["ligand"].str.lower().isin(excluded))
        )
        keep = enrich[mask].sort_values("enrichment", ascending=False)
        selected = keep["ligand"].tolist()

    ligand_index = {name.lower(): i for i, name in enumerate(model.ligands)}
    mean_ligand = model.mean_ligand_np

    genes: list[str] = []
    species: list[str] = []
    size: list[float] = []
    color: list[float] = []
    for lig in selected:
        i = ligand_index.get(lig.lower())
        if i is None:
            continue
        canonical = model.ligands[i]
        for sp_name, adata, row_idx in (
            ("Human", adata_human, 1),
            ("Mouse", adata_mouse, 0),
        ):
            genes.append(canonical)
            species.append(sp_name)
            size.append(pct_expressed(adata, canonical))
            color.append(float(mean_ligand[row_idx, i]))
    return {"genes": genes, "species": species, "size": size, "color": color}


# ---------------------------------------------------------------------------
# Ensemble helpers (fig5)
# ---------------------------------------------------------------------------


@dataclass
class StagedModel:
    """Lightweight stand-in for XenocommModel holding only what figures need.

    Produced by `load_staged_model` from a `.staged.npz` artifact — the
    xenocomm training pipeline (or the v1 API on first access) writes one
    of these alongside each trained model to skip TF on repeated loads.
    """

    ligands: list[str]
    receptors: list[str]
    targets: list[str]
    mean_ligand_np: np.ndarray
    ligand_receptor_matrix_np: np.ndarray
    receptor_target_matrix_np: np.ndarray


def load_staged_model(path: str | Path) -> tuple[StagedModel, dict, dict]:
    """Load a `<model>.staged.npz` → (StagedModel, samples, var_dict).

    Unlike `XenocommModel.load`, this doesn't pull in TensorFlow and
    doesn't need adata — the staged blob carries pre-sampled posterior
    draws and the var_dict parameter snapshot directly.
    """
    data = np.load(str(path), allow_pickle=True)
    model = StagedModel(
        ligands=data["ligands"].tolist(),
        receptors=data["receptors"].tolist(),
        targets=data["targets"].tolist(),
        mean_ligand_np=data["mean_ligand_np"],
        ligand_receptor_matrix_np=data["ligand_receptor_matrix_np"],
        receptor_target_matrix_np=data["receptor_target_matrix_np"],
    )
    samples = {k[2:]: data[k] for k in data.files if k.startswith("s_")}
    var_dict = {k[2:]: data[k] for k in data.files if k.startswith("v_")}
    return model, samples, var_dict


def _staged_sibling(model_path: Path) -> Path:
    """The `.staged.npz` path that lives next to a raw `.npz` model."""
    suffix = "".join(model_path.suffixes)
    return model_path.with_name(model_path.name.removesuffix(suffix) + ".staged.npz")


def _save_staged(staged_path: Path, model, samples: dict, var_dict: dict) -> None:
    payload: dict[str, np.ndarray] = {
        "ligands": np.array(list(model.ligands), dtype=object),
        "receptors": np.array(list(model.receptors), dtype=object),
        "targets": np.array(list(model.targets), dtype=object),
        "mean_ligand_np": np.asarray(model.mean_ligand_np),
        "ligand_receptor_matrix_np": np.asarray(model.ligand_receptor_matrix_np),
        "receptor_target_matrix_np": np.asarray(model.receptor_target_matrix_np),
    }
    for k, v in samples.items():
        payload[f"s_{k}"] = np.asarray(v)
    for k, v in var_dict.items():
        payload[f"v_{k}"] = np.asarray(v)
    np.savez(str(staged_path), **payload)


def stage_and_load_model(
    model_path: str | Path,
    adata_mouse,
    adata_human,
    *,
    n_samples: int = 100,
) -> tuple[StagedModel, dict, dict]:
    """Load a raw `.npz` via XenocommModel, sample, and write a staged sibling.

    Returns a StagedModel-shaped tuple (same output as `load_staged_model`).
    If the staged sibling already exists, uses it directly. Otherwise loads
    with full TF machinery, draws `n_samples` posterior samples, best-effort
    writes `<model>.staged.npz`, and returns the same result.
    """
    model_path = Path(model_path)
    staged = _staged_sibling(model_path)
    if staged.exists():
        return load_staged_model(staged)

    from .model import XenocommModel

    model = XenocommModel.load(str(model_path), adata_mouse, adata_human)
    samples_raw = model.sample(n_samples)
    samples = {k: np.asarray(v) for k, v in samples_raw.items()}
    var_dict_raw = model.get_parameters()
    var_dict = {k: np.asarray(v) for k, v in var_dict_raw.items()}

    # Read-only directory, permission issue, etc. — non-fatal.
    with contextlib.suppress(OSError):
        _save_staged(staged, model, samples, var_dict)

    staged_model = StagedModel(
        ligands=list(model.ligands),
        receptors=list(model.receptors),
        targets=list(model.targets),
        mean_ligand_np=np.asarray(model.mean_ligand_np),
        ligand_receptor_matrix_np=np.asarray(model.ligand_receptor_matrix_np),
        receptor_target_matrix_np=np.asarray(model.receptor_target_matrix_np),
    )
    return staged_model, samples, var_dict


def load_ensemble(
    models_dir: str | Path,
    rates: list[float],
    n_trials: int,
    *,
    stem: str = "mix",
    adata_mouse=None,
    adata_human=None,
) -> dict[float, dict[int, tuple[StagedModel, dict, dict]]]:
    """Load a (rate × trial) grid of models from a directory.

    For each `(rate, trial)`, prefers `{stem}_{rate}_trial_{trial}.staged.npz`
    (fast, no TF). Falls back to `{stem}_{rate}_trial_{trial}.npz` via
    `stage_and_load_model` when adata is provided (first-time staging cost,
    cached thereafter). Missing trials are silently skipped.
    """
    root = Path(models_dir)
    ensemble: dict[float, dict[int, tuple[StagedModel, dict, dict]]] = {}
    for rate in rates:
        rate_dict: dict[int, tuple[StagedModel, dict, dict]] = {}
        for trial in range(n_trials):
            staged = root / f"{stem}_{rate}_trial_{trial}.staged.npz"
            raw = root / f"{stem}_{rate}_trial_{trial}.npz"
            if staged.exists():
                rate_dict[trial] = load_staged_model(staged)
            elif raw.exists() and adata_mouse is not None and adata_human is not None:
                rate_dict[trial] = stage_and_load_model(raw, adata_mouse, adata_human)
        if rate_dict:
            ensemble[rate] = rate_dict
    return ensemble


def ensemble_rate_sweep_df(
    ensemble: dict[float, dict[int, tuple[StagedModel, dict, dict]]],
    metric: str,
    *,
    enrichment_threshold: float = 0.01,
    enrichment_human_fraction: float = 0.9,
) -> pd.DataFrame:
    """Long-format rate-sweep DataFrame.

    Columns: name, rate, trial, value.

    `metric` is one of:

    - ``"binding"``: per-receptor posterior-mean receptor binding.
    - ``"delta_h"``: per-ligand counterfactual_human with the human
      half of `mean_ligand` scaled by `rate`.
    - ``"enrichment_count"`` / ``"enrichment_sum"``: number / sum of
      ligands exceeding `(counterfactual_human ≥ enrichment_threshold)
      & (human_fraction ≥ enrichment_human_fraction)` in the
      rate-scaled marginal enrichment DataFrame.
    """
    names: list[str] = []
    rates_col: list[float] = []
    trials_col: list[int] = []
    values: list[float] = []

    for rate, trials in ensemble.items():
        for trial_idx, (model, samples, var_dict) in trials.items():
            if metric == "binding":
                binding = compute_receptor_binding(samples)
                for i, name in enumerate(model.receptors):
                    names.append(name)
                    rates_col.append(rate)
                    trials_col.append(trial_idx)
                    values.append(float(binding[i]))
            elif metric in ("delta_h", "enrichment_count", "enrichment_sum"):
                beta = get_receptor_sensitivity(var_dict)
                rate_mean_ligand = model.mean_ligand_np.copy()
                rate_mean_ligand[1] = rate * rate_mean_ligand[1]
                df = create_marginal_enrichment_dataframe(
                    model.ligands,
                    samples,
                    rate_mean_ligand,
                    model.ligand_receptor_matrix_np,
                    beta,
                )
                if metric == "delta_h":
                    for _, row in df.iterrows():
                        names.append(row["ligand"])
                        rates_col.append(rate)
                        trials_col.append(trial_idx)
                        values.append(float(row["counterfactual_human"]))
                else:
                    enriched = df[
                        (df["counterfactual_human"] >= enrichment_threshold)
                        & (df["human_fraction"] >= enrichment_human_fraction)
                    ]
                    if metric == "enrichment_count":
                        names.append("enriched_count")
                        rates_col.append(rate)
                        trials_col.append(trial_idx)
                        values.append(float(len(enriched)))
                    else:
                        names.append("total_enrichment")
                        rates_col.append(rate)
                        trials_col.append(trial_idx)
                        values.append(float(enriched["counterfactual_human"].sum()))
            else:
                raise ValueError(f"unknown metric: {metric!r}")

    return pd.DataFrame(
        {
            "name": names,
            "rate": rates_col,
            "trial": trials_col,
            "value": values,
        }
    )


def ensemble_parameter_distribution_df(
    ensemble: dict[float, dict[int, tuple[StagedModel, dict, dict]]],
    param: str,
    *,
    normalize: str | None = None,
) -> pd.DataFrame:
    """Long-format parameter-distribution DataFrame.

    Columns: value, rate (rate as string for categorical grouping).

    `param` is ``"alpha"`` (from var_dict, per-ligand) or
    ``"beta"`` (1-D beta / gamma in log-space, per-receptor).
    `normalize` is ``None`` or ``"minmax"`` (per-rate rescale to
    [0, 1]).
    """
    all_values: list[float] = []
    all_rates: list[str] = []
    rate_boundaries: list[tuple[int, int]] = []

    for rate, trials in ensemble.items():
        rate_label = str(rate)
        start = len(all_values)
        for _trial, (_model, _samples, var_dict) in trials.items():
            if param == "alpha":
                vals = var_dict["alpha"]
            elif param == "beta":
                key = (
                    "beta"
                    if "beta" in var_dict and var_dict["beta"].ndim == 1
                    else "gamma"
                )
                vals = var_dict[key]
            else:
                raise ValueError(f"unknown param: {param!r}")
            for v in np.ravel(vals):
                all_values.append(float(v))
                all_rates.append(rate_label)
        rate_boundaries.append((start, len(all_values)))

    if normalize == "minmax":
        for start, end in rate_boundaries:
            chunk = all_values[start:end]
            if not chunk:
                continue
            v_min = min(chunk)
            v_max = max(chunk)
            rng = v_max - v_min
            if rng > 0:
                all_values[start:end] = [(v - v_min) / rng for v in chunk]
            else:
                all_values[start:end] = [0.0] * len(chunk)

    return pd.DataFrame({"value": all_values, "rate": all_rates})


# ---------------------------------------------------------------------------
# Spatial / commot helpers (fig4)
# ---------------------------------------------------------------------------


def _normalize_01(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def _normalize_expr(arr: np.ndarray, min_median: float = 0.5) -> np.ndarray:
    """Percentile-based normalization with a min median for non-zero cells.

    Clips at the 99th percentile of non-zero values, then boosts so the
    median of non-zero cells is at least `min_median`.
    """
    nz = arr[arr > 0]
    if len(nz) == 0:
        return np.zeros_like(arr, dtype=float)
    vmax = float(np.percentile(nz, 99))
    if vmax < 1e-12:
        return np.zeros_like(arr, dtype=float)
    result = np.clip(arr / vmax, 0.0, 1.0)
    if min_median > 0:
        median = float(np.median(result[result > 0]))
        if median < min_median:
            result = np.clip(result * (min_median / median), 0.0, 1.0)
    return result


def _balance_dual_channels(
    a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Scale two arrays so their 95th-percentile non-zero values match."""
    nz_a = a[a > 0]
    nz_b = b[b > 0]
    p95_a = float(np.percentile(nz_a, 95)) if len(nz_a) else 1.0
    p95_b = float(np.percentile(nz_b, 95)) if len(nz_b) else 1.0
    if p95_a < 1e-12 or p95_b < 1e-12:
        return a, b
    target = (p95_a + p95_b) / 2
    return (
        np.clip(a * (target / p95_a), 0.0, 1.0),
        np.clip(b * (target / p95_b), 0.0, 1.0),
    )


def _apply_bbox(adata, bbox: tuple[float, float, float, float] | None):
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    x = coords[:, 0]
    y = coords[:, 1]
    if bbox is None:
        return adata, x, y, None
    xmin, xmax, ymin, ymax = bbox
    mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    return adata[mask], x[mask], y[mask], mask


def _spot_diameter(adata) -> float | None:
    spatial_uns = adata.uns.get("spatial")
    if not isinstance(spatial_uns, dict) or len(spatial_uns) != 1:
        return None
    key_name = next(iter(spatial_uns))
    sf = spatial_uns[key_name].get("scalefactors", {})
    if "spot_diameter_fullres" in sf:
        return float(sf["spot_diameter_fullres"])
    return None


def spatial_gene_panel_data(
    adata,
    gene: str,
    gene2: str | None = None,
    *,
    bbox: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Build a spatial-panel dict (`x`, `y`, `values`, `type`, plus optional
    `values2`, `spotDiameter`). Handles dual-channel gene pairs with
    percentile normalization + p95 channel balancing matching the v1
    `/python/spatial` endpoint output.
    """
    sub, x, y, _ = _apply_bbox(adata, bbox)
    expr = gene_expression_array(sub, gene)
    result: dict[str, Any] = {
        "x": x.tolist(),
        "y": (-y).tolist(),
        "values": _normalize_expr(expr).tolist(),
        "type": "continuous",
    }
    if gene2:
        expr2 = gene_expression_array(sub, gene2)
        normed, normed2 = _balance_dual_channels(
            _normalize_expr(expr), _normalize_expr(expr2)
        )
        result["values"] = normed.tolist()
        result["values2"] = normed2.tolist()
    diameter = _spot_diameter(sub)
    if diameter is not None:
        result["spotDiameter"] = diameter
    return result


def spatial_obs_panel_data(
    adata,
    obs_key: str,
    *,
    bbox: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Spatial panel sourced from an obs column. Categorical columns return
    string labels; numeric ones return min-max-normalized floats."""
    sub, x, y, _ = _apply_bbox(adata, bbox)
    col = sub.obs[obs_key]
    result: dict[str, Any] = {"x": x.tolist(), "y": (-y).tolist()}
    diameter = _spot_diameter(sub)
    if diameter is not None:
        result["spotDiameter"] = diameter
    if col.dtype.name == "category" or col.dtype == object:
        result["values"] = col.astype(str).tolist()
        result["type"] = "categorical"
    else:
        vals = np.asarray(col, dtype=np.float64)
        result["values"] = _normalize_01(vals).tolist()
        result["type"] = "continuous"
    return result


def commot_pathway_sums(
    adata, *, summary: str = "sender", database: str = "xenocomm"
) -> dict[str, float]:
    """Per-pathway total of the `commot-{database}-sum-{summary}` obsm
    DataFrame. Used to rank ligands' pathway activity for the
    enrichment-vs-commot scatter.
    """
    key = f"commot-{database}-sum-{summary}"
    df = adata.obsm[key]
    sums: dict[str, float] = {}
    for col in df.columns:
        pathway = col[2:].split("-")[0]
        sums[pathway] = sums.get(pathway, 0.0) + float(np.sum(df[col].values))
    return sums


def enrichment_vs_commot_df(
    model,
    samples,
    var_dict,
    commot_adata,
    *,
    database: str = "xenocomm",
    summary: str = "sender",
    exclude_ligands: tuple[str, ...] = ("Farp2", "Fstl5", "Il1rapl1"),
    enriched_prob_threshold: float = 0.9,
    enriched_human_fraction_threshold: float = 0.8,
) -> pd.DataFrame:
    """DataFrame pairing per-ligand enrichment with commot pathway activity.

    Columns: ligand, enrichment, commot_activity, category ("Enriched" or
    "Other"), percentile (position of the activity in the sorted pathway
    sums, 0–100), prob_enriched, human_fraction. Ligand name matching to
    pathway names is case-insensitive.
    """
    from bisect import bisect_left

    df = enrichment_df(model, samples, var_dict)
    pathway_sums = commot_pathway_sums(commot_adata, summary=summary, database=database)
    upper_to_pathway = {p.upper(): p for p in pathway_sums}
    sorted_activities = sorted(pathway_sums.values())
    n_pathways = len(sorted_activities)
    exclude = set(exclude_ligands)

    rows: list[dict] = []
    for _, row in df.iterrows():
        lig = row["ligand"]
        prob_e = float(row["prob_enriched"])
        hf = float(row["human_fraction"])
        activity = pathway_sums.get(upper_to_pathway.get(lig.upper()), 0.0)
        enriched = (
            prob_e >= enriched_prob_threshold
            and hf > enriched_human_fraction_threshold
            and lig not in exclude
        )
        rank = bisect_left(sorted_activities, activity)
        pct = 100.0 * rank / n_pathways if n_pathways > 0 else 0.0
        rows.append(
            {
                "ligand": lig,
                "enrichment": float(row["enrichment"]),
                "commot_activity": activity,
                "category": "Enriched" if enriched else "Other",
                "percentile": round(pct, 1),
                "prob_enriched": prob_e,
                "human_fraction": hf,
            }
        )
    return pd.DataFrame(rows)


def commot_quiver_data(
    adata,
    pathway: str,
    *,
    summary: str = "sender",
    database: str = "xenocomm",
    k: int = 5,
    n_grid: int = 25,
    bandwidth: float = 1.5,
    bbox: tuple[float, float, float, float] | None = None,
    include_histology: bool = True,
) -> dict[str, Any]:
    """Turn a COMMOT pathway obsp matrix into QuiverPlot bindings.

    Per-cell direction vectors are computed as the displacement-weighted
    mean of the top-k targets (largest values in each cell's obsp row).
    Grid points are placed over the bbox (or the full adata if no bbox);
    directions + magnitudes are smoothed with a Gaussian bandwidth of
    `bandwidth * grid_spacing`. Grid points whose nearest real cell is
    more than 3× the median cell spacing are dropped ("off-tissue").

    Returns a dict shaped for the v2 QuiverPlot contract: x, y, u, v,
    magnitude, bg_x, bg_y, bg_values, plus optional spot_diameter and
    histology (base64-encoded JPEG + extents).
    """
    from scipy.spatial import cKDTree

    if "spatial" not in adata.obsm:
        raise ValueError("commot adata missing obsm['spatial']")

    key = f"commot-{database}-{pathway}"
    if key not in adata.obsp:
        matches = [k2 for k2 in adata.obsp if k2.lower() == key.lower()]
        if not matches:
            raise ValueError(f"pathway {pathway!r} missing obsp key {key!r}")
        key = matches[0]

    mat = adata.obsp[key]
    mat = sp.csr_matrix(mat) if not sp.issparse(mat) else mat.tocsr()
    if summary == "receiver":
        mat = mat.T.tocsr()

    coords = adata.obsm["spatial"].astype(np.float64)
    n_cells = coords.shape[0]

    directions = np.zeros((n_cells, 2), dtype=np.float64)
    magnitudes = np.zeros(n_cells, dtype=np.float64)

    for i in range(n_cells):
        row = mat.getrow(i)
        if row.nnz == 0:
            continue
        cols = row.indices
        vals = row.data.astype(np.float64)
        top_idx = (
            np.argpartition(vals, -k)[-k:] if len(vals) > k else np.arange(len(vals))
        )
        target_coords = coords[cols[top_idx]]
        weights = vals[top_idx]
        displacements = target_coords - coords[i]
        direction = (displacements * weights[:, None]).sum(axis=0)
        mag = float(np.linalg.norm(direction))
        if mag > 1e-12:
            directions[i] = direction / mag
        magnitudes[i] = mag

    mag_max = magnitudes.max()
    bg_values = magnitudes / mag_max if mag_max > 1e-12 else magnitudes

    if bbox is not None:
        bx0, bx1, by0, by1 = bbox
        bg_mask = (
            (coords[:, 0] >= bx0)
            & (coords[:, 0] <= bx1)
            & (coords[:, 1] >= by0)
            & (coords[:, 1] <= by1)
        )
    else:
        bg_mask = np.ones(n_cells, dtype=bool)
        bx0 = bx1 = by0 = by1 = None

    bg_coords = coords[bg_mask]
    bg_vals = bg_values[bg_mask]

    x_min, x_max = bg_coords[:, 0].min(), bg_coords[:, 0].max()
    y_min, y_max = bg_coords[:, 1].min(), bg_coords[:, 1].max()
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, n_grid),
        np.linspace(y_min, y_max, n_grid),
    )
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    bg_tree = cKDTree(bg_coords)
    k_nn = min(15, len(bg_coords))
    bg_dists, bg_idxs = bg_tree.query(grid_pts, k=k_nn)

    cell_nn_dists, _ = bg_tree.query(bg_coords, k=2)
    median_cell_spacing = float(np.median(cell_nn_dists[:, 1]))
    dist_cutoff = median_cell_spacing * 3.0

    grid_spacing = max((x_max - x_min), (y_max - y_min)) / n_grid
    sigma = bandwidth * grid_spacing

    full_idx = np.where(bg_mask)[0]
    grid_dirs = np.zeros((len(grid_pts), 2), dtype=np.float64)
    grid_mags = np.zeros(len(grid_pts), dtype=np.float64)
    for gi in range(len(grid_pts)):
        w = np.exp(-0.5 * (bg_dists[gi] / sigma) ** 2)
        w_sum = w.sum()
        if w_sum < 1e-12:
            continue
        w /= w_sum
        cell_idx = full_idx[bg_idxs[gi]]
        grid_dirs[gi] = (
            directions[cell_idx] * magnitudes[cell_idx, None] * w[:, None]
        ).sum(axis=0)
        grid_mags[gi] = (magnitudes[cell_idx] * w).sum()

    norms = np.maximum(np.linalg.norm(grid_dirs, axis=1, keepdims=True), 1e-12)
    grid_dirs = grid_dirs / norms

    on_tissue = bg_dists[:, 0] <= dist_cutoff
    arrow_x = grid_pts[on_tissue, 0]
    arrow_y = grid_pts[on_tissue, 1]
    arrow_dx = grid_dirs[on_tissue, 0]
    arrow_dy = grid_dirs[on_tissue, 1]
    arrow_mag = grid_mags[on_tissue]

    am_max = arrow_mag.max() if len(arrow_mag) else 1.0
    if am_max > 1e-12:
        arrow_mag = arrow_mag / am_max

    out: dict[str, Any] = {
        "bg_x": bg_coords[:, 0].tolist(),
        "bg_y": (-bg_coords[:, 1]).tolist(),
        "bg_values": bg_vals.tolist(),
        "x": arrow_x.tolist(),
        "y": (-arrow_y).tolist(),
        "u": arrow_dx.tolist(),
        "v": (-arrow_dy).tolist(),
        "magnitude": arrow_mag.tolist(),
    }

    if include_histology:
        histology = _extract_histology(adata, bx0, bx1, by0, by1)
        if histology is not None:
            out["histology"] = histology

    return out


def _extract_histology(
    adata,
    bx0: float | None,
    bx1: float | None,
    by0: float | None,
    by1: float | None,
) -> dict[str, Any] | None:
    import base64
    import io

    spatial_uns = adata.uns.get("spatial", {})
    if not isinstance(spatial_uns, dict):
        return None
    for _lib_id, lib_data in spatial_uns.items():
        images = lib_data.get("images", {})
        scalefactors = lib_data.get("scalefactors", {})
        img = images.get("hires")
        scalef = scalefactors.get("tissue_hires_scalef", 1.0)
        if img is None:
            img = images.get("lowres")
            scalef = scalefactors.get("tissue_lowres_scalef", 1.0)
        if img is None:
            continue
        img_arr = np.array(img)
        if img_arr.dtype in (np.float32, np.float64):
            img_arr = np.clip(img_arr * 255, 0, 255).astype(np.uint8)
        origin_x = scalefactors.get("image_origin_x", 0.0)
        origin_y = scalefactors.get("image_origin_y", 0.0)
        ext_x0 = origin_x
        ext_y0 = origin_y
        ext_x1 = origin_x + img_arr.shape[1] / scalef
        ext_y1 = origin_y + img_arr.shape[0] / scalef
        if bx0 is not None:
            px0 = max(0, int((bx0 - origin_x) * scalef))
            px1 = min(img_arr.shape[1], int(np.ceil((bx1 - origin_x) * scalef)))
            py0 = max(0, int((by0 - origin_y) * scalef))
            py1 = min(img_arr.shape[0], int(np.ceil((by1 - origin_y) * scalef)))
            img_arr = img_arr[py0:py1, px0:px1]
            ext_x0, ext_x1 = bx0, bx1
            ext_y0, ext_y1 = by0, by1
        try:
            from PIL import Image
        except ImportError:
            return None
        buf = io.BytesIO()
        Image.fromarray(img_arr).save(buf, format="JPEG", quality=85)
        return {
            "data": base64.b64encode(buf.getvalue()).decode("ascii"),
            "extent_x": [ext_x0, ext_x1],
            "extent_y": [-ext_y1, -ext_y0],
        }
    return None


def targets_per_receptor_counts(var_dict: dict, threshold: float = 0.01) -> np.ndarray:
    """Per-receptor count of targets with gamma² ≥ `threshold`."""
    weights = get_receptor_target_weights(var_dict)
    return np.sum(weights >= threshold, axis=1)


def ligand_receptor_sankey_data(
    model,
    samples,
    var_dict,
    *,
    ligands: list[str] | None = None,
    top_n: int | None = None,
    enriched: bool = False,
    enriched_prob_threshold: float = 0.9,
    enriched_human_fraction_threshold: float = 0.8,
    enriched_exclude: tuple[str, ...] = ("Farp2", "Fstl5", "Il1rapl1"),
    n_top_receptors: int = 5,
) -> dict[str, list[dict]]:
    """Build sankey {nodes, links} for ligand→receptor signaling.

    Resolves the ligand set in this order:
      - `top_n`: top-N ligands by aggregate binding (species_bias_df order)
      - `enriched`: filter by prob_enriched / human_fraction (excluding
        `enriched_exclude`) and sort by counterfactual_human
      - `ligands`: explicit list (validated against model.ligands)

    For each ligand, keeps its top `n_top_receptors` connected receptors
    ranked by `|Δ_human_lr|` (compute_ligand_receptor_counterfactual).
    Returns a dict shaped to the SankeyPlot v2 contract: `{nodes: [...],
    links: [...]}` — each node has `id`, `label`, `type`; each link has
    `source`, `target`, `value` (absolute Δ_lr).
    """
    import math

    if top_n is not None:
        df = species_bias_df(model, samples, var_dict)
        ligands = (
            df.sort_values("binding", ascending=False).head(top_n)["ligand"].tolist()
        )
    elif enriched:
        enrich = enrichment_df(model, samples, var_dict)
        excluded = {name.lower() for name in enriched_exclude}
        mask = (
            (enrich["prob_enriched"] >= enriched_prob_threshold)
            & (enrich["human_fraction"] > enriched_human_fraction_threshold)
            & (~enrich["ligand"].str.lower().isin(excluded))
        )
        ligands = (
            enrich[mask].sort_values("enrichment", ascending=False)["ligand"].tolist()
        )
    elif ligands is None:
        ligands = []

    beta = get_receptor_sensitivity(var_dict)
    lr_matrix = model.ligand_receptor_matrix_np
    delta_h = compute_ligand_receptor_counterfactual(
        samples, model.mean_ligand_np, lr_matrix, beta
    )

    nodes_by_id: dict[str, dict] = {}
    links: list[dict] = []

    ligand_index = {name: i for i, name in enumerate(model.ligands)}
    for lig in ligands:
        i = ligand_index.get(lig)
        if i is None:
            continue
        lig_id = f"L:{lig}"
        nodes_by_id[lig_id] = {"id": lig_id, "label": lig, "type": "ligand"}
        connected_mask = lr_matrix[i, :] == 1
        connected_indices = np.where(connected_mask)[0]
        if len(connected_indices) == 0:
            continue
        receptor_delta = delta_h[i, :]
        top_local = np.argsort(-receptor_delta[connected_indices])[:n_top_receptors]
        for r_idx in connected_indices[top_local]:
            rname = model.receptors[r_idx]
            val = abs(float(receptor_delta[r_idx]))
            if val <= 0 or math.isnan(val) or math.isinf(val):
                continue
            r_id = f"R:{rname}"
            nodes_by_id.setdefault(
                r_id, {"id": r_id, "label": rname, "type": "receptor"}
            )
            links.append({"source": lig_id, "target": r_id, "value": val})

    return {"nodes": list(nodes_by_id.values()), "links": links}


def receptor_counterfactual_df(
    model,
    samples,
    var_dict,
    *,
    remove_ligands: list[str],
    receptors: list[str] | None = None,
    label: str = "",
) -> pd.DataFrame:
    """Baseline + Δ receptor activation when `remove_ligands` are removed.

    Long-format columns: receptor, component, activation. Two rows per
    receptor: one "Baseline" (activation without the removed ligands'
    human contribution), one "+<LABEL>" (delta added back by their
    contribution). `label` defaults to the joined remove_ligands list.

    If `receptors` is None, uses every receptor connected to any of the
    removed ligands in the LR matrix.
    """
    beta = get_receptor_sensitivity(var_dict)
    lr_matrix = model.ligand_receptor_matrix_np

    _, _, total_binding = compute_species_decomposed_binding(
        samples, model.mean_ligand_np, lr_matrix, beta
    )

    if not receptors:
        connected: set[str] = set()
        for lig in remove_ligands:
            if lig in model.ligands:
                li = model.ligands.index(lig)
                for ri in np.where(lr_matrix[li, :] == 1)[0]:
                    connected.add(model.receptors[ri])
        receptors = sorted(connected)

    alpha_human = _get_sample(samples, "alpha_human")
    receptor_rate = _get_sample(samples, "receptor_rate")

    removed_binding = np.zeros_like(total_binding)
    for lig in remove_ligands:
        if lig not in model.ligands:
            continue
        li = model.ligands.index(lig)
        contrib = (
            alpha_human[:, li : li + 1]
            * model.mean_ligand_np[1, li]
            * lr_matrix[li : li + 1, :]
            * beta
            * receptor_rate
        )
        removed_binding += contrib

    binding_without = total_binding - removed_binding
    activation_total = hill_function(np.mean(total_binding, axis=0))
    activation_without = hill_function(np.mean(binding_without, axis=0))

    contrib_label = label or ", ".join(remove_ligands)
    rows: list[dict] = []
    for rname in receptors:
        if rname not in model.receptors:
            continue
        ri = model.receptors.index(rname)
        baseline = float(activation_without[ri])
        delta = float(activation_total[ri] - activation_without[ri])
        rows.append(
            {"receptor": rname, "component": "Baseline", "activation": baseline}
        )
        rows.append(
            {
                "receptor": rname,
                "component": f"+{contrib_label.upper()}",
                "activation": delta,
            }
        )
    return pd.DataFrame(rows)
