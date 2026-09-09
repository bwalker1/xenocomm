from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def _sample(samples: Any, key: str) -> np.ndarray:
    return np.asarray(samples[key], dtype=float)


def _counterfactuals(samples, mean_ligand, ligand_receptor_matrix, sensitivity):
    alpha_mouse = _sample(samples, "alpha_mouse")
    alpha_human = _sample(samples, "alpha_human")
    receptor_rate = _sample(samples, "receptor_rate")
    mouse_ligand = alpha_mouse * mean_ligand[0]
    human_ligand = alpha_human * mean_ligand[1]
    receptor_multiplier = receptor_rate * sensitivity
    total_binding = (
        mouse_ligand @ ligand_receptor_matrix + human_ligand @ ligand_receptor_matrix
    ) * receptor_multiplier
    total_activation = total_binding / (1 + total_binding)
    delta_h = np.empty_like(alpha_human)
    delta_mouse = np.empty_like(alpha_mouse)
    for index, edges in enumerate(ligand_receptor_matrix):
        human_contribution = human_ligand[:, index, None] * edges * receptor_multiplier
        mouse_contribution = mouse_ligand[:, index, None] * edges * receptor_multiplier
        without_human = total_binding - human_contribution
        without_mouse = total_binding - mouse_contribution
        delta_h[:, index] = np.sum(
            total_activation - without_human / (1 + without_human), axis=1
        )
        delta_mouse[:, index] = np.sum(
            total_activation - without_mouse / (1 + without_mouse), axis=1
        )
    return delta_h, delta_mouse


def _summary(values: np.ndarray, ci_probability: float) -> dict[str, np.ndarray]:
    tail = (1 - ci_probability) / 2
    return {
        "mean": np.mean(values, axis=0),
        "median": np.median(values, axis=0),
        "sd": np.std(values, axis=0),
        "ci_lower": np.quantile(values, tail, axis=0),
        "ci_upper": np.quantile(values, 1 - tail, axis=0),
    }


def ligand_result_table(
    ligands: Sequence[str],
    samples: Any,
    mean_ligand: np.ndarray,
    ligand_receptor_matrix: np.ndarray,
    receptor_sensitivity: np.ndarray,
    *,
    delta_h_threshold: float = 0.1,
    human_fraction_threshold: float = 0.8,
    call_probability_threshold: float = 0.9,
    ci_probability: float = 0.95,
) -> pd.DataFrame:
    mean_ligand = np.asarray(mean_ligand, dtype=float)
    ligand_receptor_matrix = np.asarray(ligand_receptor_matrix, dtype=float)
    receptor_sensitivity = np.asarray(receptor_sensitivity, dtype=float)
    delta_h, delta_mouse = _counterfactuals(
        samples, mean_ligand, ligand_receptor_matrix, receptor_sensitivity
    )
    delta_h_summary = _summary(delta_h, ci_probability)
    delta_mouse_summary = _summary(delta_mouse, ci_probability)
    delta_total = delta_h + delta_mouse
    human_fraction = np.divide(
        delta_h,
        delta_total,
        out=np.zeros_like(delta_h),
        where=delta_total > 0,
    )
    human_fraction_summary = _summary(human_fraction, ci_probability)

    alpha_mouse = _sample(samples, "alpha_mouse")
    alpha_human = _sample(samples, "alpha_human")
    binding_scale = ligand_receptor_matrix @ receptor_sensitivity
    mouse_binding = np.mean(alpha_mouse * mean_ligand[0] * binding_scale, axis=0)
    human_binding = np.mean(alpha_human * mean_ligand[1] * binding_scale, axis=0)
    prob_delta_h = np.mean(delta_h > delta_h_threshold, axis=0)
    prob_human_fraction = np.mean(human_fraction > human_fraction_threshold, axis=0)
    called = (prob_delta_h >= call_probability_threshold) & (
        human_fraction_summary["mean"] > human_fraction_threshold
    )
    result = pd.DataFrame(
        {
            "ligand": [str(ligand) for ligand in ligands],
            "delta_h_mean": delta_h_summary["mean"],
            "delta_h_median": delta_h_summary["median"],
            "delta_h_sd": delta_h_summary["sd"],
            "delta_h_ci_lower": delta_h_summary["ci_lower"],
            "delta_h_ci_upper": delta_h_summary["ci_upper"],
            "prob_delta_h_gt_threshold": prob_delta_h,
            "delta_mouse_mean": delta_mouse_summary["mean"],
            "delta_total_mean": np.mean(delta_total, axis=0),
            "human_fraction_mean": human_fraction_summary["mean"],
            "human_fraction_sd": human_fraction_summary["sd"],
            "human_fraction_ci_lower": human_fraction_summary["ci_lower"],
            "human_fraction_ci_upper": human_fraction_summary["ci_upper"],
            "prob_human_fraction_gt_threshold": prob_human_fraction,
            "mouse_binding_mean": mouse_binding,
            "human_binding_mean": human_binding,
            "total_binding_mean": mouse_binding + human_binding,
            "mouse_expression_mean": mean_ligand[0],
            "human_expression_mean": mean_ligand[1],
            "delta_h_threshold": delta_h_threshold,
            "human_fraction_threshold": human_fraction_threshold,
            "call_probability_threshold": call_probability_threshold,
            "called": called,
        }
    )
    result.sort_values(
        ["delta_h_mean", "ligand"],
        ascending=[False, True],
        kind="mergesort",
        inplace=True,
    )
    result.reset_index(drop=True, inplace=True)
    result.insert(1, "rank", np.arange(1, len(result) + 1))
    return result
