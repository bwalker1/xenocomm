from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.sparse as sp


def get_receptor_sensitivity(var_dict: dict[str, np.ndarray]) -> np.ndarray:
    return np.exp(var_dict["beta"])


def _receptor_target_weights(var_dict: dict[str, np.ndarray]) -> np.ndarray:
    weights = np.square(var_dict["gamma"])
    return weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-6, 1)


def _hill(x: np.ndarray) -> np.ndarray:
    return x / (1 + x)


def _species_binding(model, samples, beta):
    alpha_mouse = samples["alpha_mouse"]
    alpha_human = samples["alpha_human"]
    receptor_rate = samples["receptor_rate"]
    mouse = (
        (alpha_mouse * model.mean_ligand_np[0])
        @ model.ligand_receptor_matrix_np
        * beta
        * receptor_rate
    )
    human = (
        (alpha_human * model.mean_ligand_np[1])
        @ model.ligand_receptor_matrix_np
        * beta
        * receptor_rate
    )
    return mouse, human, mouse + human


def _global_activation(model, samples, var_dict):
    beta = get_receptor_sensitivity(var_dict)
    mouse, _, total = _species_binding(model, samples, beta)
    activation_mouse = _hill(mouse)
    delta = _hill(total) - activation_mouse
    return np.mean(delta, axis=0), np.mean(activation_mouse, axis=0)


def species_bias_df(model, samples, var_dict) -> pd.DataFrame:
    beta = get_receptor_sensitivity(var_dict)
    alpha_mouse = np.mean(samples["alpha_mouse"], axis=0)
    alpha_human = np.mean(samples["alpha_human"], axis=0)
    expression = model.mean_ligand_np
    binding_scale = model.ligand_receptor_matrix_np @ beta
    mouse_binding = alpha_mouse * expression[0] * binding_scale
    human_binding = alpha_human * expression[1] * binding_scale
    df = pd.DataFrame(
        {
            "ligand": model.ligands,
            "binding": mouse_binding + human_binding,
            "mouse_binding": mouse_binding,
            "human_binding": human_binding,
            "mouse_expression": expression[0],
            "human_expression": expression[1],
        }
    ).sort_values("binding", ascending=False)
    total = df["mouse_expression"] + df["human_expression"]
    df = df[total >= np.percentile(total, 25)].copy()
    return df[["ligand", "mouse_binding", "human_binding"]].reset_index(drop=True)


def receptor_marginal_df(model, samples, var_dict) -> pd.DataFrame:
    delta, mouse = _global_activation(model, samples, var_dict)
    return pd.DataFrame({"receptor": model.receptors, "delta": delta, "mouse": mouse})


def targets_marginal_df(model, samples, var_dict, top_n: int = 10) -> pd.DataFrame:
    delta, mouse = _global_activation(model, samples, var_dict)
    values = (delta + mouse) @ _receptor_target_weights(var_dict)
    order = np.argsort(-values)[:top_n]
    return pd.DataFrame(
        {
            "target": [model.targets[index] for index in order],
            "value": values[order],
        }
    )


def _find_gene(adata, gene: str) -> int:
    lookup = {str(name).upper(): index for index, name in enumerate(adata.var_names)}
    return lookup[gene.upper()]


def _fraction_expressed(adata, gene: str) -> float:
    index = _find_gene(adata, gene)
    values = adata.X[:, index]
    nonzero = (
        values.astype(bool).sum() if sp.issparse(values) else np.count_nonzero(values)
    )
    return float(nonzero / adata.n_obs)


def species_dotplot_data(
    model,
    adata_mouse,
    adata_human,
    *,
    ligands: list[str],
) -> dict[str, list]:
    ligand_index = {name.lower(): index for index, name in enumerate(model.ligands)}
    result: dict[str, list] = {"genes": [], "species": [], "size": [], "color": []}
    for ligand in ligands:
        index = ligand_index[ligand.lower()]
        canonical = model.ligands[index]
        for species, adata, row, expression_gene in (
            ("Human", adata_human, 1, model.human_ligands[index]),
            ("Mouse", adata_mouse, 0, canonical),
        ):
            result["genes"].append(canonical)
            result["species"].append(species)
            result["size"].append(_fraction_expressed(adata, expression_gene))
            result["color"].append(float(model.mean_ligand_np[row, index]))
    return result


def load_staged_model(path: str | Path):
    with np.load(path, allow_pickle=False) as data:
        model = SimpleNamespace(
            ligands=data["ligands"].tolist(),
            receptors=data["receptors"].tolist(),
            targets=data["targets"].tolist(),
            human_ligands=data["human_ligands"].tolist(),
            mean_ligand_np=np.array(data["mean_ligand_np"]),
            ligand_receptor_matrix_np=np.array(data["ligand_receptor_matrix_np"]),
        )
        samples = {
            key.removeprefix("s_"): np.array(data[key])
            for key in data.files
            if key.startswith("s_")
        }
        var_dict = {
            key.removeprefix("v_"): np.array(data[key])
            for key in data.files
            if key.startswith("v_")
        }
    return model, samples, var_dict


def receptor_counterfactual_df(
    model,
    samples,
    var_dict,
    *,
    remove_ligands: list[str],
    label: str = "",
) -> pd.DataFrame:
    beta = get_receptor_sensitivity(var_dict)
    _, _, total = _species_binding(model, samples, beta)
    alpha_human = samples["alpha_human"]
    receptor_rate = samples["receptor_rate"]
    removed = np.zeros_like(total)
    connected: set[str] = set()
    for ligand in remove_ligands:
        ligand_index = model.ligands.index(ligand)
        receptor_indices = np.where(model.ligand_receptor_matrix_np[ligand_index] == 1)[
            0
        ]
        connected.update(model.receptors[index] for index in receptor_indices)
        removed += (
            alpha_human[:, ligand_index : ligand_index + 1]
            * model.mean_ligand_np[1, ligand_index]
            * model.ligand_receptor_matrix_np[ligand_index : ligand_index + 1]
            * beta
            * receptor_rate
        )
    activation_total = _hill(np.mean(total, axis=0))
    activation_without = _hill(np.mean(total - removed, axis=0))
    rows = []
    for receptor in sorted(connected):
        index = model.receptors.index(receptor)
        rows.extend(
            [
                {
                    "receptor": receptor,
                    "component": "Baseline",
                    "activation": float(activation_without[index]),
                },
                {
                    "receptor": receptor,
                    "component": f"+{(label or ', '.join(remove_ligands)).upper()}",
                    "activation": float(
                        activation_total[index] - activation_without[index]
                    ),
                },
            ]
        )
    return pd.DataFrame(rows)
