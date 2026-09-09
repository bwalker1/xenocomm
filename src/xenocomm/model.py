import math
import os
import random
from typing import Any

import anndata as ad
import numpy as np
import scipy.sparse as sp


def _import_tf():
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_op_determinism()
    tf.experimental.numpy.experimental_enable_numpy_behavior()

    import tensorflow_probability as tfp
    from tensorflow_probability import bijectors as tfb
    from tensorflow_probability import distributions as tfd

    return tf, tfp, tfb, tfd


def _seed_pair(seed: int, stream: int) -> tuple[int, int]:
    modulus = 2**31 - 1
    return int(seed % modulus), int(stream % modulus)


def _gene_indices(adata: ad.AnnData | ad.Raw, genes: tuple[str, ...]) -> list[int]:
    lookup = {str(name).casefold(): index for index, name in enumerate(adata.var_names)}
    return [lookup[gene.casefold()] for gene in genes]


def _dense_float32(matrix: Any) -> np.ndarray:
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _raw_matrix(adata: ad.AnnData) -> sp.csr_matrix:
    if adata.raw is not None:
        values = adata.raw.X
    elif "counts" in adata.layers:
        values = adata.layers["counts"]
    else:
        raise ValueError("Raw counts are required in .raw or layers['counts']")
    return sp.csr_matrix(values)


def _depth_standardized_cloglog_mean(
    counts: sp.csr_matrix, ligand_indices: list[int], reference_depth: float
) -> np.ndarray:
    median_depth = float(np.median(np.asarray(counts.sum(axis=1)).ravel()))
    detected = counts[:, ligand_indices].tocsr().astype(np.float64)
    detected.data = 1.0 - np.exp(-detected.data * reference_depth / median_depth)
    prevalence = np.asarray(detected.mean(axis=0)).ravel()
    return -np.log(np.maximum(1.0 - prevalence, 1e-8))


def compute_ligand_abundance(
    adata_mouse: ad.AnnData,
    adata_human: ad.AnnData,
    ligands: tuple[str, ...] | list[str],
    human_ligands: tuple[str, ...] | list[str],
    ligand_receptor_matrix: np.ndarray,
    *,
    ligand_reference_depth: float = 25_000.0,
    ligand_degree_power: float = 0.25,
) -> np.ndarray:
    ligands = tuple(ligands)
    human_ligands = tuple(human_ligands)
    mouse_ligand_indices = _gene_indices(
        adata_mouse.raw if adata_mouse.raw is not None else adata_mouse, ligands
    )
    human_ligand_indices = _gene_indices(
        adata_human.raw if adata_human.raw is not None else adata_human, human_ligands
    )
    mouse_counts = _raw_matrix(adata_mouse)
    human_counts = _raw_matrix(adata_human)
    abundance = np.stack(
        [
            np.clip(
                _depth_standardized_cloglog_mean(
                    mouse_counts,
                    mouse_ligand_indices,
                    ligand_reference_depth,
                ),
                1e-3,
                None,
            ),
            np.clip(
                _depth_standardized_cloglog_mean(
                    human_counts,
                    human_ligand_indices,
                    ligand_reference_depth,
                ),
                1e-3,
                None,
            ),
        ]
    ).astype(np.float32)
    ligand_degree = np.maximum(np.asarray(ligand_receptor_matrix).sum(axis=1), 1.0)
    abundance /= ligand_degree[None, :] ** ligand_degree_power
    return abundance


class XenocommModel:
    def __init__(
        self,
        adata_mouse: ad.AnnData,
        ligands: tuple[str, ...] | list[str],
        receptors: tuple[str, ...] | list[str],
        targets: tuple[str, ...] | list[str],
        ligand_receptor_matrix: np.ndarray,
        receptor_target_matrix: np.ndarray,
        mean_ligand: np.ndarray,
        human_ligands: tuple[str, ...] | list[str] | None = None,
        *,
        receptor_target_mode: str = "learned",
        training_seed: int = 0,
        posterior_seed: int = 1,
        batch_size: int = 1024,
        steps_per_batch: int = 250,
        epochs: int = 1,
        learning_rate: float = 1e-3,
        gamma_l2: float = 1e-3,
        combined_receptor_gain_l2: float = 0.1,
        combined_receptor_gain_cap: float = 1.0,
        receptor_gain_cap: float = 0.5,
    ):
        self.ligands = tuple(str(value) for value in ligands)
        if human_ligands is None:
            human_ligands = self.ligands
        self.human_ligands = tuple(str(value) for value in human_ligands)
        self.receptors = tuple(str(value) for value in receptors)
        self.targets = tuple(str(value) for value in targets)
        self.ligand_receptor_matrix_np = np.asarray(
            ligand_receptor_matrix, dtype=np.float32
        )
        self.receptor_target_matrix_np = np.asarray(
            receptor_target_matrix, dtype=np.float32
        )
        self.mean_ligand_np = np.asarray(mean_ligand, dtype=np.float32)
        if receptor_target_mode not in {"learned", "fixed"}:
            raise ValueError("receptor_target_mode must be learned or fixed")
        if self.receptor_target_matrix_np.shape != (
            len(self.receptors),
            len(self.targets),
        ):
            raise ValueError("receptor_target_matrix has an incompatible shape")
        if np.any(self.receptor_target_matrix_np.sum(axis=1) == 0):
            raise ValueError("Every receptor must have a database target")

        self.receptor_target_mode = receptor_target_mode
        self.training_seed = training_seed
        self.posterior_seed = posterior_seed
        self.batch_size = batch_size
        self.steps_per_batch = steps_per_batch
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.gamma_l2 = gamma_l2
        self.combined_receptor_gain_l2 = combined_receptor_gain_l2
        self.combined_receptor_gain_cap = combined_receptor_gain_cap
        self.receptor_gain_cap = receptor_gain_cap

        receptor_indices = _gene_indices(adata_mouse, self.receptors)
        target_indices = _gene_indices(adata_mouse, self.targets)

        self._X_receptors = _dense_float32(adata_mouse.X[:, receptor_indices])
        self._X_targets = _dense_float32(adata_mouse.X[:, target_indices])
        self._initial_mu_target = np.log(
            np.mean(self._X_targets, axis=0) + 1e-6
        ).astype(np.float32)
        self._initial_mu_receptor = np.zeros(
            self._X_receptors.shape[1], dtype=np.float32
        )
        self._initial_sigma_target = np.ones(self._X_targets.shape[1], dtype=np.float32)

        self._model: Any | None = None
        self._surrogate_posterior: Any | None = None
        self._var_dict: dict[str, Any] = {}

    def build_model(self) -> None:
        random.seed(self.training_seed)
        np.random.seed(self.training_seed)
        tf, tfp, tfb, tfd = _import_tf()
        tf.keras.utils.set_random_seed(self.training_seed)
        n_ligands = len(self.ligands)
        n_receptors = len(self.receptors)
        n_targets = len(self.targets)

        ligand_receptor_matrix = tf.constant(
            self.ligand_receptor_matrix_np, dtype=tf.float32
        )
        receptor_target_matrix = tf.constant(
            self.receptor_target_matrix_np, dtype=tf.float32
        )
        mean_ligand = tf.Variable(
            self.mean_ligand_np,
            dtype=tf.float32,
            trainable=False,
            name="mean_ligand",
        )
        sigma_bijector = tfb.Chain([tfb.Exp(), tfb.Shift(shift=1e-2)])

        mu_receptor = tfp.util.TransformedVariable(
            tf.constant(self._initial_mu_receptor, dtype=tf.float32),
            tfb.Identity(),
            dtype=tf.float32,
            trainable=True,
            name="mu_receptor",
        )
        sigma_receptor = tfp.util.TransformedVariable(
            tf.ones([n_receptors], dtype=tf.float32),
            sigma_bijector,
            trainable=True,
            name="sigma_receptor",
        )
        mu_target = tfp.util.TransformedVariable(
            tf.constant(self._initial_mu_target, dtype=tf.float32),
            tfb.Identity(),
            trainable=True,
            name="mu_target",
        )
        sigma_target = tfp.util.TransformedVariable(
            tf.constant(self._initial_sigma_target, dtype=tf.float32),
            sigma_bijector,
            trainable=True,
            name="sigma_target",
        )
        alpha = tfp.util.TransformedVariable(
            tf.ones(n_ligands, dtype=tf.float32),
            bijector=tfb.Softplus(),
            name="alpha",
            trainable=True,
        )
        sigma_alpha = tfp.util.TransformedVariable(
            2.0 * tf.ones([n_ligands], dtype=tf.float32),
            tfb.Softplus(),
            name="sigma_alpha",
            trainable=True,
        )
        if self.receptor_target_mode == "learned":
            gamma_initial = (
                tf.random.stateless_normal(
                    (n_receptors, n_targets),
                    seed=_seed_pair(self.training_seed, 1),
                    stddev=0.1,
                )
                * receptor_target_matrix
            )
        else:
            edge_count = self.receptor_target_matrix_np.sum(axis=1, keepdims=True)
            gamma_initial = tf.constant(
                self.receptor_target_matrix_np / np.sqrt(edge_count),
                dtype=tf.float32,
            )
        gamma = tf.Variable(
            gamma_initial,
            name="gamma",
            trainable=self.receptor_target_mode == "learned",
        )
        beta = tfp.util.TransformedVariable(
            tf.ones([n_receptors], dtype=tf.float32),
            tfb.Exp(),
            name="beta",
            trainable=True,
        )
        receptor_gain_cap = tf.constant(self.receptor_gain_cap, dtype=tf.float32)

        def receptor_binding_map(receptor_rate, alpha_mouse, alpha_human, beta_value):
            alpha_stacked = tf.stack([alpha_mouse, alpha_human], axis=0)
            alpha_combined = tf.math.reduce_sum(
                mean_ligand * alpha_stacked, axis=0, keepdims=True
            )
            source_drive = tf.squeeze(tf.matmul(alpha_combined, ligand_receptor_matrix))
            receptor_expression = tf.clip_by_value(receptor_rate, 1e-6, 100)
            receptor_gain = beta_value * receptor_expression
            receptor_gain = receptor_gain_cap * tf.math.tanh(
                receptor_gain / receptor_gain_cap
            )
            receptor_binding = source_drive * receptor_gain
            receptor_binding = tf.reshape(receptor_binding, [1, -1])
            return receptor_binding / (1 + receptor_binding)

        def target_map(receptor_binding, gamma_value, mu_target_value):
            gamma_squared = tf.square(gamma_value) * receptor_target_matrix
            denominator = tf.reduce_sum(gamma_squared, axis=1, keepdims=True)
            effective_gamma = gamma_squared / tf.clip_by_value(denominator, 1e-6, 1.0)
            return (
                tf.matmul(receptor_binding, effective_gamma).flatten() + mu_target_value
            )

        @tfd.JointDistributionCoroutineAutoBatched
        def joint_model():
            alpha_human = yield tfd.Gamma.experimental_from_mean_variance(
                mean=alpha, variance=sigma_alpha, name="alpha_human"
            )
            alpha_mouse = yield tfd.Gamma.experimental_from_mean_variance(
                mean=alpha, variance=sigma_alpha, name="alpha_mouse"
            )
            receptor_rate = yield tfd.LogNormal(
                loc=mu_receptor,
                scale=sigma_receptor,
                name="receptor_rate",
            )
            yield tfd.Gamma(
                rate=1 / (receptor_rate + 1e-6),
                concentration=1.0,
                name="receptor_count",
            )
            receptor_binding = tfp.util.DeferredTensor(
                receptor_rate,
                lambda value: receptor_binding_map(
                    value, alpha_mouse, alpha_human, beta
                ),
                shape=[1, n_receptors],
                also_track=[alpha, sigma_alpha, beta],
            )
            yield tfd.Deterministic(loc=receptor_binding, name="receptor_binding")
            target_activation = tfp.util.DeferredTensor(
                receptor_binding,
                lambda value: target_map(value, gamma, mu_target),
                shape=[n_targets],
                also_track=[gamma, mu_target],
            )
            target_log_rate = yield tfd.LogNormal(
                loc=target_activation,
                scale=sigma_target,
                name="target_log_rate",
            )
            yield tfd.Gamma(
                rate=1 / (target_log_rate + 1e-6),
                concentration=1.0,
                name="target_count",
            )

        self._model = joint_model
        self._var_dict = {
            variable.name.split(":")[0]: variable for variable in self._model.variables
        }

    def _build_surrogate_posterior(self) -> None:
        _, tfp, _, _ = _import_tf()
        target_model = self._model.experimental_pin(
            receptor_count=self._X_receptors[0],
            target_count=self._X_targets[0],
        )
        self._surrogate_posterior = (
            tfp.experimental.vi.build_factored_surrogate_posterior(
                event_shape=target_model.event_shape,
                bijector=target_model.experimental_default_event_space_bijector(),
                seed=_seed_pair(self.training_seed, 2),
            )
        )

    def _combined_receptor_gain_regularizer(self, tf: Any) -> Any:
        receptor_loc = [
            variable
            for variable in self._surrogate_posterior.trainable_variables
            if variable.name.startswith("loc_0002")
            and tuple(variable.shape) == (len(self.receptors),)
        ]
        if len(receptor_loc) != 1:
            raise ValueError("Could not identify the receptor-rate posterior location")
        log_gain = self._var_dict["beta"] + receptor_loc[0]
        log_cap = math.log(self.combined_receptor_gain_cap)
        return self.combined_receptor_gain_l2 * tf.reduce_sum(
            tf.square(tf.nn.relu(log_gain - log_cap))
        )

    def _batch_indices(self, epoch: int) -> list[np.ndarray]:
        rng = np.random.RandomState(self.training_seed + epoch)
        indices = rng.permutation(self._X_targets.shape[0])
        stop = (len(indices) // self.batch_size) * self.batch_size
        indices = indices[:stop]
        return [
            indices[start : start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
        ]

    def _validation_arrays(
        self, validation_mouse: ad.AnnData
    ) -> tuple[np.ndarray, np.ndarray]:
        receptor_indices = _gene_indices(validation_mouse, self.receptors)
        target_indices = _gene_indices(validation_mouse, self.targets)
        receptors = _dense_float32(validation_mouse.X[:, receptor_indices])
        targets = _dense_float32(validation_mouse.X[:, target_indices])
        return receptors, targets

    def _snapshot_state(self) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        model_values = tuple(
            np.asarray(variable).copy() for variable in self._var_dict.values()
        )
        surrogate_values = tuple(
            np.asarray(variable).copy()
            for variable in self._surrogate_posterior.trainable_variables
        )
        return model_values, surrogate_values

    def _restore_state(
        self, state: tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]
    ) -> None:
        model_values, surrogate_values = state
        for variable, value in zip(self._var_dict.values(), model_values, strict=True):
            variable.assign(value)
        for variable, value in zip(
            self._surrogate_posterior.trainable_variables,
            surrogate_values,
            strict=True,
        ):
            variable.assign(value)

    def train(
        self,
        *,
        validation_mouse: ad.AnnData | None = None,
        absolute_tolerance: float = 0.0,
        relative_tolerance: float = 0.0,
        patience: int = 3,
        min_evaluations: int = 2,
        evaluation_interval_batches: int = 1,
        validation_mc_replicates: int = 1,
        validation_seed: int = 0,
        restore_best: bool = True,
    ) -> dict[str, Any]:
        for name in ("batch_size", "steps_per_batch", "epochs"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        n_cells = self._X_targets.shape[0]
        if n_cells < self.batch_size:
            raise ValueError(
                f"Training requires at least batch_size={self.batch_size} mouse cells; "
                f"received {n_cells}. Provide more cells or reduce batch_size."
            )
        if validation_mouse is not None and (
            validation_mouse.n_obs == 0
            or validation_mouse.n_obs % self.batch_size != 0
        ):
            raise ValueError(
                "Validation cell count must be a positive multiple of "
                f"batch_size={self.batch_size}; received {validation_mouse.n_obs}."
            )
        self.build_model()
        self._build_surrogate_posterior()
        tf, tfp, _, _ = _import_tf()

        optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)

        def regularization_loss():
            result = tf.constant(0.0, dtype=tf.float32)
            if self.receptor_target_mode == "learned":
                result += self.gamma_l2 * tf.reduce_sum(
                    tf.square(self._var_dict["gamma"])
                )
            if self.combined_receptor_gain_l2 != 0.0:
                result += self._combined_receptor_gain_regularizer(tf)
            return result

        def discrepancy_fn(logu):
            return tfp.vi.kl_reverse(logu) + regularization_loss()

        receptor_count = self._X_receptors.shape[1]
        target_count = self._X_targets.shape[1]

        @tf.function(
            input_signature=(
                tf.TensorSpec([self.batch_size, receptor_count], dtype=tf.float32),
                tf.TensorSpec([self.batch_size, target_count], dtype=tf.float32),
                tf.TensorSpec([2], dtype=tf.int32),
            ),
            autograph=False,
        )
        def fit_one_batch(x_receptors, x_targets, batch_seed):
            target_model = self._model.experimental_pin(
                receptor_count=x_receptors, target_count=x_targets
            )
            return tfp.vi.fit_surrogate_posterior(
                target_model.unnormalized_log_prob,
                self._surrogate_posterior,
                optimizer=optimizer,
                discrepancy_fn=discrepancy_fn,
                num_steps=self.steps_per_batch,
                sample_size=self.batch_size,
                jit_compile=False,
                seed=batch_seed,
            )

        validation_receptors = None
        validation_targets = None
        validation_batches: tuple[np.ndarray, ...] = ()
        validation_history: list[float] = []
        last_validation_batch = 0
        best_validation_loss = np.inf
        best_batch = 0
        best_state = None
        checks_without_improvement = 0
        converged = False

        if validation_mouse is not None:
            validation_receptors, validation_targets = self._validation_arrays(
                validation_mouse
            )
            validation_batches = tuple(
                np.arange(start, start + self.batch_size)
                for start in range(0, len(validation_receptors), self.batch_size)
            )

            @tf.function(
                input_signature=(
                    tf.TensorSpec([self.batch_size, receptor_count], dtype=tf.float32),
                    tf.TensorSpec([self.batch_size, target_count], dtype=tf.float32),
                    tf.TensorSpec([2], dtype=tf.int32),
                ),
                autograph=False,
            )
            def validation_loss(x_receptors, x_targets, validation_seed):
                target_model = self._model.experimental_pin(
                    receptor_count=x_receptors, target_count=x_targets
                )
                return (
                    tfp.vi.monte_carlo_variational_loss(
                        target_model.unnormalized_log_prob,
                        self._surrogate_posterior,
                        sample_size=self.batch_size,
                        seed=validation_seed,
                    )
                    + regularization_loss()
                )

            def evaluate_validation() -> float:
                replicate_values = []
                for replicate in range(validation_mc_replicates):
                    batch_values = []
                    for validation_batch, indices in enumerate(validation_batches):
                        seed_stream = replicate * len(validation_batches)
                        seed_stream += validation_batch
                        value = validation_loss(
                            tf.convert_to_tensor(
                                validation_receptors[indices], dtype=tf.float32
                            ),
                            tf.convert_to_tensor(
                                validation_targets[indices], dtype=tf.float32
                            ),
                            tf.constant(
                                _seed_pair(validation_seed, seed_stream),
                                dtype=tf.int32,
                            ),
                        )
                        batch_values.append(float(value))
                    replicate_values.append(float(np.mean(batch_values)))
                return float(np.mean(replicate_values))

            best_validation_loss = evaluate_validation()
            validation_history.append(best_validation_loss)
            if restore_best:
                best_state = self._snapshot_state()

        losses: list[float] = []
        batch_number = 0
        stop_training = False

        def check_validation() -> None:
            nonlocal best_batch
            nonlocal best_state
            nonlocal best_validation_loss
            nonlocal checks_without_improvement
            nonlocal converged
            nonlocal last_validation_batch
            current = evaluate_validation()
            required_improvement = absolute_tolerance + relative_tolerance * abs(
                best_validation_loss
            )
            if best_validation_loss - current > required_improvement:
                best_validation_loss = current
                best_batch = batch_number
                checks_without_improvement = 0
                if restore_best:
                    best_state = self._snapshot_state()
            else:
                checks_without_improvement += 1
            validation_history.append(current)
            last_validation_batch = batch_number
            converged = (
                len(validation_history) >= min_evaluations
                and checks_without_improvement >= patience
            )

        for epoch in range(self.epochs):
            for indices in self._batch_indices(epoch):
                batch_losses = np.asarray(
                    fit_one_batch(
                        tf.convert_to_tensor(
                            self._X_receptors[indices], dtype=tf.float32
                        ),
                        tf.convert_to_tensor(
                            self._X_targets[indices], dtype=tf.float32
                        ),
                        tf.constant(
                            _seed_pair(self.training_seed, 1000 + batch_number),
                            dtype=tf.int32,
                        ),
                    ),
                    dtype=float,
                )
                losses.extend(batch_losses.tolist())
                batch_number += 1
                if (
                    validation_mouse is not None
                    and batch_number % evaluation_interval_batches == 0
                ):
                    check_validation()
                    if converged:
                        stop_training = True
                        break
            if stop_training:
                break

        if validation_mouse is not None and last_validation_batch != batch_number:
            check_validation()

        loss_array = np.asarray(losses, dtype=np.float64)
        result = {
            "losses": loss_array,
            "num_batches": batch_number,
        }
        if validation_mouse is not None:
            maximum_batches = (
                self._X_targets.shape[0] // self.batch_size
            ) * self.epochs
            restored_best = bool(restore_best and best_batch != batch_number)
            if restored_best:
                self._restore_state(best_state)
            result.update(
                {
                    "validation_history": tuple(validation_history),
                    "converged": converged,
                    "stopped_early": batch_number < maximum_batches,
                    "best_validation_loss": best_validation_loss,
                    "restored_best": restored_best,
                }
            )
        return result

    def sample(
        self, n_samples: int = 2000, *, seed: int | None = None
    ) -> dict[str, np.ndarray]:
        if self._surrogate_posterior is None:
            raise ValueError("Cannot sample before training")
        if seed is None:
            seed = self.posterior_seed

        samples = self._surrogate_posterior.sample(n_samples, seed=_seed_pair(seed, 0))
        values = samples._asdict()
        result = {
            key: np.asarray(values[key])
            for key in ("alpha_human", "alpha_mouse", "receptor_rate")
        }
        receptor_sensitivity = np.exp(np.asarray(self._var_dict["beta"], dtype=float))
        result["receptor_rate"] = (
            self.receptor_gain_cap
            * np.tanh(
                result["receptor_rate"] * receptor_sensitivity / self.receptor_gain_cap
            )
            / receptor_sensitivity
        )
        return result

    def get_parameters(self) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(self._var_dict[name])
            for name in ("alpha", "beta", "gamma")
        }
