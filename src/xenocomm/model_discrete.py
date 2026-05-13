import os
import time
from collections.abc import Callable
from typing import Any

import anndata as ad
import numpy as np
import scanpy as sc

from xenocomm._data import load_lr_rtg_pairs
from xenocomm.model import _import_tf


class XenocommDiscreteModel:
    """Xenocomm variational inference model using Poisson-LogNormal likelihoods.

    Unlike XenocommModel which uses continuous Gamma likelihoods on log-normalized
    data, this model uses Poisson likelihoods on raw integer counts with LogNormal
    latent rates. Ligand expression remains on the log-normalized scale.
    """

    def __init__(
        self,
        adata_mouse: ad.AnnData,
        adata_human: ad.AnnData,
        name: str | None = None,
        max_targets: int = 2000,
        cutoff: float = -10,
        mix_rate: float = 1.0,
        cell_type_key: str = "cell_type",
    ):
        if not 0.0 <= mix_rate <= 1.0:
            raise ValueError(f"mix_rate must be between 0.0 and 1.0, got {mix_rate}")
        self._adata_mouse = adata_mouse
        self._adata_human = adata_human
        self._adata_mouse_lognorm: ad.AnnData | None = None
        self._adata_human_lognorm: ad.AnnData | None = None
        self.name = name
        self._load_path: str | None = None
        self._model_built = False
        self._max_targets = max_targets
        self._cutoff = cutoff
        self._mix_rate = mix_rate
        self._cell_type_key = cell_type_key

        self.ligands: list[str] | None = None
        self.receptors: list[str] | None = None
        self.targets: list[str] | None = None
        self.ligand_receptor_matrix: Any | None = None
        self.receptor_target_matrix: Any | None = None
        self.mean_ligand: Any | None = None

        self._model: Any | None = None
        self._surrogate_posterior: Any | None = None
        self._var_dict: dict[str, Any] = {}
        self._receptor_binding_map: Callable | None = None
        self._training_history: list[dict] = []
        self._preloaded_surrogate: dict[str, np.ndarray] = {}
        self._cell_type_names: list[str] | None = None
        self._cell_type_to_idx: dict[str, int] | None = None
        self._cell_type_indices: np.ndarray | None = None
        self._n_cell_types: int = 0
        self._cell_type_idx_var: Any | None = None

        self._preprocess_data()
        lr, rtg, ligands, receptors, targets = load_lr_rtg_pairs(
            self._adata_human_lognorm,
            self._adata_mouse_lognorm,
            max_targets=self._max_targets,
            cutoff=self._cutoff,
        )
        self._lr = lr
        self._rtg = rtg
        self.ligands = ligands
        self.receptors = receptors
        self.targets = targets

    def _preprocess_data(self) -> None:
        """Preprocess AnnData objects.

        Keeps raw counts in _adata_mouse for Poisson likelihoods.
        Subsamples human to max 50k cells, normalizes in-place.
        Log-normalizes mouse in-place (keeping raw in .raw).
        """
        self._adata_mouse.var.index = self._adata_mouse.var.index.astype(str)
        self._adata_human.var.index = self._adata_human.var.index.astype(str)
        self._adata_mouse.var_names_make_unique()
        self._adata_human.var_names_make_unique()

        self._adata_human.var_names = [
            str(v).capitalize() for v in self._adata_human.var_names
        ]
        self._adata_mouse.var_names = [
            str(v).capitalize() for v in self._adata_mouse.var_names
        ]

        # Subsample human if large (only used for ligand means)
        if self._adata_human.n_obs > 50000:
            rng = np.random.RandomState(0)
            idx = rng.choice(self._adata_human.n_obs, 50000, replace=False)
            self._adata_human = self._adata_human[idx].copy()

        shared_genes = set(self._adata_human.var_names) & set(
            self._adata_mouse.var_names
        )
        shared_genes = list(shared_genes)
        if len(shared_genes) == 0:
            raise ValueError("No shared genes between mouse and human datasets")
        self._adata_human = self._adata_human[:, shared_genes].copy()
        self._adata_mouse = self._adata_mouse[:, shared_genes].copy()

        # Normalize human in-place
        if "log1p" not in self._adata_human.uns:
            sc.pp.normalize_total(self._adata_human)
            sc.pp.log1p(self._adata_human)
        self._adata_human_lognorm = self._adata_human

        # Mouse: compute dispersions, then store raw in .raw and log-normalize in-place
        if "log1p" not in self._adata_mouse.uns:
            sc.pp.filter_cells(self._adata_mouse, min_genes=5)
            if "dispersions_norm" not in self._adata_mouse.var:
                adata_sub = self._adata_mouse.copy()
                sc.pp.normalize_total(adata_sub)
                sc.pp.log1p(adata_sub)
                sc.pp.highly_variable_genes(adata_sub)
                self._adata_mouse.var["dispersions_norm"] = adata_sub.var[
                    "dispersions_norm"
                ]
                del adata_sub
            self._adata_mouse.raw = self._adata_mouse.copy()
            sc.pp.normalize_total(self._adata_mouse)
            sc.pp.log1p(self._adata_mouse)
        elif not hasattr(self._adata_mouse, "raw") or self._adata_mouse.raw is None:
            raise ValueError(
                "Mouse data is already log-normalized but has no .raw layer. "
                "Pass raw count data for the discrete model."
            )

        self._adata_mouse_lognorm = self._adata_mouse

        # Extract cell type annotations
        if self._cell_type_key in self._adata_mouse.obs.columns:
            ct_values = self._adata_mouse.obs[self._cell_type_key].values
            self._cell_type_names = sorted(set(ct_values))
            self._cell_type_to_idx = {
                ct: i for i, ct in enumerate(self._cell_type_names)
            }
            self._cell_type_indices = np.array(
                [self._cell_type_to_idx[ct] for ct in ct_values]
            )
            self._n_cell_types = len(self._cell_type_names)

    def _build_model_from_genes(self) -> None:
        preloaded = dict(self._var_dict) if self._var_dict else None

        tf, tfp, tfb, tfd = _import_tf()

        # When loaded from file or after a prior build, structural arrays already exist
        has_structural = (
            self.mean_ligand is not None
            and self.ligand_receptor_matrix is not None
            and self.receptor_target_matrix is not None
        )

        if not has_structural:
            if self._adata_mouse_lognorm is None and self._adata_mouse is not None:
                self._preprocess_data()

            if self._lr is None or self._rtg is None:
                if (
                    self._adata_human_lognorm is None
                    or self._adata_mouse_lognorm is None
                ):
                    raise ValueError(
                        "Cannot build model: adata_mouse and adata_human are required."
                    )
                self._lr, self._rtg, _, _, _ = load_lr_rtg_pairs(
                    self._adata_human_lognorm,
                    self._adata_mouse_lognorm,
                    max_targets=self._max_targets,
                    cutoff=self._cutoff,
                )

        n_ligands = len(self.ligands)
        n_receptors = len(self.receptors)
        n_targets = len(self.targets)

        if has_structural:
            ligand_receptor_matrix = tf.constant(
                self.ligand_receptor_matrix, dtype=tf.float32
            )
            receptor_target_matrix = tf.constant(
                self.receptor_target_matrix, dtype=tf.float32
            )
            sample_mean = np.asarray(self.mean_ligand)
            sample_var = np.ones_like(sample_mean)
        else:
            ligands_set = set(self.ligands)
            receptors_set = set(self.receptors)
            targets_set = set(self.targets)
            lr = [
                (l, r) for l, r in self._lr if l in ligands_set and r in receptors_set
            ]
            rtg = [
                (r, t) for r, t in self._rtg if r in receptors_set and t in targets_set
            ]

            ligand_receptor_matrix_np = np.zeros((n_ligands, n_receptors))
            for l, r in lr:
                if l not in self.ligands or r not in self.receptors:
                    continue
                ligand_receptor_matrix_np[
                    self.ligands.index(l), self.receptors.index(r)
                ] = 1
            ligand_receptor_matrix = tf.constant(
                ligand_receptor_matrix_np, dtype=tf.float32
            )

            receptor_target_matrix_np = np.zeros((n_receptors, n_targets))
            for r, t in rtg:
                if r not in self.receptors or t not in self.targets:
                    continue
                receptor_target_matrix_np[
                    self.receptors.index(r), self.targets.index(t)
                ] = 1
            receptor_target_matrix = tf.constant(
                receptor_target_matrix_np, dtype=tf.float32
            )

            X_ligand = self._adata_mouse_lognorm[:, self.ligands].X.toarray()
            ligands_human = [l.capitalize() for l in self.ligands]
            X_ligand_human = self._adata_human_lognorm[:, ligands_human].X.toarray()

            sample_mean = np.array(
                [np.mean(X_ligand, axis=0), np.mean(X_ligand_human, axis=0)],
                dtype=np.float32,
            )
            sample_mean[1] = (
                self._mix_rate * sample_mean[1] + (1 - self._mix_rate) * sample_mean[0]
            )
            sample_var = np.array(
                [np.var(X_ligand, axis=0), np.var(X_ligand_human, axis=0)],
                dtype=np.float32,
            )
            sample_mean = np.clip(sample_mean, 1e-3, None)
            sample_var = np.clip(sample_var, 1e-3, None)

        # Compute raw count stats for initialization (or use zeros if loading from file)
        n_cell_types = self._n_cell_types if self._n_cell_types > 0 else 1

        if has_structural:
            mu_receptor_init = tf.zeros([n_cell_types, n_receptors], dtype=tf.float32)
            mu_target_init = tf.zeros([n_targets], dtype=tf.float32)
        else:
            X_receptor = self._adata_mouse.raw[:, self.receptors].X.toarray()
            X_target = self._adata_mouse.raw[:, self.targets].X.toarray()
            mu_target_init = tf.constant(
                np.log(np.mean(X_target, axis=0) + 1e-6), dtype=tf.float32
            )

            if self._cell_type_indices is not None:
                mu_r = np.zeros((n_cell_types, n_receptors), dtype=np.float32)
                for ct_idx in range(n_cell_types):
                    mask = self._cell_type_indices == ct_idx
                    if mask.sum() > 0:
                        mu_r[ct_idx] = np.log(np.mean(X_receptor[mask], axis=0) + 1e-6)
                mu_receptor_init = tf.constant(mu_r, dtype=tf.float32)
            else:
                mu_receptor_init = tf.constant(
                    np.log(np.mean(X_receptor, axis=0) + 1e-6)[None, :],
                    dtype=tf.float32,
                )

        sigma_bijector = tfb.Chain([tfb.Exp(), tfb.Shift(shift=1e-2)])

        # Full per-cell-type parameters (trainable, but NOT used directly in coroutine)
        mu_receptor_full = tfp.util.TransformedVariable(
            mu_receptor_init,
            tfb.Identity(),
            dtype=tf.float32,
            trainable=True,
            name="mu_receptor",
        )
        sigma_receptor_full = tfp.util.TransformedVariable(
            tf.ones([n_cell_types, n_receptors], dtype=tf.float32),
            sigma_bijector,
            trainable=True,
            name="sigma_receptor",
        )
        self._mu_receptor_full = mu_receptor_full
        self._sigma_receptor_full = sigma_receptor_full

        # Per-cell batch variables (non-trainable, shape [n_receptors]).
        # The coroutine uses these — they keep event shape = [n_receptors].
        # Before each training batch, we assign from tf.gather(full, ct_idx).
        # Gradient propagation through gather is handled by a custom train step.
        mu_receptor = tf.Variable(
            tf.zeros([n_receptors], dtype=tf.float32),
            trainable=False,
            name="mu_receptor_batch",
        )
        sigma_receptor = tf.Variable(
            sigma_bijector.forward(tf.ones([n_receptors], dtype=tf.float32)),
            trainable=False,
            name="sigma_receptor_batch",
        )

        mu_target = tfp.util.TransformedVariable(
            mu_target_init,
            tfb.Identity(),
            trainable=True,
            name="mu_target",
        )
        sigma_target = tfp.util.TransformedVariable(
            tf.ones([n_targets], dtype=tf.float32),
            sigma_bijector,
            trainable=True,
            name="sigma_target",
        )

        mean_ligand = tf.squeeze(sample_mean.copy())

        alpha_bijector = tfb.Softplus()

        mean_ligand_var = tf.Variable(
            mean_ligand, dtype=tf.float32, trainable=False, name="mean_ligand"
        )

        alpha = tfp.util.TransformedVariable(
            tf.ones(n_ligands, dtype=tf.float32),
            bijector=alpha_bijector,
            name="alpha",
            trainable=True,
        )

        sigma_alpha = tfp.util.TransformedVariable(
            2.0 * tf.ones([n_ligands], dtype=tf.float32),
            tfb.Softplus(),
            name="sigma_alpha",
            trainable=True,
        )
        gamma = tf.Variable(
            tf.random.normal(
                shape=(n_receptors, n_targets), mean=0.0, stddev=0.1, dtype=tf.float32
            ),
            name="gamma",
            trainable=True,
        )

        beta = tfp.util.TransformedVariable(
            tf.ones([n_receptors], dtype=tf.float32),
            tfb.Exp(),
            name="beta",
            trainable=True,
        )

        def receptor_binding_map(receptor_rate, alpha_mouse, alpha_human, beta):
            alpha_stacked = tf.stack([alpha_mouse, alpha_human], axis=0)
            alpha_combined = tf.math.reduce_sum(
                mean_ligand_var * alpha_stacked, axis=0, keepdims=True
            )
            receptor_binding_base = tf.squeeze(
                tf.matmul(alpha_combined, ligand_receptor_matrix)
            )
            receptor_binding_base = tf.math.multiply(receptor_binding_base, beta)
            log_receptor_rate = tf.math.log1p(tf.clip_by_value(receptor_rate, 0.0, 1e6))
            receptor_binding = tf.math.multiply(
                receptor_binding_base, log_receptor_rate
            )
            receptor_binding = tf.reshape(receptor_binding, [1, -1])
            receptor_binding = receptor_binding / (1 + receptor_binding)
            return receptor_binding

        def target_map(x, gamma, mu_target):
            gamma2 = tf.square(gamma)
            Z = tf.clip_by_value(tf.reduce_max(gamma2, axis=1), 1e-6, 1)
            gamma2 = gamma2 / Z[:, None]
            return tf.matmul(x, gamma2).flatten() + tf.math.exp(mu_target)

        @tfd.JointDistributionCoroutineAutoBatched
        def _model():
            ligand_accessibility_human = yield tfd.Gamma.experimental_from_mean_variance(
                mean=alpha, variance=sigma_alpha, name="alpha_human"
            )
            ligand_accessibility_mouse = yield tfd.Gamma.experimental_from_mean_variance(
                mean=alpha, variance=sigma_alpha, name="alpha_mouse"
            )

            receptor_rate = yield tfd.LogNormal(
                loc=mu_receptor,
                scale=sigma_receptor,
                name="receptor_rate",
            )

            _ = yield tfd.Poisson(rate=receptor_rate, name="receptor_count")

            receptor_binding = tfp.util.DeferredTensor(
                receptor_rate,
                lambda x: receptor_binding_map(
                    x, ligand_accessibility_mouse, ligand_accessibility_human, beta
                ),
                shape=[1, n_receptors],
                also_track=[alpha, sigma_alpha, beta],
            )

            _ = yield tfd.Deterministic(loc=receptor_binding, name="receptor_binding")

            target_activation = tfp.util.DeferredTensor(
                receptor_binding,
                lambda x: target_map(x, gamma, mu_target),
                shape=[n_targets],
                also_track=[gamma, mu_target],
            )

            target_log_rate = yield tfd.LogNormal.experimental_from_mean_variance(
                mean=target_activation,
                variance=sigma_target,
                name="target_log_rate",
            )

            _ = yield tfd.Poisson(rate=target_log_rate, name="target_count")

        self.ligand_receptor_matrix = ligand_receptor_matrix
        self.receptor_target_matrix = receptor_target_matrix
        self.mean_ligand = mean_ligand_var
        self._model = _model
        self._receptor_binding_map = receptor_binding_map
        self._model_built = True

        self._var_dict = {v.name.split(":")[0]: v for v in self._model.variables}
        # Store the full per-cell-type parameters (not the batch variables)
        self._var_dict["mu_receptor"] = mu_receptor_full
        self._var_dict["sigma_receptor"] = sigma_receptor_full
        if preloaded:
            for key, val in preloaded.items():
                if key in self._var_dict:
                    self._var_dict[key].assign(tf.constant(val, dtype=tf.float32))

    def build_model(self) -> None:
        if self.ligands is None or self.receptors is None or self.targets is None:
            raise ValueError("Gene lists not determined")
        self._build_model_from_genes()

    def train(
        self,
        num_steps: int = 250,
        learning_rate: float = 1e-3,
        batch_size: int = 1024,
        num_epochs: int = 1,
        shuffle: bool = True,
        stratify_by: str | None = None,
        verbose: str | bool = "rich",
    ) -> dict:
        if not self._model_built:
            raise ValueError("Cannot train: model not built. Call build_model() first.")

        tf, tfp, tfb, tfd = _import_tf()
        from xenocomm._batching import DataBatcher

        # Raw counts for Poisson likelihoods
        X_targets = self._adata_mouse.raw[:, self.targets].X.toarray()
        X_receptors = self._adata_mouse.raw[:, self.receptors].X.toarray()

        if batch_size > X_targets.shape[0]:
            batch_size = X_targets.shape[0]

        if self._surrogate_posterior is None:
            target_model = self._model.experimental_pin(
                receptor_count=X_receptors[0, :], target_count=X_targets[0, :]
            )

            self._surrogate_posterior = (
                tfp.experimental.vi.build_factored_surrogate_posterior(
                    event_shape=target_model.event_shape,
                    bijector=target_model.experimental_default_event_space_bijector(),
                )
            )

            preloaded_surrogate = getattr(self, "_preloaded_surrogate", None)
            if preloaded_surrogate:
                for i, var in enumerate(self._surrogate_posterior.trainable_variables):
                    key = f"_surrogate/{i}"
                    if key in preloaded_surrogate:
                        var.assign(
                            tf.constant(preloaded_surrogate[key], dtype=var.dtype)
                        )
                self._preloaded_surrogate = {}

        var_dict = {v.name.split(":")[0]: v for v in self._model.variables}
        var_dict["mu_receptor"] = self._mu_receptor_full
        var_dict["sigma_receptor"] = self._sigma_receptor_full
        self._var_dict = var_dict

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        ct_optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        def discrepancy_fn(logu):
            loss = tfp.vi.kl_reverse(logu)
            loss += 1e-3 * tf.reduce_sum(tf.square(var_dict["gamma"]))
            loss += 1e-2 * tf.reduce_sum(tf.square(var_dict["beta"]))
            return loss

        stratify_labels = None
        if stratify_by is not None:
            if stratify_by not in self._adata_mouse.obs:
                raise ValueError(f"Column '{stratify_by}' not found in adata_mouse.obs")
            stratify_labels = self._adata_mouse.obs[stratify_by].values

        extra_arrays = {}
        if self._cell_type_indices is not None:
            extra_arrays["cell_type_idx"] = self._cell_type_indices

        batcher = DataBatcher(
            X_receptors=X_receptors,
            X_targets=X_targets,
            batch_size=batch_size,
            shuffle=shuffle,
            stratify_by=stratify_labels,
            seed=None,
            extra_arrays=extra_arrays if extra_arrays else None,
        )

        num_batches = batcher.num_batches
        mvn_loss = []
        smoothed_losses = []
        epoch_losses = []
        ema_decay = 1 - 2 / (num_steps + 1)

        if verbose is True:
            verbose = "rich"
        elif verbose is False:
            verbose = None

        use_rich = verbose == "rich"
        use_print = verbose == "print"

        if use_rich:
            import psutil
            from rich.progress import (
                BarColumn,
                Progress,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            total_iterations = num_epochs * num_batches

            class MetricsColumn(TextColumn):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)

                def render(self, task):
                    epoch = task.fields.get("epoch", 1)
                    total_epochs = task.fields.get("total_epochs", 1)
                    batch_per_s = task.fields.get("batch_per_s", 0.0)
                    cells_per_s = task.fields.get("cells_per_s", 0.0)
                    loss = task.fields.get("loss", 0.0)
                    gpu_mem = task.fields.get("gpu_mem", 0.0)
                    ram_mem = task.fields.get("ram_mem", 0.0)

                    cells_str = (
                        f"{cells_per_s / 1000:.1f}K"
                        if cells_per_s < 1e6
                        else f"{cells_per_s / 1e6:.1f}M"
                    )

                    parts = [
                        f"Epoch {epoch}/{total_epochs}",
                        f"{batch_per_s:.1f} batch/s",
                        f"{cells_str} cells/s",
                        f"Loss {loss:.2f}",
                    ]

                    if gpu_mem > 0:
                        parts.append(f"GPU {gpu_mem:.1f}GB")
                    parts.append(f"RAM {ram_mem:.1f}GB")

                    return "  ".join(parts)

            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TextColumn("<"),
                TimeRemainingColumn(),
                MetricsColumn(""),
                expand=False,
            )
            progress.start()

            task_id = progress.add_task(
                "Training",
                total=total_iterations,
                epoch=1,
                total_epochs=num_epochs,
                batch_per_s=0.0,
                cells_per_s=0.0,
                loss=0.0,
                gpu_mem=0.0,
                ram_mem=0.0,
            )

            ema_batch_per_s = None
            alpha = 0.1

        train_start_time = time.time()

        for epoch in range(num_epochs):
            epoch_loss_values = []

            has_cell_types = self._cell_type_indices is not None

            # Get references to the batch variables used in the model coroutine
            mu_receptor_batch_var = next(
                v for v in self._model.variables if "mu_receptor_batch" in v.name
            )
            sigma_receptor_batch_var = next(
                v for v in self._model.variables if "sigma_receptor_batch" in v.name
            )

            for batch_idx, batch_data in enumerate(batcher.get_batches(epoch)):
                if len(batch_data) == 3:
                    X_receptors_batch, X_targets_batch, extras = batch_data
                    ct_idx_batch = extras["cell_type_idx"]
                else:
                    X_receptors_batch, X_targets_batch = batch_data
                    ct_idx_batch = None

                if use_rich or use_print:
                    batch_start_time = time.time()

                # Set batch variables to mean of cell-type params for this batch
                if has_cell_types and ct_idx_batch is not None:
                    mu_receptor_batch_var.assign(
                        tf.reduce_mean(
                            tf.gather(self._mu_receptor_full, ct_idx_batch), axis=0
                        )
                    )
                    sigma_receptor_batch_var.assign(
                        tf.reduce_mean(
                            tf.gather(self._sigma_receptor_full, ct_idx_batch), axis=0
                        )
                    )

                target_model = self._model.experimental_pin(
                    receptor_count=X_receptors_batch, target_count=X_targets_batch
                )

                # Custom training step: uses fit_surrogate_posterior for model/surrogate
                # params, then a separate gradient step for cell-type params
                losses = tfp.vi.fit_surrogate_posterior(
                    target_model.unnormalized_log_prob,
                    self._surrogate_posterior,
                    optimizer=optimizer,
                    discrepancy_fn=discrepancy_fn,
                    num_steps=num_steps,
                    sample_size=batch_size,
                    jit_compile=False,
                )

                # Gradient step for cell-type-specific receptor params
                if has_cell_types and ct_idx_batch is not None:
                    # Gather from underlying raw variables to avoid
                    # IndexedSlices issues with TransformedVariable bijectors
                    mu_raw_var = self._mu_receptor_full.trainable_variables[0]
                    sigma_raw_var = self._sigma_receptor_full.trainable_variables[0]
                    sigma_bij = self._sigma_receptor_full.bijector

                    with tf.GradientTape() as tape:
                        mu_r = tf.gather(mu_raw_var, ct_idx_batch)
                        sigma_r = sigma_bij.forward(
                            tf.gather(sigma_raw_var, ct_idx_batch)
                        )
                        samples_ct = self._surrogate_posterior.sample(batch_size)
                        if hasattr(samples_ct, "_asdict"):
                            receptor_rate = samples_ct.receptor_rate
                        else:
                            receptor_rate = samples_ct[2]
                        receptor_rate = tf.stop_gradient(tf.abs(receptor_rate) + 1e-6)
                        ct_log_prob = tfd.LogNormal(loc=mu_r, scale=sigma_r).log_prob(
                            receptor_rate
                        )
                        ct_loss = -tf.reduce_mean(ct_log_prob)
                    ct_grads = tape.gradient(ct_loss, [mu_raw_var, sigma_raw_var])
                    ct_grads = [
                        tf.convert_to_tensor(g)
                        if isinstance(g, tf.IndexedSlices)
                        else g
                        for g in ct_grads
                    ]
                    ct_optimizer.apply_gradients(
                        (g, v)
                        for g, v in zip(
                            ct_grads, [mu_raw_var, sigma_raw_var], strict=False
                        )
                        if g is not None
                    )

                batch_losses = losses.numpy()
                mvn_loss.extend(batch_losses)
                epoch_loss_values.extend(batch_losses)

                for val in batch_losses:
                    if len(smoothed_losses) == 0:
                        smoothed_losses.append(float(val))
                    else:
                        smoothed_losses.append(
                            ema_decay * smoothed_losses[-1]
                            + (1 - ema_decay) * float(val)
                        )

                if np.any(np.isnan(mvn_loss)):
                    if use_rich:
                        progress.stop()
                    print(f"NaN loss at epoch {epoch + 1}, batch {batch_idx + 1}")
                    break

                if use_rich or use_print:
                    batch_end_time = time.time()
                    batch_time = batch_end_time - batch_start_time
                    current_batch_per_s = 1.0 / batch_time if batch_time > 0 else 0.0

                if use_rich:
                    if ema_batch_per_s is None:
                        ema_batch_per_s = current_batch_per_s
                    else:
                        ema_batch_per_s = (
                            alpha * current_batch_per_s + (1 - alpha) * ema_batch_per_s
                        )

                    cells_per_s = ema_batch_per_s * batch_size

                    try:
                        gpu_info = tf.config.experimental.get_memory_info("GPU:0")
                        gpu_mem_gb = gpu_info["current"] / 1e9
                    except Exception:
                        gpu_mem_gb = 0.0

                    ram_mem_gb = psutil.virtual_memory().used / 1e9

                    progress.update(
                        task_id,
                        advance=1,
                        epoch=epoch + 1,
                        total_epochs=num_epochs,
                        batch_per_s=ema_batch_per_s,
                        cells_per_s=cells_per_s,
                        loss=losses[-1],
                        gpu_mem=gpu_mem_gb,
                        ram_mem=ram_mem_gb,
                    )
                elif use_print:
                    print(
                        f"Epoch {epoch + 1}/{num_epochs}  "
                        f"Batch {batch_idx + 1}/{num_batches}  "
                        f"Loss {losses[-1]:.2f}  "
                        f"{batch_time:.1f}s/batch",
                        flush=True,
                    )

            if np.any(np.isnan(mvn_loss)):
                break

            epoch_avg_loss = np.mean(epoch_loss_values)
            epoch_losses.append(epoch_avg_loss)
            if use_print:
                print(
                    f"Epoch {epoch + 1}/{num_epochs} complete  Avg loss: {epoch_avg_loss:.2f}",
                    flush=True,
                )

        if use_rich:
            progress.stop()

        elapsed_seconds = time.time() - train_start_time

        results = {
            "losses": np.array(mvn_loss),
            "smoothed_losses": np.array(smoothed_losses),
            "epoch_losses": epoch_losses,
            "num_steps": num_steps,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "num_batches_per_epoch": num_batches,
            "shuffle": shuffle,
            "stratify_by": stratify_by,
            "elapsed_seconds": elapsed_seconds,
        }
        self._training_history.append(results)
        return results

    @property
    def training_history(self) -> list[dict]:
        return self._training_history

    @property
    def mean_ligand_np(self) -> np.ndarray:
        return np.asarray(self.mean_ligand)

    @property
    def ligand_receptor_matrix_np(self) -> np.ndarray:
        return np.asarray(self.ligand_receptor_matrix)

    @property
    def receptor_target_matrix_np(self) -> np.ndarray:
        return np.asarray(self.receptor_target_matrix)

    def save(self, path: str) -> None:
        if self._surrogate_posterior is None:
            raise ValueError("Cannot save: model not trained. Call train() first.")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        save_dict = {
            "ligands": np.array(self.ligands, dtype=str),
            "receptors": np.array(self.receptors, dtype=str),
            "targets": np.array(self.targets, dtype=str),
            "mean_ligand": self.mean_ligand.numpy(),
            "ligand_receptor_matrix": self.ligand_receptor_matrix.numpy(),
            "receptor_target_matrix": self.receptor_target_matrix.numpy(),
        }

        for key, var in self._var_dict.items():
            save_dict[key] = var.numpy()

        for i, var in enumerate(self._surrogate_posterior.trainable_variables):
            save_dict[f"_surrogate/{i}"] = var.numpy()

        if self._cell_type_names:
            save_dict["cell_type_names"] = np.array(self._cell_type_names, dtype=str)

        metadata_dict = {
            "name": self.name,
            "mix_rate": self._mix_rate,
            "model_type": "discrete",
            "n_cell_types": self._n_cell_types,
            "training_history": self._training_history,
        }
        save_dict["_metadata"] = np.array(metadata_dict, dtype=object)

        np.savez(path, **save_dict)

    @classmethod
    def load(
        cls,
        path: str,
        adata_mouse: ad.AnnData | None = None,
        adata_human: ad.AnnData | None = None,
        for_analysis: bool = True,
    ) -> "XenocommDiscreteModel":
        data = np.load(path, allow_pickle=True)

        if "ligands" not in data:
            raise ValueError("Old format not supported for discrete model.")

        return cls._load_new_format(path, data, adata_mouse, adata_human)

    @classmethod
    def _load_new_format(cls, path, data, adata_mouse, adata_human):
        model = object.__new__(cls)

        model._adata_mouse = adata_mouse
        model._adata_human = adata_human
        model._adata_mouse_lognorm = None
        model._adata_human_lognorm = None
        model._load_path = os.path.abspath(path)
        model._model_built = False
        model._model = None
        model._surrogate_posterior = None
        model._receptor_binding_map = None
        model._training_history = []
        model._max_targets = 2000
        model._cutoff = -10
        model._mix_rate = 1.0
        model._lr = None
        model._rtg = None
        model._preloaded_surrogate = {}
        model._cell_type_key = "cell_type"
        model._cell_type_idx_var = None

        model.ligands = list(data["ligands"])
        model.receptors = list(data["receptors"])
        model.targets = list(data["targets"])

        model.mean_ligand = np.array(data["mean_ligand"])
        model.ligand_receptor_matrix = np.array(data["ligand_receptor_matrix"])
        model.receptor_target_matrix = np.array(data["receptor_target_matrix"])

        metadata = data["_metadata"].item()
        model.name = metadata.get("name")
        model._mix_rate = metadata.get("mix_rate", 1.0)
        model._training_history = metadata.get("training_history", [])

        if "cell_type_names" in data:
            model._cell_type_names = list(data["cell_type_names"])
            model._cell_type_to_idx = {
                ct: i for i, ct in enumerate(model._cell_type_names)
            }
            model._n_cell_types = len(model._cell_type_names)
        else:
            model._cell_type_names = None
            model._cell_type_to_idx = None
            model._n_cell_types = metadata.get("n_cell_types", 0)
        model._cell_type_indices = None

        skip_keys = {
            "ligands",
            "receptors",
            "targets",
            "mean_ligand",
            "ligand_receptor_matrix",
            "receptor_target_matrix",
            "cell_type_names",
        }
        model._var_dict = {}
        model._preloaded_surrogate = {}
        for key in data.files:
            if key.startswith("_surrogate/"):
                model._preloaded_surrogate[key] = np.array(data[key])
            elif key.startswith("_") or key in skip_keys:
                continue
            else:
                model._var_dict[key] = np.array(data[key])

        return model

    def _rebuild_surrogate_posterior(self) -> None:
        if not self._preloaded_surrogate:
            raise ValueError("No preloaded surrogate weights to restore.")

        tf, tfp, tfb, tfd = _import_tf()

        self._build_model_from_genes()

        # Use dummy data for pinning (only need event shapes)
        n_receptors = len(self.receptors)
        n_targets = len(self.targets)
        dummy_receptors = tf.zeros([n_receptors], dtype=tf.float32)
        dummy_targets = tf.zeros([n_targets], dtype=tf.float32)

        target_model = self._model.experimental_pin(
            receptor_count=dummy_receptors, target_count=dummy_targets
        )
        self._surrogate_posterior = (
            tfp.experimental.vi.build_factored_surrogate_posterior(
                event_shape=target_model.event_shape,
                bijector=target_model.experimental_default_event_space_bijector(),
            )
        )

        for i, var in enumerate(self._surrogate_posterior.trainable_variables):
            key = f"_surrogate/{i}"
            if key in self._preloaded_surrogate:
                var.assign(tf.constant(self._preloaded_surrogate[key], dtype=var.dtype))
        self._preloaded_surrogate = {}

    def sample(
        self, n_samples: int = 100, cell_type: str | None = None
    ) -> dict[str, np.ndarray]:
        if self._surrogate_posterior is None and self._preloaded_surrogate:
            self._rebuild_surrogate_posterior()

        if self._surrogate_posterior is None:
            raise ValueError(
                "Cannot sample: model not trained or loaded. "
                "Call train() or load() first."
            )

        if cell_type is not None and self._cell_type_to_idx is not None:
            if cell_type not in self._cell_type_to_idx:
                raise ValueError(
                    f"Unknown cell type '{cell_type}'. "
                    f"Available: {self._cell_type_names}"
                )
            ct_idx = self._cell_type_to_idx[cell_type]
            tf, _, _, _ = _import_tf()
            # Set batch variables to this cell type's params
            mu_r = self._mu_receptor_full[ct_idx]
            sigma_r = self._sigma_receptor_full[ct_idx]
            for v in self._model.variables:
                if "mu_receptor_batch" in v.name:
                    v.assign(mu_r)
                elif "sigma_receptor_batch" in v.name:
                    v.assign(sigma_r)

        samples = self._surrogate_posterior.sample(n_samples)

        samples_dict = {}
        if hasattr(samples, "_asdict"):
            for i, key in enumerate(samples._asdict().keys()):
                samples_dict[key] = samples[i].numpy()
        else:
            sample_names = [
                "alpha_human",
                "alpha_mouse",
                "receptor_rate",
                "receptor_binding",
                "target_log_rate",
            ]
            for i, key in enumerate(sample_names[: len(samples)]):
                samples_dict[key] = samples[i].numpy()

        return samples_dict

    def get_parameters(self) -> dict[str, np.ndarray]:
        if not self._var_dict:
            raise ValueError(
                "No parameters available. Build and train the model, or load from file."
            )

        params = {}
        for key, var in self._var_dict.items():
            if isinstance(var, np.ndarray):
                params[key] = var
            else:
                params[key] = var.numpy()

        return params
