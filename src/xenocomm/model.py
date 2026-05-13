import os
import pickle
import time
from collections.abc import Callable
from typing import Any

import anndata as ad
import numpy as np
import scanpy as sc

from xenocomm._data import load_lr_rtg_pairs


def _import_tf():
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.experimental.numpy.experimental_enable_numpy_behavior()
    import tensorflow_probability as tfp
    from tensorflow_probability import bijectors as tfb
    from tensorflow_probability import distributions as tfd

    return tf, tfp, tfb, tfd


class XenocommModel:
    """Xenocomm variational inference model for cell-cell communication."""

    def __init__(
        self,
        adata_mouse: ad.AnnData,
        adata_human: ad.AnnData,
        name: str | None = None,
        max_targets: int = 2000,
        cutoff: float = -10,
        mix_rate: float = 1.0,
    ):
        """
        Create model instance and determine gene lists.

        Preprocesses data and determines which genes will be used.
        Call build_model() to construct the TFP model.

        Parameters
        ----------
        adata_mouse : AnnData
            Mouse gene expression data
        adata_human : AnnData
            Human gene expression data
        name : str, optional
            Model identifier for caching and organization
        max_targets : int
            Maximum number of target genes
        cutoff : float
            Dispersion cutoff for gene filtering
        """
        if not 0.0 <= mix_rate <= 1.0:
            raise ValueError(f"mix_rate must be between 0.0 and 1.0, got {mix_rate}")
        self._adata_mouse = adata_mouse
        self._adata_human = adata_human
        self.name = name
        self._load_path: str | None = None
        self._model_built = False
        self._max_targets = max_targets
        self._cutoff = cutoff
        self._mix_rate = mix_rate

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

        self._preprocess_data()
        lr, rtg, ligands, receptors, targets = load_lr_rtg_pairs(
            self._adata_human,
            self._adata_mouse,
            max_targets=self._max_targets,
            cutoff=self._cutoff,
        )
        self._lr = lr
        self._rtg = rtg
        self.ligands = ligands
        self.receptors = receptors
        self.targets = targets

    def _preprocess_data(self) -> None:
        """Preprocess AnnData objects: normalize, calculate dispersions, etc."""
        self._adata_mouse.var_names_make_unique()
        self._adata_human.var_names_make_unique()

        self._adata_human.var_names = [
            str(v).capitalize() for v in self._adata_human.var_names
        ]
        self._adata_mouse.var_names = [
            str(v).capitalize() for v in self._adata_mouse.var_names
        ]

        shared_genes = set(self._adata_human.var_names) & set(
            self._adata_mouse.var_names
        )
        shared_genes = list(shared_genes)
        if len(shared_genes) == 0:
            raise ValueError("No shared genes between mouse and human datasets")
        self._adata_human = self._adata_human[:, shared_genes].copy()
        self._adata_mouse = self._adata_mouse[:, shared_genes].copy()

        if "log1p" not in self._adata_mouse.uns:
            sc.pp.filter_cells(self._adata_mouse, min_genes=5)
            adata_sub = self._adata_mouse.copy()
            sc.pp.log1p(adata_sub)
            sc.pp.highly_variable_genes(adata_sub)
            self._adata_mouse.var["dispersions_norm"] = adata_sub.var[
                "dispersions_norm"
            ]
            sc.pp.normalize_total(self._adata_mouse)
            sc.pp.log1p(self._adata_mouse)
        elif "dispersions_norm" not in self._adata_mouse.var:
            adata_sub = self._adata_mouse.copy()
            sc.pp.highly_variable_genes(adata_sub)
            self._adata_mouse.var["dispersions_norm"] = adata_sub.var[
                "dispersions_norm"
            ]
        if "log1p" not in self._adata_human.uns:
            sc.pp.normalize_total(self._adata_human)
            sc.pp.log1p(self._adata_human)

    def _build_model_from_genes(self) -> None:
        """Build TFP model from already-determined gene lists (used by load())."""
        preloaded = dict(self._var_dict) if self._var_dict else None

        tf, tfp, tfb, tfd = _import_tf()

        if self._lr is None or self._rtg is None:
            if self._adata_human is None or self._adata_mouse is None:
                raise ValueError(
                    "Cannot build model: adata_mouse and adata_human are required. "
                    "Pass them to load() or set them before calling build_model()."
                )
            self._preprocess_data()
            self._lr, self._rtg, _, _, _ = load_lr_rtg_pairs(
                self._adata_human,
                self._adata_mouse,
                max_targets=self._max_targets,
                cutoff=self._cutoff,
            )

        n_ligands = len(self.ligands)
        n_receptors = len(self.receptors)
        n_targets = len(self.targets)

        ligands_set = set(self.ligands)
        receptors_set = set(self.receptors)
        targets_set = set(self.targets)
        lr = [(l, r) for l, r in self._lr if l in ligands_set and r in receptors_set]
        rtg = [(r, t) for r, t in self._rtg if r in receptors_set and t in targets_set]

        ligand_receptor_matrix = np.zeros((n_ligands, n_receptors))
        for l, r in lr:
            if l not in self.ligands or r not in self.receptors:
                continue
            ligand_receptor_matrix[self.ligands.index(l), self.receptors.index(r)] = 1
        ligand_receptor_matrix = tf.constant(ligand_receptor_matrix, dtype=tf.float32)

        receptor_target_matrix = np.zeros((n_receptors, n_targets))
        for r, t in rtg:
            if r not in self.receptors or t not in self.targets:
                continue
            receptor_target_matrix[self.receptors.index(r), self.targets.index(t)] = 1
        receptor_target_matrix = tf.constant(receptor_target_matrix, dtype=tf.float32)

        X_ligand = self._adata_mouse[:, self.ligands].X.toarray()
        ligands_human = [l.capitalize() for l in self.ligands]
        X_ligand_human = self._adata_human[:, ligands_human].X.toarray()

        sample_mean = np.array(
            [np.mean(X_ligand, axis=0), np.mean(X_ligand_human, axis=0)],
            dtype=np.float32,
        )
        sample_mean[1] = (
            self._mix_rate * sample_mean[1] + (1 - self._mix_rate) * sample_mean[0]
        )
        sample_var = np.array(
            [np.var(X_ligand, axis=0), np.var(X_ligand_human, axis=0)], dtype=np.float32
        )
        sample_mean = np.clip(sample_mean, 1e-3, None)
        sample_var = np.clip(sample_var, 1e-3, None)

        X_target = self._adata_mouse[:, self.targets].X.toarray()

        sigma_bijector = tfb.Chain([tfb.Exp(), tfb.Shift(shift=1e-2)])

        mu_receptor = tfp.util.TransformedVariable(
            tf.zeros([n_receptors], dtype=tf.float32),
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
            tf.constant(np.log(np.mean(X_target, axis=0) + 1e-6), dtype=tf.float32),
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
            receptor_rate = tf.clip_by_value(receptor_rate, 1e-6, 100)
            receptor_binding = tf.math.multiply(receptor_binding_base, receptor_rate)
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

            _ = yield tfd.Gamma(
                rate=1 / (receptor_rate + 1e-6),
                concentration=1.0,
                name="receptor_count",
            )

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

            _ = yield tfd.Gamma(
                rate=1 / (target_log_rate + 1e-6),
                concentration=1.0,
                name="target_count",
            )

        self.ligand_receptor_matrix = ligand_receptor_matrix
        self.receptor_target_matrix = receptor_target_matrix
        self.mean_ligand = mean_ligand_var
        self._model = _model
        self._receptor_binding_map = receptor_binding_map
        self._model_built = True

        self._var_dict = {v.name.split(":")[0]: v for v in self._model.variables}
        if preloaded:
            for key, val in preloaded.items():
                if key in self._var_dict:
                    self._var_dict[key].assign(tf.constant(val, dtype=tf.float32))

    def build_model(self) -> None:
        """
        Build the TensorFlow Probability model (heavy operation).

        Constructs the probabilistic model from already-determined gene lists.
        Gene lists are determined during __init__() based on max_targets and
        cutoff parameters.

        Notes
        -----
        This is the slow step that builds the full TFP model with all
        bijectors and distributions. The genes to use have already been
        determined in __init__().
        """
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
        """
        Train model using variational inference.

        Requires build_model() to have been called first.

        Parameters
        ----------
        num_steps : int
            Number of optimization steps per batch
        learning_rate : float
            Adam optimizer learning rate
        batch_size : int
            Batch size
        num_epochs : int
            Number of passes through the data
        shuffle : bool
            Whether to shuffle data at start of each epoch
        stratify_by : str, optional
            Column name in adata_mouse.obs for stratified sampling
        verbose : str or bool
            "rich" for interactive progress bar, "print" for line-by-line
            output, True is an alias for "rich", False disables output

        Returns
        -------
        results : dict
            Dictionary containing:
            - 'losses': All losses across all epochs/batches
            - 'epoch_losses': Average loss per epoch
            - 'num_epochs': Number of epochs trained
            - 'num_batches_per_epoch': Number of batches per epoch

        Notes
        -----
        Logic from current train_model() in vi.py:631-733.
        """
        if not self._model_built:
            raise ValueError("Cannot train: model not built. Call build_model() first.")

        tf, tfp, tfb, tfd = _import_tf()
        from xenocomm._batching import DataBatcher

        X_targets = self._adata_mouse[:, self.targets].X.toarray()
        X_receptors = self._adata_mouse[:, self.receptors].X.toarray()

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
        self._var_dict = var_dict

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        def discrepancy_fn(logu):
            loss = tfp.vi.kl_reverse(logu)
            loss += 1e-3 * tf.reduce_sum(tf.square(var_dict["gamma"]))
            return loss

        stratify_labels = None
        if stratify_by is not None:
            if stratify_by not in self._adata_mouse.obs:
                raise ValueError(f"Column '{stratify_by}' not found in adata_mouse.obs")
            stratify_labels = self._adata_mouse.obs[stratify_by].values

        batcher = DataBatcher(
            X_receptors=X_receptors,
            X_targets=X_targets,
            batch_size=batch_size,
            shuffle=shuffle,
            stratify_by=stratify_labels,
            seed=None,
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

            for batch_idx, (X_receptors_batch, X_targets_batch) in enumerate(
                batcher.get_batches(epoch)
            ):
                if use_rich or use_print:
                    batch_start_time = time.time()

                target_model = self._model.experimental_pin(
                    receptor_count=X_receptors_batch, target_count=X_targets_batch
                )

                losses = tfp.vi.fit_surrogate_posterior(
                    target_model.unnormalized_log_prob,
                    self._surrogate_posterior,
                    optimizer=optimizer,
                    discrepancy_fn=discrepancy_fn,
                    num_steps=num_steps,
                    sample_size=batch_size,
                    jit_compile=False,
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
        """
        Save trained model parameters to a single self-contained .npz file.

        Parameters
        ----------
        path : str
            Explicit path to .npz file (no default)

        Raises
        ------
        ValueError
            If model not trained (surrogate_posterior is None)
        """
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

        metadata_dict = {
            "name": self.name,
            "mix_rate": self._mix_rate,
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
    ) -> "XenocommModel":
        """
        Load trained model from .npz file.

        Parameters
        ----------
        path : str
            Path to saved .npz file
        adata_mouse : AnnData, optional
            Mouse data (required for old format only)
        adata_human : AnnData, optional
            Human data (required for old format only)
        for_analysis : bool
            Ignored, kept for API compatibility

        Returns
        -------
        model : XenocommModel
            Loaded model with trained parameters
        """
        data = np.load(path, allow_pickle=True)

        if "ligands" in data:
            return cls._load_new_format(path, data, adata_mouse, adata_human)
        else:
            return cls._load_old_format(path, data, adata_mouse, adata_human)

    @classmethod
    def _load_new_format(cls, path, data, adata_mouse, adata_human):
        model = object.__new__(cls)

        model._adata_mouse = adata_mouse
        model._adata_human = adata_human
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

        skip_keys = {
            "ligands",
            "receptors",
            "targets",
            "mean_ligand",
            "ligand_receptor_matrix",
            "receptor_target_matrix",
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

    @classmethod
    def _load_old_format(cls, path, data, adata_mouse, adata_human):
        if adata_mouse is None or adata_human is None:
            raise ValueError(
                "adata_mouse and adata_human are required for loading old format models."
            )

        tf, tfp, tfb, tfd = _import_tf()

        metadata_path = path.replace(".npz", "_metadata.pkl")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}. "
                "This model may have been saved with an older version."
            )

        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        model = cls(
            adata_mouse,
            adata_human,
            name=metadata.get("name"),
            mix_rate=metadata.get("mix_rate", 1.0),
        )
        model._load_path = os.path.abspath(path)

        model.ligands = sorted(metadata["ligands"])
        model.receptors = sorted(metadata["receptors"])
        model.targets = sorted(metadata["targets"])
        model._build_model_from_genes()

        X_receptors = model._adata_mouse[:, model.receptors].X.toarray()
        X_targets = model._adata_mouse[:, model.targets].X.toarray()

        target_model = model._model.experimental_pin(
            receptor_count=X_receptors[0, :], target_count=X_targets[0, :]
        )

        model._surrogate_posterior = (
            tfp.experimental.vi.build_factored_surrogate_posterior(
                event_shape=target_model.event_shape,
                bijector=target_model.experimental_default_event_space_bijector(),
            )
        )

        variables = [
            *model._model.trainable_variables,
            *model._surrogate_posterior.trainable_variables,
        ]
        tfp.experimental.nn.util.variables_load(path, variables)

        model._var_dict = {v.name.split(":")[0]: v for v in model._model.variables}
        model._training_history = metadata.get("training_history", [])

        return model

    def _rebuild_surrogate_posterior(self) -> None:
        """Rebuild TF model and surrogate posterior from preloaded weights."""
        if not self._preloaded_surrogate:
            raise ValueError("No preloaded surrogate weights to restore.")
        if self._adata_mouse is None or self._adata_human is None:
            raise ValueError(
                "Cannot rebuild surrogate posterior without adata. "
                "Pass adata_mouse and adata_human to load()."
            )

        tf, tfp, tfb, tfd = _import_tf()

        self._build_model_from_genes()

        X_receptors = self._adata_mouse[:, self.receptors].X.toarray()
        X_targets = self._adata_mouse[:, self.targets].X.toarray()

        target_model = self._model.experimental_pin(
            receptor_count=X_receptors[0, :], target_count=X_targets[0, :]
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

    def sample(self, n_samples: int = 100) -> dict[str, np.ndarray]:
        """Draw samples from the surrogate posterior distribution.

        Parameters
        ----------
        n_samples : int
            Number of posterior samples.

        Returns
        -------
        dict mapping sample name to array of shape (n_samples, ...).
        """
        if self._surrogate_posterior is None and self._preloaded_surrogate:
            self._rebuild_surrogate_posterior()

        if self._surrogate_posterior is None:
            raise ValueError(
                "Cannot sample: model not trained or loaded. "
                "Call train() or load() first."
            )

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
        """
        Get current parameter values as numpy arrays.

        Returns
        -------
        params : dict
            Dictionary with keys: alpha, beta, gamma,
            mu_receptor, sigma_receptor, mu_target, sigma_target
        """
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


def _build_lightweight_surrogate(flat_event_size):
    """Build lightweight surrogate posterior for analysis (no expensive bijectors)."""
    tf, tfp, tfb, tfd = _import_tf()

    base_standard_dist = tfd.JointDistributionSequential(
        [tfd.Sample(tfd.Normal(loc=0.0, scale=1.0), s) for s in flat_event_size]
    )

    loc_bijector = tfb.JointMap(
        tf.nest.map_structure(
            lambda s: tfb.Shift(
                tf.Variable(
                    tf.random.uniform((s,), minval=-0.1, maxval=0.1, dtype=tf.float32)
                )
            ),
            flat_event_size,
        )
    )

    scale_bijector = tfb.JointMap(
        tf.nest.map_structure(
            lambda s: tfb.ScaleMatvecDiag(
                tf.Variable(
                    tf.ones(
                        (s,),
                        dtype=tf.float32,
                    )
                ),
            ),
            flat_event_size,
        )
    )

    surrogate_bijector = tfb.Chain([scale_bijector, loc_bijector])

    return tfd.TransformedDistribution(base_standard_dist, surrogate_bijector)
