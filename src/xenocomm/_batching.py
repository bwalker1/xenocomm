from collections.abc import Iterator

import numpy as np
import tensorflow as tf


class DataBatcher:
    """
    Batching system for training with support for epochs, shuffling, and stratified sampling.

    Maintains numpy arrays internally and lazily converts batches to TF tensors.
    """

    def __init__(
        self,
        X_receptors: np.ndarray,
        X_targets: np.ndarray,
        batch_size: int = 4096,
        shuffle: bool = True,
        stratify_by: np.ndarray | None = None,
        seed: int | None = None,
        extra_arrays: dict[str, np.ndarray] | None = None,
    ):
        if X_receptors.shape[0] != X_targets.shape[0]:
            raise ValueError(
                f"X_receptors and X_targets must have same number of cells, "
                f"got {X_receptors.shape[0]} and {X_targets.shape[0]}"
            )

        self.X_receptors = X_receptors
        self.X_targets = X_targets
        self.n_cells = X_receptors.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.stratify_by = stratify_by
        self.seed = seed
        self.extra_arrays = extra_arrays or {}

        self._rng = np.random.RandomState(seed)

        if stratify_by is not None:
            if len(stratify_by) != self.n_cells:
                raise ValueError(
                    f"stratify_by must have same length as n_cells, "
                    f"got {len(stratify_by)} and {self.n_cells}"
                )
            self._setup_stratified_sampling()
        else:
            self._group_indices = None

    def _setup_stratified_sampling(self):
        """Prepare indices for each stratum/group."""
        unique_groups = np.unique(self.stratify_by)
        self._group_indices = {}
        self._group_sizes = {}

        for group in unique_groups:
            indices = np.where(self.stratify_by == group)[0]
            self._group_indices[group] = indices
            self._group_sizes[group] = len(indices)

        self._groups = list(unique_groups)
        self._group_proportions = np.array(
            [self._group_sizes[g] / self.n_cells for g in self._groups]
        )

    @property
    def num_batches(self) -> int:
        """Number of batches per epoch."""
        return self.n_cells // self.batch_size

    def get_batches(
        self, epoch: int
    ) -> Iterator[
        tuple[tf.Tensor, tf.Tensor] | tuple[tf.Tensor, tf.Tensor, dict[str, tf.Tensor]]
    ]:
        if self.stratify_by is not None:
            indices = self._get_stratified_indices(epoch)
        else:
            indices = self._get_shuffled_indices(epoch)

        num_batches = self.num_batches
        has_extras = bool(self.extra_arrays)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = (batch_idx + 1) * self.batch_size
            batch_indices = indices[start_idx:end_idx]

            X_receptors_tf = tf.constant(
                self.X_receptors[batch_indices], dtype=tf.float32
            )
            X_targets_tf = tf.constant(self.X_targets[batch_indices], dtype=tf.float32)

            if has_extras:
                extras = {}
                for key, arr in self.extra_arrays.items():
                    batch_arr = arr[batch_indices]
                    dtype = (
                        tf.int32
                        if batch_arr.dtype
                        in (np.int32, np.int64, np.int_, np.int8, np.int16)
                        else tf.float32
                    )
                    extras[key] = tf.constant(batch_arr, dtype=dtype)
                yield X_receptors_tf, X_targets_tf, extras
            else:
                yield X_receptors_tf, X_targets_tf

    def _get_shuffled_indices(self, epoch: int) -> np.ndarray:
        """Get shuffled or ordered indices for the epoch."""
        if self.shuffle:
            if self.seed is not None:
                epoch_seed = self.seed + epoch
                rng = np.random.RandomState(epoch_seed)
            else:
                rng = self._rng
            return rng.permutation(self.n_cells)
        else:
            return np.arange(self.n_cells)

    def _get_stratified_indices(self, epoch: int) -> np.ndarray:
        """Get stratified sample indices for the epoch."""
        if self.seed is not None:
            epoch_seed = self.seed + epoch
            rng = np.random.RandomState(epoch_seed)
        else:
            rng = self._rng

        if self.shuffle:
            for group in self._groups:
                rng.shuffle(self._group_indices[group])

        num_batches = self.num_batches
        all_batch_indices = []

        for batch_idx in range(num_batches):
            batch_indices = []
            for group_idx, group in enumerate(self._groups):
                target_count = int(
                    np.round(self.batch_size * self._group_proportions[group_idx])
                )

                group_inds = self._group_indices[group]
                available = len(group_inds)

                if available == 0:
                    continue

                start = (batch_idx * target_count) % available
                end = start + target_count

                if end <= available:
                    selected = group_inds[start:end]
                else:
                    selected = np.concatenate(
                        [group_inds[start:], group_inds[: end - available]]
                    )

                batch_indices.extend(selected)

            batch_indices = np.array(batch_indices, dtype=np.int64)
            if self.shuffle:
                rng.shuffle(batch_indices)

            if len(batch_indices) > self.batch_size:
                batch_indices = batch_indices[: self.batch_size]
            elif len(batch_indices) < self.batch_size:
                deficit = self.batch_size - len(batch_indices)
                all_indices = np.arange(self.n_cells)
                extra = rng.choice(all_indices, size=deficit, replace=False)
                batch_indices = np.concatenate([batch_indices, extra])

            all_batch_indices.append(batch_indices)

        return np.concatenate(all_batch_indices)
