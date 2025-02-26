from typing import List, Sequence, Tuple

import chex
import jax
import jax.extend
import numpy as np
from jax import numpy as jnp


@chex.dataclass
class SplitInfo:
	"""Stores split information for a dimension."""

	axis: int
	indices: np.ndarray
	sizes: np.ndarray


def _split_tensor(tensor: jax.Array, axis: int, indices: jax.Array) -> List[jax.Array]:
	"""JIT-compiled tensor splitting."""
	return jnp.split(tensor, indices_or_sections=indices, axis=axis)


def _merge_tensors(tensors: Sequence[jax.Array], axis: int) -> jax.Array:
	"""JIT-compiled tensor merging."""
	return jnp.concatenate(tensors, axis=axis)


class BlockPartitioner:
	"""Partitions a tensor into smaller tensors with optimized operations.

	Modified from distributed_shampoo with performance improvements.
	https://github.com/google-research/google-research/blob/master/scalable_shampoo/optax/distributed_shampoo.py
	"""

	def __init__(
		self, param_shape: Tuple[int, ...], block_size: int, dim_diag: Sequence[bool]
	):
		"""Initialize the partitioner with optimized preprocessing.

		Args:
		    param_shape: Shape of the parameter tensor to partition
		    block_size: Size of blocks to partition into
		    dim_diag: Boolean flags indicating diagonal dimensions
		"""
		assert len(dim_diag) == len(param_shape), "dim_diag must match param_shape length"

		self._shape = param_shape
		self._splits: List[SplitInfo] = []
		split_sizes: List[np.ndarray] = []

		# Vectorized split calculation
		for i, (d, is_diag) in enumerate(zip(param_shape, dim_diag)):
			if 0 < block_size < d and not is_diag:
				# Compute splits efficiently using numpy
				nsplit = (d - 1) // block_size
				indices = np.arange(1, nsplit + 1, dtype=np.int32) * block_size

				# Calculate sizes with vectorized operations
				sizes = np.full(nsplit + 1, block_size, dtype=np.int32)
				sizes[-1] = d - indices[-1]

				self._splits.append(SplitInfo(i, indices, sizes))
				split_sizes.append(sizes)
			else:
				split_sizes.append(np.array([d], dtype=np.int32))

		self._split_sizes = split_sizes

		# Precompute shapes for efficiency
		single_shape = np.array([s[0] for s in split_sizes])
		padded_single_shape = (
			np.ceil(single_shape / block_size).astype(np.int32) * block_size
		)

		# Compute stack size efficiently
		split_lengths = np.array([len(s) for s in split_sizes])
		stack_size = max(1, np.prod(np.maximum(1, split_lengths)))

		self._padded_stacked_shape = tuple([stack_size] + padded_single_shape.tolist())

		# Cache frequently used values
		self._nsplits = len(self._splits)
		self._reverse_splits = list(reversed(self._splits))

	@property
	def split_sizes(self) -> List[np.ndarray]:
		"""Return the sizes of splits."""
		return self._split_sizes

	def partition(self, tensor: jax.Array) -> Tuple[jax.Array, ...]:
		"""Partition tensor into blocks with optimized operations."""
		assert tensor.shape == self._shape, (
			f"Expected shape {self._shape}, got {tensor.shape}"
		)

		if not self._splits:
			return (tensor,)

		tensors = [tensor]
		for split_info in self._splits:
			# Process each split level
			tensors_local = []
			for t in tensors:
				# Use JIT-compiled split function
				split_results = _split_tensor(t, split_info.axis, split_info.indices)
				tensors_local.extend(split_results)
			tensors = tensors_local

		return tuple(tensors)

	def merge_partitions(self, partitions: Sequence[jax.Array]) -> jax.Array:
		"""Merge partitions back to original shape with optimized operations."""
		if not self._splits:
			assert len(partitions) == 1
			return partitions[0]

		parts = list(partitions)
		for split_info in self._reverse_splits:
			n = len(split_info.indices) + 1
			merged = []

			# Process chunks in parallel where possible
			for start_idx in range(0, len(parts), n):
				chunk = parts[start_idx : start_idx + n]
				# Use JIT-compiled merge function
				merged.append(_merge_tensors(chunk, split_info.axis))

			parts = merged

		assert len(parts) == 1, f"Expected 1 partition, got {len(parts)}"
		return parts[0]
