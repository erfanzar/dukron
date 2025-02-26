from typing import List, Optional, Sequence, Tuple

import jax
import jax.extend
import numpy as np
from jax import numpy as jnp
from jax.sharding import PartitionSpec


def partitions(lst: List[int]):
	"""Generate all partitions of a list."""
	if not lst:
		yield [[]]
	else:
		for i in range(len(lst)):
			for part in partitions(lst[i + 1 :]):
				yield [lst[: i + 1]] + part


def merge_small_dims(
	shape_to_merge: Tuple[int, ...],
	max_dim: int,
	dim_diag: List[bool],
	sharding_to_merge: Optional[Tuple] = None,
) -> Tuple[List[int], List[bool], Optional[PartitionSpec]]:
	"""
	Merge small dimensions with optimized computations.

	Uses caching to avoid recompiling the loss function for each partition and
	to avoid recomputing products for groups that appear in multiple partitions.
	"""
	# Early exits for trivial cases.
	if not shape_to_merge:
		return [], [True], PartitionSpec() if sharding_to_merge is not None else None

	shape_array = np.array(shape_to_merge)
	if np.all(shape_array == 1):
		return (
			[1],
			[True],
			PartitionSpec(None) if sharding_to_merge is not None else None,
		)

	# Cache for group products (keyed by tuple(group))
	group_product_cache = {}

	# Cache for loss functions keyed by the number of groups (length of input array).
	dim2loss_funcs = {}

	def get_dim2loss(n: int):
		"""Return a JIT-compiled loss function for input arrays of length n."""
		if n not in dim2loss_funcs:

			def loss_fn(d: jnp.ndarray, dim0: float):
				too_small = dim0 / 8
				too_large = dim0 * 8

				small_loss = jnp.where(
					d < dim0,
					jnp.log2(dim0 / d)
					+ jnp.where(d < too_small, 100 * jnp.log2(too_small / d), 0.0),
					0.0,
				)
				large_loss = jnp.where(
					d >= dim0,
					10 * jnp.log2(d / dim0)
					+ jnp.where(d > too_large, 1000 * jnp.log2(d / too_large), 0.0),
					0.0,
				)
				return jnp.sum(small_loss + large_loss)

			dim2loss_funcs[n] = loss_fn
		return dim2loss_funcs[n]

	shape_indices = list(range(len(shape_to_merge)))
	best_loss = float("inf")
	best_partition = None

	for partition in partitions(shape_indices):
		if not partition:
			continue

		group_products = []
		for group in partition:
			if not group:
				continue
			key = tuple(group)
			if key not in group_product_cache:
				group_product_cache[key] = np.prod(
					[shape_to_merge[i] for i in group], dtype=np.float64
				)
			group_products.append(group_product_cache[key])
		# Convert the list to a JAX array.
		group_products_arr = jnp.array(group_products)
		n_groups = len(group_products)
		total_loss = float(get_dim2loss(n_groups)(group_products_arr, max_dim))
		if total_loss < best_loss:
			best_loss = total_loss
			best_partition = [group for group in partition if group]

	# Compute the final merged shapes and diagonal flags.
	merged_shape = [
		int(np.prod([shape_to_merge[i] for i in group], dtype=np.int64))
		for group in best_partition
	]
	merged_diag = [all(dim_diag[i] for i in group) for group in best_partition]

	if sharding_to_merge is None:
		return merged_shape, merged_diag, None

	merged_sharding = []
	for group in best_partition:
		group_shardings = [sharding_to_merge[i] for i in group]
		valid_shardings = [s for s in group_shardings if s is not None]
		if len(valid_shardings) > 1:
			merged_sharding.append(tuple(valid_shardings))
		elif valid_shardings:
			merged_sharding.append(valid_shardings[0])
		else:
			merged_sharding.append(None)

	return merged_shape, merged_diag, PartitionSpec(*merged_sharding)


def pad_and_stack_matrices(array_list: List[jax.Array], block_size: int) -> jax.Array:
	"""Efficiently pad and stack matrices with minimized memory operations.

	- Scalars are promoted to 1D arrays.
	- The maximum shape along each axis is determined.
	- Each array is padded to a shape that is a multiple of block_size.
	"""
	if len(array_list[0].shape) == 0:
		array_list = [jnp.atleast_1d(arr) for arr in array_list]
	shapes = [jnp.array(arr.shape) for arr in array_list]
	max_dims = jnp.max(jnp.stack(shapes), axis=0)
	padded_shape = jnp.ceil(max_dims / block_size).astype(int) * block_size
	padded_shape = tuple(int(x) for x in padded_shape)

	def pad_array(arr):
		pad_width = [(0, p - s) for s, p in zip(arr.shape, padded_shape)]
		return jnp.pad(arr, pad_width)

	padded_arrays = jnp.stack([pad_array(arr) for arr in array_list])
	return padded_arrays


def unstack_and_unpad_matrices(
	stacked_array: jax.Array,
	original_shapes: Sequence[Tuple[int, ...]],
) -> Tuple[jax.Array, ...]:
	"""Efficiently unstack and unpad matrices, restoring their original shapes.

	Each matrix is sliced according to its original shape.
	"""
	unpadded_list = []
	for i, shape in enumerate(original_shapes):
		slices = tuple(slice(0, s) for s in shape)
		unpadded_list.append(stacked_array[i][slices])
	if len(original_shapes[0]) == 0:
		unpadded_list = [arr.squeeze() for arr in unpadded_list]
	return tuple(unpadded_list)


def norm_lower_bound(A: jax.Array):
	"""Returns a cheap lower bound for the spectral norm of A.

	Optimized for hermitian matrices.
	"""

	max_abs = jnp.max(jnp.abs(A))

	def calc():
		A_norm = A / max_abs
		col_norms = jnp.sum(A_norm * A_norm, axis=0)
		i = jnp.argmax(col_norms)
		x = A_norm[:, i]
		x_A = jnp.dot(x, A_norm)
		x_norm = jnp.linalg.norm(x_A)
		normalized_x = x_A / x_norm
		result = jnp.sqrt(jnp.dot(normalized_x, jnp.dot(A_norm, normalized_x)))
		return max_abs * result

	return jax.lax.cond(max_abs > 0, calc, lambda: max_abs)


def solve_triangular_right(X, A):
	"""Compute X @ inv(A) with a triangular solve.

	Instead of nested vmaps over each leading dimension, we flatten the batch
	dimensions and use one vmap. (A triangular solve has roughly the same
	complexity as a matmul.)
	"""

	orig_shape = X.shape
	X = jnp.atleast_2d(X)
	dtype_out = jnp.promote_types(A.dtype, X.dtype)
	A = A.astype(dtype_out)
	X = X.astype(dtype_out)
	solution = jax.lax.linalg.triangular_solve(
		A,
		X,
		left_side=False,
		lower=False,
		transpose_a=False,
	)
	return solution[0] if len(orig_shape) < 2 else solution


def conjB(Q, G, V):
	"""Compute conjB with optimized transpositions and solves.

	Uses more efficient array operations and minimizes temporary allocations.
	"""
	order = G.ndim
	conjB = jnp.transpose(V, [*range(1, order), 0])
	for i, q in enumerate(Q):
		if q.ndim < 2:
			conjB = conjB / q
		else:
			conjB = solve_triangular_right(conjB, q)
		if i < order - 1:
			conjB = jnp.moveaxis(conjB, i, order - 1)
	return conjB
