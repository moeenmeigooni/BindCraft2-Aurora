"""JAX primitives expressed without CUDA-only sorting custom calls."""

import jax.numpy as jnp

from bindcraft import oneapi_accelerator


def stable_argsort(values, axis: int=-1):
    """Return ascending stable indices without the GPU CUB sorting FFI on XPU.

    Intel's current oneAPI JAX plugin lowers ``jnp.argsort`` to the CUDA CUB
    ``sort_pairs`` FFI, which has no SYCL implementation.  On Aurora, obtain
    the stable rank of every element through broadcast comparisons and invert
    that rank with a reduction.  BindCraft's arrays are short (residue-sized),
    so the O(n^2) fallback is practical and preserves the semantics required by
    contact selection and ProteinMPNN decoding.  Other backends retain JAX's
    native sort.
    """
    normalized_axis = axis if axis >= 0 else values.ndim + axis
    if not oneapi_accelerator() or normalized_axis != values.ndim - 1:
        return jnp.argsort(values, axis=axis, stable=True)

    count = values.shape[-1]
    positions = jnp.arange(count, dtype=jnp.int32)
    # For each candidate i, count candidates j ordered before it. The index
    # comparison gives identical values their stable JAX ordering.
    left = values[..., :, None]
    right = values[..., None, :]
    earlier = (right < left) | ((right == left) & (positions[None, :] < positions[:, None]))
    ranks = earlier.sum(axis=-1, dtype=jnp.int32)
    return jnp.argmax(ranks[..., :, None] == positions, axis=-2).astype(jnp.int32)

