"""Bounded Aurora execution gate for the JAX primitives BC2 requires."""
import os
from pathlib import Path

import jax
import jax.numpy as jnp

from bindcraft.af.alphafold.model import quat_affine
from bindcraft.loss import orthonormal_columns, positive_definite_inverse
from bindcraft.protein import kabsch
from bindcraft.xpu import stable_argsort


def check_project_paths() -> None:
    home = Path(os.environ['HOME']).resolve()
    cache = Path(os.environ['BINDCRAFT2_CACHE_ROOT']).resolve()
    if home == Path('/home') or '/home/' in str(home):
        raise RuntimeError(f'HOME was not redirected from user home: {home}')
    for variable in ('XDG_CACHE_HOME', 'JAX_COMPILATION_CACHE_DIR', 'BINDCRAFT_WEIGHTS', 'TMPDIR'):
        value = Path(os.environ[variable]).resolve()
        if '/home/' in str(value):
            raise RuntimeError(f'{variable} still points into user home: {value}')
    print(f'cache_root={cache}')
    print(f'runtime_home={home}')


def main() -> None:
    check_project_paths()
    devices = jax.devices()
    if not devices or not any(device.platform in {'oneapi', 'sycl'} for device in devices):
        raise RuntimeError(f'JAX did not expose an Aurora oneAPI/SYCL device: {devices}')
    device = devices[0]
    matrix = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)
    product = jax.jit(lambda x: x @ x)(matrix).block_until_ready()

    # AlphaFold converts backbone rotation frames to quaternions.  Exercise the
    # trace-positive and all three largest-diagonal branches; this replaces the
    # unavailable oneAPI eigh primitive used by upstream BindCraft2.
    rotations = jnp.asarray((
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((1, 0, 0), (0, -1, 0), (0, 0, -1)),
        ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
        ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
    ), dtype=jnp.float32)

    def quaternion_round_trip(rotation):
        quaternion = quat_affine.rot_to_quat(rotation, unstack_inputs=True)
        return quat_affine.rot_list_to_tensor(quat_affine.quat_to_rot(quaternion))

    recovered_rotations = jax.jit(quaternion_round_trip)(rotations).block_until_ready()

    # Kabsch alignment ordinarily uses SVD.  Aurora uses the equivalent
    # Davenport-quaternion fallback; compile it on device rather than relying
    # on the unavailable oneAPI SVD lowering.
    alignment_coordinates = jnp.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0),
                                         (0, 0, 1)), dtype=jnp.float32)
    alignment_rotation = jnp.asarray(((0, -1, 0), (1, 0, 0), (0, 0, 1)),
                                     dtype=jnp.float32)
    fitted_rotation, _, _ = jax.jit(kabsch)(
        alignment_coordinates, alignment_coordinates @ alignment_rotation.T,
        jnp.ones(len(alignment_coordinates), dtype=jnp.float32))

    # Collective-softness ordinarily uses QR and a matrix inverse. Aurora uses
    # compatible Gram--Schmidt and Newton--Schulz fallbacks.
    basis = jax.jit(orthonormal_columns)(matrix + jnp.eye(4))
    positive_matrix = (matrix + jnp.eye(4)).T @ (matrix + jnp.eye(4)) + jnp.eye(4)
    inverse = jax.jit(positive_definite_inverse)(positive_matrix)
    # JAX oneAPI currently routes argsort through a CUDA CUB custom call. The
    # Aurora fallback must compile and preserve stable sorting semantics.
    sort_input = jnp.asarray(((3, 1, 1, 2), (4, -1, 2, 2)), dtype=jnp.float32)
    sort_order = jax.jit(stable_argsort)(sort_input)
    jax.block_until_ready((basis, inverse))
    if (float(product.sum()) != 3920.0
            or not bool(jnp.allclose(recovered_rotations, rotations, atol=1e-5))
            or not bool(jnp.allclose(fitted_rotation, alignment_rotation, atol=1e-4))
            or not bool(jnp.allclose(basis.T @ basis, jnp.eye(4), atol=1e-4))
            or not bool(jnp.allclose(positive_matrix @ inverse, jnp.eye(4), atol=1e-4))
            or not bool(jnp.array_equal(sort_order, jnp.asarray(((1, 2, 3, 0), (1, 2, 3, 0))))):
        raise RuntimeError('oneAPI linear algebra smoke result was invalid')
    print(f'backend={jax.default_backend()}')
    print(f'devices={devices}')
    print(f'ZE_AFFINITY_MASK={os.environ.get("ZE_AFFINITY_MASK", "unset")}')
    print('BindCraft2 Aurora XPU smoke passed.')


if __name__ == '__main__':
    main()
