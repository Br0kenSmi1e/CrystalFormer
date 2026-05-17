import jax.numpy as jnp

from crystalformer.src.layergroup import mult_table


def make_lattice_mask() -> jnp.ndarray:
    oblique = [[True, True, True, False, False, True]] * 7
    rectangular = [[True, True, True, False, False, False]] * 41
    square = [[True, False, True, False, False, False]] * 16
    hexagonal = [[True, False, True, False, False, False]] * 16
    return jnp.array(oblique + rectangular + square + hexagonal)


def symmetrize_lattice(g: int, lattice: jnp.ndarray) -> jnp.ndarray:
    a, b, c, _alpha, _beta, gamma = lattice

    oblique = jnp.array([a, b, c, 90.0, 90.0, gamma])
    rectangular = jnp.array([a, b, c, 90.0, 90.0, 90.0])
    square = jnp.array([a, a, c, 90.0, 90.0, 90.0])
    hexagonal = jnp.array([a, a, c, 90.0, 90.0, 120.0])

    lattice = jnp.where(g <= 7, oblique, rectangular)
    lattice = jnp.where((g >= 49) & (g <= 64), square, lattice)
    lattice = jnp.where(g >= 65, hexagonal, lattice)
    return lattice


def _norm_lattice_angles(lattice: jnp.ndarray) -> jnp.ndarray:
    length, angle = jnp.split(lattice, 2, axis=-1)
    return jnp.concatenate([length, angle * (jnp.pi / 180.0)], axis=-1)


def norm_lattice(*args: jnp.ndarray) -> jnp.ndarray:
    if len(args) == 1:
        return _norm_lattice_angles(args[0])
    if len(args) == 3:
        groups, wyckoffs, lattice = args
        multiplicities = mult_table[groups[:, None] - 1, wyckoffs]
        num_atoms = jnp.sum(multiplicities, axis=1)
        length, angle = jnp.split(lattice, 2, axis=-1)
        length = length / num_atoms[:, None] ** (1 / 3)
        return jnp.concatenate([length, angle * (jnp.pi / 180.0)], axis=-1)
    raise TypeError("norm_lattice expects either lattice or groups, wyckoffs, lattice")
