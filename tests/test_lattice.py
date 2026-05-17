import jax
import jax.numpy as jnp

from crystalformer.src.layergroup import mult_table
from crystalformer.src.lattice import make_lattice_mask, norm_lattice, symmetrize_lattice


def test_make_lattice_mask_has_80_layer_groups():
    mask = make_lattice_mask()

    assert mask.shape == (80, 6)
    assert mask[0].tolist() == [True, True, True, False, False, True]
    assert mask[7].tolist() == [True, True, True, False, False, False]
    assert mask[47].tolist() == [True, True, True, False, False, False]
    assert mask[48].tolist() == [True, False, True, False, False, False]
    assert mask[63].tolist() == [True, False, True, False, False, False]
    assert mask[64].tolist() == [True, False, True, False, False, False]
    assert mask[79].tolist() == [True, False, True, False, False, False]


def test_symmetrize_layer_lattice_oblique_rectangular_square_and_hexagonal():
    lattice = jnp.array([3.0, 4.0, 20.0, 80.0, 85.0, 71.0])

    assert jnp.allclose(symmetrize_lattice(1, lattice), jnp.array([3.0, 4.0, 20.0, 90.0, 90.0, 71.0]))
    assert jnp.allclose(symmetrize_lattice(7, lattice), jnp.array([3.0, 4.0, 20.0, 90.0, 90.0, 71.0]))
    assert jnp.allclose(symmetrize_lattice(8, lattice), jnp.array([3.0, 4.0, 20.0, 90.0, 90.0, 90.0]))
    assert jnp.allclose(symmetrize_lattice(48, lattice), jnp.array([3.0, 4.0, 20.0, 90.0, 90.0, 90.0]))
    assert jnp.allclose(symmetrize_lattice(49, lattice), jnp.array([3.0, 3.0, 20.0, 90.0, 90.0, 90.0]))
    assert jnp.allclose(symmetrize_lattice(64, lattice), jnp.array([3.0, 3.0, 20.0, 90.0, 90.0, 90.0]))
    assert jnp.allclose(symmetrize_lattice(65, lattice), jnp.array([3.0, 3.0, 20.0, 90.0, 90.0, 120.0]))
    assert jnp.allclose(symmetrize_lattice(80, lattice), jnp.array([3.0, 3.0, 20.0, 90.0, 90.0, 120.0]))


def test_symmetrize_lattice_supports_jit_vmap_for_representative_groups():
    groups = jnp.array([1, 8, 49, 65])
    lattices = jnp.array(
        [
            [3.0, 4.0, 20.0, 80.0, 85.0, 71.0],
            [3.0, 4.0, 20.0, 80.0, 85.0, 71.0],
            [3.0, 4.0, 20.0, 80.0, 85.0, 71.0],
            [3.0, 4.0, 20.0, 80.0, 85.0, 71.0],
        ]
    )

    symmetrized = jax.jit(jax.vmap(symmetrize_lattice, (0, 0)))(groups, lattices)

    assert jnp.allclose(
        symmetrized,
        jnp.array(
            [
                [3.0, 4.0, 20.0, 90.0, 90.0, 71.0],
                [3.0, 4.0, 20.0, 90.0, 90.0, 90.0],
                [3.0, 3.0, 20.0, 90.0, 90.0, 90.0],
                [3.0, 3.0, 20.0, 90.0, 90.0, 120.0],
            ]
        ),
    )


def test_norm_lattice_keeps_lengths_and_converts_angles_to_radians():
    lattice = jnp.array([[3.0, 4.0, 20.0, 90.0, 90.0, 120.0]])
    normed = norm_lattice(lattice)

    assert jnp.allclose(normed[0, :3], jnp.array([3.0, 4.0, 20.0]))
    assert jnp.allclose(normed[0, 3:], jnp.array([jnp.pi / 2, jnp.pi / 2, 2 * jnp.pi / 3]))


def test_norm_lattice_three_argument_form_scales_lengths_with_layer_multiplicities():
    groups = jnp.array([1, 65])
    wyckoffs = jnp.array([[1, 2], [1, 2]])
    lattice = jnp.array(
        [
            [3.0, 4.0, 20.0, 90.0, 90.0, 120.0],
            [6.0, 8.0, 30.0, 90.0, 90.0, 90.0],
        ]
    )

    normed = norm_lattice(groups, wyckoffs, lattice)
    num_atoms = jnp.sum(mult_table[groups[:, None] - 1, wyckoffs], axis=1)

    assert jnp.allclose(normed[:, :3], lattice[:, :3] / num_atoms[:, None] ** (1.0 / 3.0))
    assert jnp.allclose(normed[:, 3:], lattice[:, 3:] * (jnp.pi / 180.0))
