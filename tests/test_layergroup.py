import jax
import jax.numpy as jnp

from crystalformer.src.layergroup import (
    LAYER_GROUP_COUNT,
    MULTIPLICITY_LCM,
    WYCKOFF_TYPES,
    dof0_table,
    fc_mask_table,
    from_xyz_str,
    mult_table,
    symmetrize_atoms,
    symmetrize_atoms_padded,
    symops,
    wmax_table,
)


def test_layer_table_shapes():
    assert LAYER_GROUP_COUNT == 80
    assert WYCKOFF_TYPES == 19
    assert MULTIPLICITY_LCM == 48
    assert symops.shape == (80, 19, 48, 3, 4)
    assert mult_table.shape == (80, 19)
    assert wmax_table.shape == (80,)
    assert dof0_table.shape == (80, 19)
    assert fc_mask_table.shape == (80, 19, 3)


def test_layer_group_one_has_free_xyz_general_position():
    assert int(wmax_table[0]) == 1
    assert int(mult_table[0, 1]) == 1
    assert fc_mask_table[0, 1].tolist() == [True, True, True]


def test_layer_group_two_reverses_wyckoff_order_to_match_existing_w_indices():
    assert int(wmax_table[1]) == 5
    assert mult_table[1, 1:6].tolist() == [1, 1, 1, 1, 2]
    assert fc_mask_table[1, 1].tolist() == [False, False, False]
    assert fc_mask_table[1, 5].tolist() == [True, True, True]


def test_from_xyz_str_parses_rotation_and_translation():
    op = from_xyz_str("-x+1/2,y,z-1/3")

    assert op.shape == (3, 4)
    assert jnp.allclose(op[0], jnp.array([-1.0, 0.0, 0.0, 0.5]))
    assert jnp.allclose(op[1], jnp.array([0.0, 1.0, 0.0, 0.0]))
    assert jnp.allclose(op[2], jnp.array([0.0, 0.0, 1.0, -1.0 / 3.0]))


def test_symmetrize_atoms_uses_layer_group_tables():
    coords = symmetrize_atoms(2, 5, jnp.array([0.2, 0.3, 0.4]))

    assert coords.shape == (2, 3)
    assert jnp.all(coords[:, :2] >= 0.0)
    assert jnp.all(coords[:, :2] < 1.0)


def test_symmetrize_atoms_preserves_nonperiodic_z():
    coords = symmetrize_atoms(1, 1, jnp.array([1.2, -0.3, 1.25]))

    assert coords.shape == (1, 3)
    assert jnp.allclose(coords[0], jnp.array([0.2, 0.7, 1.25]))


def test_symmetrize_atoms_padded_is_jittable():
    coords, mask = jax.jit(symmetrize_atoms_padded)(
        jnp.array(2), jnp.array(5), jnp.array([0.2, 0.3, 0.4])
    )

    assert coords.shape == (MULTIPLICITY_LCM, 3)
    assert mask.shape == (MULTIPLICITY_LCM,)
    assert int(jnp.sum(mask)) == 2
    assert jnp.all(coords[:, :2] >= 0.0)
    assert jnp.all(coords[:, :2] < 1.0)
