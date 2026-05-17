import jax.numpy as jnp

from crystalformer.src.layergroup import LAYER_GROUP_COUNT, WYCKOFF_TYPES, fc_mask_table, symops, wmax_table


def test_layer_group_two_fc_mask_marks_fixed_and_free_positions():
    assert fc_mask_table[1, 1].tolist() == [False, False, False]
    assert fc_mask_table[1, 5].tolist() == [True, True, True]


def test_layer_fc_mask_table_shape_and_padding():
    assert fc_mask_table.shape == (LAYER_GROUP_COUNT, WYCKOFF_TYPES, 3)
    assert not jnp.any(fc_mask_table[:, 0, :])


def test_layer_fc_mask_matches_symop_free_coordinate_rows():
    for g in range(LAYER_GROUP_COUNT):
        for w in range(1, int(wmax_table[g]) + 1):
            op = symops[g, w, 0, :3, :3]
            fc_mask = jnp.abs(op).sum(axis=1) != 0
            assert jnp.array_equal(fc_mask, fc_mask_table[g, w])
