import jax
import jax.numpy as jnp

from crystalformer.src.layergroup import (
    LAYER_GROUP_COUNT,
    WYCKOFF_TYPES,
    mult_table,
    wmax_table,
)
from crystalformer.src.transformer import make_transformer


def _make_small_transformer(key, n_max=4, atom_types=12):
    return make_transformer(
        key,
        Nf=2,
        Kx=3,
        Kl=2,
        n_max=n_max,
        h0_size=16,
        num_layers=1,
        num_heads=2,
        key_size=4,
        model_size=16,
        embed_size=8,
        atom_types=atom_types,
        wyck_types=WYCKOFF_TYPES,
        dropout_rate=0.0,
        attn_dropout=0.0,
    )


def _example_inputs(n_max=4, group=2):
    G = jnp.array(group)
    W = jnp.array([5, 1, 0, 0])[:n_max]
    A = jnp.array([6, 8, 0, 0])[:n_max]
    XYZ = jnp.zeros((n_max, 3))
    M = mult_table[G - 1, W]
    return G, XYZ, A, W, M


def _flat_param_items(params):
    return [
        (f"{module}/{name}", value)
        for module, module_params in params.items()
        for name, value in module_params.items()
    ]


def test_transformer_returns_only_autoregressive_logits():
    key = jax.random.PRNGKey(0)
    n_max = 4
    params, transformer = _make_small_transformer(key, n_max=n_max)
    G, XYZ, A, W, M = _example_inputs(n_max=n_max)

    h = transformer(params, key, G, XYZ, A, W, M, False)

    assert h.shape[0] == 5 * n_max + 1
    assert h.ndim == 2


def test_transformer_uses_80_layer_group_embedding_table():
    key = jax.random.PRNGKey(0)
    params, _transformer = _make_small_transformer(key, n_max=2)

    assert params["network"]["g_embedding_table"].shape == (LAYER_GROUP_COUNT, 8)


def test_transformer_has_no_formula_or_group_prediction_parameters():
    key = jax.random.PRNGKey(0)
    params, _transformer = _make_small_transformer(key, n_max=2)
    flat_params = _flat_param_items(params)
    param_names = [name for name, _value in flat_params]

    forbidden_fragments = ("c_embedding_uncond", "c_embedding_cond", "c_embedding")
    assert not any(
        fragment in name
        for name in param_names
        for fragment in forbidden_fragments
    )
    assert not any(
        value.shape and value.shape[-1] == LAYER_GROUP_COUNT
        for name, value in flat_params
        if name != "network/g_embedding_table"
    )


def test_initial_h0_masks_unavailable_wyckoff_logits_for_layer_group():
    key = jax.random.PRNGKey(0)
    n_max = 4
    params, transformer = _make_small_transformer(key, n_max=n_max)
    G, XYZ, A, W, M = _example_inputs(n_max=n_max, group=1)
    w_max = int(wmax_table[G - 1])

    h = transformer(params, key, G, XYZ, A, W, M, False)
    h0_w_logits = h[0, :WYCKOFF_TYPES]

    assert jnp.isfinite(h0_w_logits[1 : w_max + 1]).any()
    assert jnp.all(h0_w_logits[w_max + 1 :] < -1e8)


def test_output_row_ordering_matches_autoregressive_token_layout():
    key = jax.random.PRNGKey(0)
    n_max = 4
    params, transformer = _make_small_transformer(key, n_max=n_max)
    G, XYZ, A, W, M = _example_inputs(n_max=n_max, group=1)
    w_max = int(wmax_table[G - 1])

    h = transformer(params, key, G, XYZ, A, W, M, False)
    wyckoff_rows = h[0::5]
    atom_lattice_rows = h[1::5]
    x_rows = h[2::5]
    y_rows = h[3::5]
    z_rows = h[4::5]

    assert wyckoff_rows.shape[0] == n_max + 1
    assert atom_lattice_rows.shape[0] == n_max
    assert x_rows.shape == y_rows.shape == z_rows.shape == (n_max, h.shape[1])
    assert jnp.isfinite(wyckoff_rows[0, 1 : w_max + 1]).any()
    assert jnp.all(wyckoff_rows[0, w_max + 1 : WYCKOFF_TYPES] < -1e8)


def test_z_embedding_is_not_periodic_under_unit_shift():
    key = jax.random.PRNGKey(0)
    n_max = 4
    params, transformer = _make_small_transformer(key, n_max=n_max)
    G, XYZ, A, W, M = _example_inputs(n_max=n_max)
    shifted_XYZ = XYZ.at[0, 2].add(1.0)

    h = transformer(params, key, G, XYZ, A, W, M, False)
    shifted_h = transformer(params, key, G, shifted_XYZ, A, W, M, False)

    assert not jnp.allclose(h[5], shifted_h[5])
