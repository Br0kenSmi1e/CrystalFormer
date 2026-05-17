import jax
import jax.numpy as jnp

from crystalformer.src.layergroup import WYCKOFF_TYPES, mult_table
from crystalformer.src.loss import make_loss_fn
from crystalformer.src.transformer import make_transformer


def test_loss_fn_returns_layer_components_without_group_logp():
    key = jax.random.PRNGKey(0)
    n_max = 4
    atom_types = 12
    params, transformer = make_transformer(
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
    loss_fn, logp_fn = make_loss_fn(n_max, atom_types, WYCKOFF_TYPES, Kx=3, Kl=2, transformer=transformer)
    G = jnp.array([2])
    W = jnp.array([[1, 5, 0, 0]])
    A = jnp.array([[6, 8, 0, 0]])
    M = mult_table[G[0] - 1, W[0]]
    L = jnp.array([[3.0, 4.0, 20.0, jnp.pi / 2, jnp.pi / 2, jnp.pi / 2]])
    XYZ = jnp.zeros((1, n_max, 3))

    logp = logp_fn(params, key, G, L, XYZ, A, W, True)
    value, components = loss_fn(params, key, G, L, XYZ, A, W, True)
    jit_logp = jax.jit(logp_fn, static_argnums=7)(params, key, G, L, XYZ, A, W, True)

    assert len(logp) == 4
    assert all(component.shape == (1,) for component in logp)
    assert all(jnp.all(jnp.isfinite(component)) for component in logp)
    assert len(components) == 4
    assert all(jnp.isfinite(component) for component in components)
    assert len(jit_logp) == 4
    assert all(component.shape == (1,) for component in jit_logp)
    assert jnp.isfinite(value)
    assert M.shape == (n_max,)
