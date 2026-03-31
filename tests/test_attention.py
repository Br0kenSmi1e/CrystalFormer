"""Tests for MultiHeadAttention -- focuses on the w_init=None fix.

Before the fix, calling MultiHeadAttention with w_init=None (the default)
raised NameError: name 'w_init_scale' is not defined.
"""
from config import *

from crystalformer.src.attention import MultiHeadAttention


def test_w_init_none_does_not_raise():
    """w_init=None (the default) must not raise NameError.

    Regression test for: NameError: name 'w_init_scale' is not defined.
    The fix replaces the undefined variable with the literal value 1.0,
    matching the upstream haiku default for VarianceScaling.
    """
    def fn(q, k, v):
        mha = MultiHeadAttention(
            num_heads=2,
            key_size=8,
            model_size=16,
            w_init=None,  # default -- was broken before fix
        )
        return mha(q, k, v)

    f = hk.without_apply_rng(hk.transform(fn))
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, 16))
    params = f.init(key, x, x, x)
    out = f.apply(params, x, x, x)

    assert out.shape == (4, 16)
    assert jnp.isfinite(out).all(), "w_init=None path produces NaN/Inf"


def test_w_init_explicit_still_works():
    """Explicit w_init continues to work after the fix."""
    def fn(q, k, v):
        mha = MultiHeadAttention(
            num_heads=2,
            key_size=8,
            model_size=16,
            w_init=hk.initializers.VarianceScaling(1.0),
        )
        return mha(q, k, v)

    f = hk.without_apply_rng(hk.transform(fn))
    key = jax.random.PRNGKey(1)
    x = jax.random.normal(key, (4, 16))
    params = f.init(key, x, x, x)
    out = f.apply(params, x, x, x)

    assert out.shape == (4, 16)
    assert jnp.isfinite(out).all()
