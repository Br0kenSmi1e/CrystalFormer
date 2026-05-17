import jax
import jax.numpy as jnp

from crystalformer.src.von_mises import normal_logpdf, sample_normal, von_mises_logpdf


def test_xy_von_mises_logpdf_is_periodic():
    loc = jnp.array(0.1)
    kappa = jnp.array(4.0)
    x = jnp.array(0.7)

    assert jnp.allclose(
        von_mises_logpdf(x, loc, kappa),
        von_mises_logpdf(x + 2.0 * jnp.pi, loc, kappa),
    )


def test_z_normal_logpdf_is_not_periodic():
    loc = jnp.array(0.1)
    precision = jnp.array(4.0)
    z = jnp.array(0.7)

    assert normal_logpdf(z, loc, precision) > normal_logpdf(z + 1.0, loc, precision)


def test_normal_logpdf_matches_closed_form():
    loc = jnp.array(0.1)
    precision = jnp.array(4.0)
    x = jnp.array(0.7)
    expected = -0.5 * (
        jnp.log(2.0 * jnp.pi)
        - jnp.log(precision)
        + precision * (x - loc) * (x - loc)
    )

    assert jnp.allclose(normal_logpdf(x, loc, precision), expected)


def test_sample_normal_uses_precision():
    key = jax.random.PRNGKey(0)
    samples = sample_normal(key, loc=jnp.array(0.5), precision=jnp.array(16.0), shape=(2048,))

    assert samples.shape == (2048,)
    assert abs(float(jnp.mean(samples)) - 0.5) < 0.05
    assert 0.15 < float(jnp.std(samples)) < 0.35
