import jax
import jax.numpy as jnp

from crystalformer.src.layergroup import WYCKOFF_TYPES
from crystalformer.src.sample import make_sample_crystal, project_xyz, sample_z
from crystalformer.src.transformer import make_transformer


def test_project_xyz_wraps_xy_but_not_z():
    xyz = project_xyz(
        jnp.array(2),
        jnp.array(5),
        jnp.array([1.2, -0.3, 1.25]),
        0,
    )

    assert jnp.allclose(xyz, jnp.array([0.2, 0.7, 1.25]))


def test_sample_z_does_not_force_unit_interval():
    key = jax.random.PRNGKey(0)
    h_z = jnp.array([[10.0, -10.0, 1.5, 0.0, 1e6, 1.0]])

    _key, z = sample_z(
        key,
        h_z,
        Kx=2,
        top_p=1.0,
        temperature=1.0,
        batchsize=1,
    )

    assert z[0] > 1.0


def test_sample_crystal_uses_fixed_layergroup():
    key = jax.random.PRNGKey(0)
    n_max = 3
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
    sampler = make_sample_crystal(
        transformer,
        n_max=n_max,
        atom_types=atom_types,
        wyck_types=WYCKOFF_TYPES,
        Kx=3,
        Kl=2,
        w_mask=None,
        top_p=1.0,
        temperature=1.0,
        layergroup=66,
    )

    G, XYZ, A, W, M, L = sampler(key, params, batchsize=2)

    assert G.tolist() == [66, 66]
    assert XYZ.shape == (2, n_max, 3)
    assert A.shape == (2, n_max)
    assert W.shape == (2, n_max)
    assert M.shape == (2, n_max)
    assert L.shape == (2, 6)


def test_make_sample_crystal_rejects_invalid_layergroup():
    key = jax.random.PRNGKey(0)
    _params, transformer = make_transformer(
        key,
        Nf=2,
        Kx=3,
        Kl=2,
        n_max=2,
        h0_size=16,
        num_layers=1,
        num_heads=2,
        key_size=4,
        model_size=16,
        embed_size=8,
        atom_types=12,
        wyck_types=WYCKOFF_TYPES,
        dropout_rate=0.0,
        attn_dropout=0.0,
    )

    try:
        make_sample_crystal(
            transformer,
            n_max=2,
            atom_types=12,
            wyck_types=WYCKOFF_TYPES,
            Kx=3,
            Kl=2,
            w_mask=None,
            top_p=1.0,
            temperature=1.0,
            layergroup=81,
        )
    except ValueError as exc:
        assert "layergroup must be in [1, 80]" in str(exc)
    else:
        raise AssertionError("invalid layergroup was accepted")
