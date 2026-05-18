import pytest
import sys
import subprocess
from pathlib import Path
from ast import literal_eval

import jax.numpy as jnp
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as cli_main
from main import build_parser, validate_args


def test_help_mentions_layergroup_not_spacegroup(capsys):
    parser = build_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])

    captured = capsys.readouterr()
    assert exc.value.code == 0
    assert "--data" not in captured.out
    assert "--train_path" in captured.out
    assert "--valid_path" in captured.out
    assert "--layergroup" in captured.out
    assert "--spacegroup" not in captured.out
    assert "--formula" not in captured.out
    assert "--sym_group" not in captured.out


def test_sampling_requires_layergroup():
    parser = build_parser()
    args = parser.parse_args(["--optimizer", "none", "--restore_path", "checkpoint.pkl"])

    with pytest.raises(SystemExit) as exc:
        validate_args(args)

    assert "--layergroup is required when --optimizer none" in str(exc.value)


def test_sampling_requires_layergroup_in_range():
    parser = build_parser()
    args = parser.parse_args(
        ["--optimizer", "none", "--restore_path", "checkpoint.pkl", "--layergroup", "81"]
    )

    with pytest.raises(SystemExit) as exc:
        validate_args(args)

    assert "--layergroup must be in [1, 80]" in str(exc.value)


def test_unsupported_upstream_options_are_unrecognized():
    parser = build_parser()

    for option in ("--data", "--spacegroup", "--formula", "--K", "--sym_group"):
        with pytest.raises(SystemExit):
            parser.parse_args([option, "1"])


def test_sampling_requires_usable_checkpoint(monkeypatch, tmp_path):
    def fail_make_transformer(*_args, **_kwargs):
        raise AssertionError("sampling should fail before model initialization")

    monkeypatch.setattr(cli_main.checkpoint, "find_ckpt_filename", lambda _path: (None, 0))
    monkeypatch.setattr(cli_main, "make_transformer", fail_make_transformer)

    with pytest.raises(SystemExit) as exc:
        cli_main.main(
            [
                "--optimizer",
                "none",
                "--restore_path",
                str(tmp_path / "missing.pkl"),
                "--save_path",
                str(tmp_path),
                "--layergroup",
                "1",
            ]
        )

    assert "No checkpoint found at --restore_path" in str(exc.value)


def test_sampling_rejects_checkpoint_without_params_before_model_init(monkeypatch, tmp_path):
    def fail_make_transformer(*_args, **_kwargs):
        raise AssertionError("sampling should fail before model initialization")

    monkeypatch.setattr(cli_main.checkpoint, "find_ckpt_filename", lambda _path: ("epoch_000001.pkl", 1))
    monkeypatch.setattr(cli_main.checkpoint, "load_data", lambda _path: {"opt_state": object()})
    monkeypatch.setattr(cli_main, "make_transformer", fail_make_transformer)

    with pytest.raises(SystemExit) as exc:
        cli_main.main(
            [
                "--optimizer",
                "none",
                "--restore_path",
                str(tmp_path / "epoch_000001.pkl"),
                "--save_path",
                str(tmp_path),
                "--layergroup",
                "1",
            ]
        )

    assert "missing params" in str(exc.value)


def test_sampling_rejects_unloadable_checkpoint_before_model_init(monkeypatch, tmp_path):
    def fail_make_transformer(*_args, **_kwargs):
        raise AssertionError("sampling should fail before model initialization")

    monkeypatch.setattr(cli_main.checkpoint, "find_ckpt_filename", lambda _path: ("epoch_000001.pkl", 1))
    monkeypatch.setattr(cli_main.checkpoint, "load_data", lambda _path: (_ for _ in ()).throw(ValueError("bad pickle")))
    monkeypatch.setattr(cli_main, "make_transformer", fail_make_transformer)

    with pytest.raises(SystemExit) as exc:
        cli_main.main(
            [
                "--optimizer",
                "none",
                "--restore_path",
                str(tmp_path / "epoch_000001.pkl"),
                "--save_path",
                str(tmp_path),
                "--layergroup",
                "1",
            ]
        )

    assert "Failed to load checkpoint" in str(exc.value)


def test_training_rejects_non_layergroup_data_before_train(monkeypatch, tmp_path):
    def fail_train(*_args, **_kwargs):
        raise AssertionError("training should fail before train()")

    monkeypatch.setattr(cli_main.multiprocessing, "cpu_count", lambda: 1)
    monkeypatch.setattr(
        cli_main,
        "GLXYZAW_from_file",
        lambda *_args: (
            jnp.array([225]),
            jnp.zeros((1, 6)),
            jnp.zeros((1, 2, 3)),
            jnp.zeros((1, 2), dtype=int),
            jnp.zeros((1, 2), dtype=int),
        ),
    )
    monkeypatch.setattr(cli_main, "make_transformer", lambda *_args: (jnp.array([0.0]), object()))
    monkeypatch.setattr(cli_main, "make_loss_fn", lambda *_args: (object(), object()))
    monkeypatch.setattr(cli_main.checkpoint, "find_ckpt_filename", lambda _path: (None, 0))
    monkeypatch.setattr(cli_main, "train", fail_train)

    with pytest.raises(SystemExit) as exc:
        cli_main.main(
            [
                "--train_path",
                "legacy.csv",
                "--valid_path",
                "valid.csv",
                "--save_path",
                str(tmp_path),
            ]
        )

    assert "CrystalFormer-2D training data must use layergroup values in [1, 80]" in str(exc.value)


def test_training_requires_explicit_train_and_valid_paths():
    parser = build_parser()

    with pytest.raises(SystemExit) as exc:
        validate_args(parser.parse_args([]))
    assert "--train_path is required when training" in str(exc.value)

    with pytest.raises(SystemExit) as exc:
        validate_args(parser.parse_args(["--train_path", "train.csv"]))
    assert "--valid_path is required when training" in str(exc.value)


def test_training_loads_distinct_train_and_valid_paths(monkeypatch, tmp_path):
    paths = []

    def fake_GLXYZAW_from_file(path, *_args):
        paths.append(path)
        group = 1 if path == "train.csv" else 2
        return (
            jnp.array([group]),
            jnp.zeros((1, 6)),
            jnp.zeros((1, 2, 3)),
            jnp.zeros((1, 2), dtype=int),
            jnp.zeros((1, 2), dtype=int),
        )

    def fake_train(
        _key,
        _optimizer,
        _opt_state,
        _loss_fn,
        params,
        _epoch_finished,
        _epochs,
        _batchsize,
        train_data,
        valid_data,
        _output_path,
        val_interval,
        _cfg_drop_prob,
    ):
        assert train_data[0][0] == 1
        assert valid_data[0][0] == 2
        assert val_interval == 7
        return params, object()

    monkeypatch.setattr(cli_main.multiprocessing, "cpu_count", lambda: 1)
    monkeypatch.setattr(cli_main, "GLXYZAW_from_file", fake_GLXYZAW_from_file)
    monkeypatch.setattr(cli_main, "make_transformer", lambda *_args: (jnp.array([0.0]), object()))
    monkeypatch.setattr(cli_main, "make_loss_fn", lambda *_args: (lambda *_loss_args: (0.0, (0.0, 0.0, 0.0, 0.0)), object()))
    monkeypatch.setattr(cli_main.checkpoint, "find_ckpt_filename", lambda _path: (None, 0))
    monkeypatch.setattr(cli_main, "train", fake_train)

    cli_main.main(
        [
            "--train_path",
            "train.csv",
            "--valid_path",
            "valid.csv",
            "--save_path",
            str(tmp_path),
            "--batchsize",
            "1",
            "--val_interval",
            "7",
        ]
    )

    assert paths == ["train.csv", "valid.csv"]


def test_sampling_csv_batches_and_normalizes_lattice_for_logp(monkeypatch, tmp_path):
    captured_lattices = []
    captured_groups = []
    captured_wyckoffs = []
    physical_lattice = jnp.array([3.0, 4.0, 20.0, 90.0, 90.0, 120.0])

    monkeypatch.setattr(cli_main.multiprocessing, "cpu_count", lambda: 1)
    monkeypatch.setattr(cli_main.checkpoint, "find_ckpt_filename", lambda _path: ("epoch_000001.pkl", 1))
    monkeypatch.setattr(cli_main.checkpoint, "load_data", lambda _path: {"params": jnp.array([1.0])})
    monkeypatch.setattr(cli_main, "make_transformer", lambda *_args: (jnp.array([0.0]), object()))
    monkeypatch.setattr(cli_main.jax, "jit", lambda fn, **_kwargs: fn)

    def fake_make_loss_fn(*_args):
        def _loss_fn(*_loss_args):
            raise AssertionError("training loss should not be used in sampling mode")

        def logp_fn(_params, _key, G, L, XYZ, A, W, _is_train):
            captured_groups.append(G)
            captured_wyckoffs.append(W)
            captured_lattices.append(L)
            return (
                jnp.zeros(G.shape[0]),
                jnp.zeros(G.shape[0]),
                jnp.zeros(G.shape[0]),
                jnp.zeros(G.shape[0]),
            )

        return _loss_fn, logp_fn

    def fake_make_sample_crystal(*_args, **_kwargs):
        def sampler(_key, _params, batchsize):
            groups = jnp.ones((batchsize,), dtype=int)
            wyckoffs = jnp.tile(jnp.array([[1, 2]]), (batchsize, 1))
            return (
                groups,
                jnp.zeros((batchsize, 2, 3)),
                jnp.tile(jnp.array([[1, 0]]), (batchsize, 1)),
                wyckoffs,
                jnp.tile(jnp.array([[1, 2]]), (batchsize, 1)),
                jnp.tile(physical_lattice[None, :], (batchsize, 1)),
            )

        return sampler

    monkeypatch.setattr(cli_main, "make_loss_fn", fake_make_loss_fn)
    monkeypatch.setattr(cli_main, "make_sample_crystal", fake_make_sample_crystal)

    cli_main.main(
        [
            "--optimizer",
            "none",
            "--restore_path",
            "epoch_000001.pkl",
            "--save_path",
            str(tmp_path),
            "--layergroup",
            "1",
            "--num_samples",
            "5",
            "--batchsize",
            "2",
        ]
    )

    output = pd.read_csv(tmp_path / "samples.csv")
    assert len(output) == 5
    assert list(output.columns) == ["G", "L", "X", "A", "W", "M", "logp_w", "logp_xyz", "logp_a", "logp_l"]
    assert literal_eval(output.loc[0, "L"]) == [3.0, 4.0, 20.0, 90.0, 90.0, 120.0]

    for G, W, L_for_logp in zip(captured_groups, captured_wyckoffs, captured_lattices):
        expected_lattice_for_logp = cli_main.norm_lattice(G, W, jnp.tile(physical_lattice[None, :], (G.shape[0], 1)))
        assert jnp.allclose(L_for_logp, expected_lattice_for_logp)
        assert not jnp.allclose(L_for_logp, jnp.tile(physical_lattice[None, :], (G.shape[0], 1)))


from crystalformer.cli import classifier, cond_gen, dataset, train_dpo, train_ppo


def test_dataset_cli_is_layer_phase_one(capsys):
    parser = dataset.build_parser()
    args = parser.parse_args(["--path", "data"])

    assert args.wyck_types == 19
    assert args.num_workers == 1

    with pytest.raises(SystemExit) as exc:
        dataset.main(["--help"])

    captured = capsys.readouterr()
    assert exc.value.code == 0
    assert "CrystalFormer-2D layer CSV preprocessing" in captured.out


def test_dataset_help_does_not_import_preprocessing_stack():
    command = [
        sys.executable,
        "-c",
        "\n".join(
            [
                "import sys",
                "from crystalformer.cli import dataset",
                "try:",
                "    dataset.main(['--help'])",
                "except SystemExit:",
                "    pass",
                "for name in ['pandas', 'spglib', 'pyxtal', 'pymatgen', 'lmdb']:",
                "    print(name, name in sys.modules)",
            ]
        ),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)

    assert "pandas False" in result.stdout
    assert "spglib False" in result.stdout
    assert "pyxtal False" in result.stdout
    assert "pymatgen False" in result.stdout
    assert "lmdb False" in result.stdout


def test_unsupported_phase_one_commands_exit_with_clear_messages():
    commands = [
        (train_ppo.main, "PPO"),
        (train_dpo.main, "DPO"),
        (classifier.main, "classifier"),
        (cond_gen.main, "formula-conditioned generation"),
    ]

    for command, label in commands:
        with pytest.raises(SystemExit) as exc:
            command([])
        assert label in str(exc.value)
        assert "not part of CrystalFormer-2D phase 1" in str(exc.value)
