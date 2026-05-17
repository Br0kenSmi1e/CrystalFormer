from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


def test_project_metadata_names_crystalformer_2d():
    metadata = tomllib.loads(Path("pyproject.toml").read_text())["project"]

    assert metadata["name"] == "crystalformer-2d"
    assert metadata["version"] == "0.1.0"
    assert metadata["description"] == "CrystalFormer-2D: fixed layer-group generation for 2D crystals"
    assert metadata["readme"] == "README.md"
    assert metadata["requires-python"] == ">=3.10"
    assert metadata["license"] == {"file": "LICENSE"}
    assert metadata["keywords"] == ["2d-materials", "layer-group", "crystal-generation"]


def test_project_dependencies_are_phase_one_requirements():
    metadata = tomllib.loads(Path("pyproject.toml").read_text())["project"]

    assert metadata["dependencies"] == [
        "dm-haiku==0.0.15",
        "jax==0.6.2",
        "jaxlib==0.6.2",
        "numpy",
        "optax==0.2.6",
        "pandas",
        "pymatgen",
        "pyxtal==1.1.1",
        "spglib",
        "tqdm",
    ]


def test_project_exposes_only_phase_one_console_scripts():
    metadata = tomllib.loads(Path("pyproject.toml").read_text())["project"]

    assert metadata["scripts"] == {
        "crystalformer-2d-dataset": "crystalformer.cli.dataset:main",
        "crystalformer-2d-awl2struct": "crystalformer.cli.awl2struct_2d:main",
    }


def test_layer_group_data_files_are_packaged():
    packaging = tomllib.loads(Path("pyproject.toml").read_text())["tool"]["setuptools"]
    data_dir = Path("crystalformer/data")

    assert packaging["packages"]["find"] == {"where": ["."], "include": ["crystalformer*"]}
    assert packaging["package-data"] == {"crystalformer": ["data/*.csv"]}
    assert (data_dir / "layer_list.csv").is_file()
    assert (data_dir / "layer_symbols.csv").is_file()
    assert "Layer Group,Wyckoff Positions" in (data_dir / "layer_list.csv").read_text().splitlines()[0]


def test_active_core_modules_do_not_import_spacegroup_or_formula_helpers():
    active_core_paths = [
        Path("main.py"),
        Path("crystalformer/src/train.py"),
        Path("crystalformer/src/transformer.py"),
        Path("crystalformer/src/loss.py"),
        Path("crystalformer/src/sample.py"),
        Path("crystalformer/src/utils.py"),
    ]
    forbidden = [
        "crystalformer.src." + name for name in ("wyckoff", "formula")
    ] + ["topk_" + "recall"]

    for path in active_core_paths:
        text = path.read_text()
        for pattern in forbidden:
            assert pattern not in text, f"{path} still references {pattern}"
