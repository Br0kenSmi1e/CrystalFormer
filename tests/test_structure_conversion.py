import csv
import json
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Structure

from crystalformer.src.structure import canonicalize_layer_z, row_to_structure, write_structures_from_samples


def test_canonicalize_layer_z_keeps_slab_near_midplane():
    coords = np.array([[0.1, 0.2, 0.95], [0.3, 0.4, 0.05]])

    centered = canonicalize_layer_z(coords)

    assert centered.shape == coords.shape
    assert np.sum((centered[:, 2] - 0.5) ** 2) <= np.sum((coords[:, 2] - 0.5) ** 2)


def test_canonicalize_layer_z_does_not_wrap_nonperiodic_z():
    coords = np.array([[1.1, -0.2, 1.2], [-0.1, 1.3, -0.2]])

    centered = canonicalize_layer_z(coords)

    assert np.all((centered[:, :2] >= 0.0) & (centered[:, :2] < 1.0))
    assert centered[0, 2] > 1.0
    assert centered[1, 2] < 0.0


def test_canonicalize_layer_z_preserves_slab_thickness():
    coords = np.array([[0.1, 0.2, 1.6], [0.3, 0.4, -0.4]])

    centered = canonicalize_layer_z(coords)

    assert np.isclose(np.ptp(centered[:, 2]), np.ptp(coords[:, 2]))
    assert np.isclose(np.mean(centered[:, 2]), 0.5)


def test_row_to_structure_sets_2d_periodicity():
    structure = row_to_structure(
        G=1,
        L=[3.0, 4.0, 20.0, 90.0, 90.0, 90.0],
        X=[[0.2, 0.3, 0.5], [0.0, 0.0, 0.0]],
        A=[6, 0],
        W=[1, 0],
        M=[1, 0],
    )

    assert structure.lattice.pbc == (True, True, False)
    assert len(structure) == 1
    assert structure[0].specie.Z == 6


def test_row_to_structure_rejects_multiplicity_mismatch():
    with pytest.raises(ValueError, match="multiplicity"):
        row_to_structure(
            G=1,
            L=[3.0, 4.0, 20.0, 90.0, 90.0, 90.0],
            X=[[0.2, 0.3, 0.5]],
            A=[6],
            W=[1],
            M=[2],
        )


def test_row_to_structure_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        row_to_structure(
            G=1,
            L=[3.0, 4.0, 20.0, 90.0, 90.0, 90.0],
            X=[[0.2, 0.3, 0.5]],
            A=[6, 8],
            W=[1],
            M=[1],
        )


def test_row_to_structure_rejects_invalid_layer_group():
    with pytest.raises(ValueError, match="Layer group"):
        row_to_structure(
            G=0,
            L=[3.0, 4.0, 20.0, 90.0, 90.0, 90.0],
            X=[[0.2, 0.3, 0.5]],
            A=[6],
            W=[1],
            M=None,
        )


def test_row_to_structure_rejects_invalid_wyckoff_index():
    with pytest.raises(ValueError, match="Wyckoff"):
        row_to_structure(
            G=1,
            L=[3.0, 4.0, 20.0, 90.0, 90.0, 90.0],
            X=[[0.2, 0.3, 0.5]],
            A=[6],
            W=[2],
            M=None,
        )


def test_write_structures_from_samples_writes_csv_and_cif(tmp_path):
    sample_csv = tmp_path / "samples.csv"
    with sample_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["G", "L", "X", "A", "W", "M"])
        writer.writeheader()
        writer.writerow(
            {
                "G": "1",
                "L": "[3.0, 4.0, 20.0, 90.0, 90.0, 90.0]",
                "X": "[[0.2, 0.3, 0.5], [0.0, 0.0, 0.0]]",
                "A": "[6, 0]",
                "W": "[1, 0]",
                "M": "[1, 0]",
            }
        )

    output_csv = write_structures_from_samples(sample_csv, tmp_path / "out", deduplicate=False)

    assert Path(output_csv).is_file()
    assert (tmp_path / "out" / "structure_000000.cif").is_file()


def test_write_structures_from_samples_deduplicates_without_structure_matcher_crash(tmp_path):
    sample_csv = tmp_path / "samples.csv"
    with sample_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["G", "L", "X", "A", "W", "M"])
        writer.writeheader()
        row = {
            "G": "1",
            "L": "[3.0, 4.0, 20.0, 90.0, 90.0, 90.0]",
            "X": "[[0.2, 0.3, 0.5]]",
            "A": "[6]",
            "W": "[1]",
            "M": "[1]",
        }
        writer.writerow(row)
        writer.writerow(row)

    output_csv = write_structures_from_samples(sample_csv, tmp_path / "out", deduplicate=True)

    with Path(output_csv).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert (tmp_path / "out" / "structure_000000.cif").is_file()


def test_write_structures_from_samples_writes_json_sidecar_with_pbc(tmp_path):
    sample_csv = tmp_path / "samples.csv"
    with sample_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["G", "L", "X", "A", "W", "M"])
        writer.writeheader()
        writer.writerow(
            {
                "G": "1",
                "L": "[3.0, 4.0, 20.0, 90.0, 90.0, 90.0]",
                "X": "[[0.2, 0.3, 0.5]]",
                "A": "[6]",
                "W": "[1]",
                "M": "[1]",
            }
        )

    output_csv = write_structures_from_samples(sample_csv, tmp_path / "out", deduplicate=False)

    with Path(output_csv).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    json_path = Path(rows[0]["structure_json"])
    with json_path.open() as handle:
        structure = Structure.from_dict(json.load(handle))
    assert structure.lattice.pbc == (True, True, False)
