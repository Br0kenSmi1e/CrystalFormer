import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

import crystalformer.src.utils as utils
from crystalformer.src.utils import GLXYZAW_from_file, process_layer_structure


def make_single_atom_layer_structure():
    lattice = Lattice.from_parameters(3.0, 4.0, 20.0, 90.0, 90.0, 90.0, pbc=[True, True, False])
    return Structure(lattice, ["C"], [[0.2, 0.3, 0.5]])


def test_process_layer_structure_returns_layer_tensors():
    structure = make_single_atom_layer_structure()

    g, lattice, xyz, atoms, wyckoffs = process_layer_structure(
        structure,
        atom_types=119,
        wyck_types=19,
        n_max=4,
    )

    assert 1 <= g <= 80
    assert lattice.shape == (6,)
    assert xyz.shape == (4, 3)
    assert atoms.shape == (4,)
    assert wyckoffs.shape == (4,)
    assert atoms[0] == 6
    assert np.count_nonzero(atoms) == 1


def test_process_layer_structure_requires_padding_slot():
    structure = make_single_atom_layer_structure()

    with pytest.raises(ValueError, match="requires n_max"):
        process_layer_structure(
            structure,
            atom_types=119,
            wyck_types=19,
            n_max=1,
        )


def test_process_layer_structure_rejects_non_layer_wyckoff_types():
    structure = make_single_atom_layer_structure()

    with pytest.raises(ValueError, match="wyck_types=19"):
        process_layer_structure(
            structure,
            atom_types=119,
            wyck_types=28,
            n_max=4,
        )


def test_process_layer_structure_sorts_sites_by_wyckoff(monkeypatch):
    class FakeDataset:
        number = 37
        std_lattice = np.diag([3.0, 4.0, 20.0])
        std_positions = np.array([[0.4, 0.2, 0.5], [0.1, 0.3, 0.5]])
        std_types = np.array([6, 8])
        equivalent_atoms = np.array([0, 1])
        wyckoffs = np.array(["d", "a"])

    monkeypatch.setattr(utils.spglib, "get_layergroup", lambda *args, **kwargs: FakeDataset())

    _g, _lattice, _xyz, atoms, wyckoffs = process_layer_structure(
        make_single_atom_layer_structure(),
        atom_types=119,
        wyck_types=19,
        n_max=4,
    )

    assert wyckoffs[:2].tolist() == [1, 4]
    assert atoms[:2].tolist() == [8, 6]


def test_GLXYZAW_from_file_reads_structure_column(tmp_path):
    structure = make_single_atom_layer_structure()
    csv_file = tmp_path / "layers.csv"
    import pandas as pd

    pd.DataFrame([{"structure": repr(structure.as_dict())}]).to_csv(csv_file, index=False)

    G, L, XYZ, A, W = GLXYZAW_from_file(csv_file, atom_types=119, wyck_types=19, n_max=4)

    assert G.shape == (1,)
    assert L.shape == (1, 6)
    assert XYZ.shape == (1, 4, 3)
    assert A.shape == (1, 4)
    assert W.shape == (1, 4)


def test_GLXYZAW_from_file_reads_cif_column(tmp_path):
    structure = make_single_atom_layer_structure()
    csv_file = tmp_path / "layers.csv"
    import pandas as pd

    pd.DataFrame([{"cif": structure.to(fmt="cif")}]).to_csv(csv_file, index=False)

    G, L, XYZ, A, W = GLXYZAW_from_file(csv_file, atom_types=119, wyck_types=19, n_max=4)

    assert G.shape == (1,)
    assert L.shape == (1, 6)
    assert XYZ.shape == (1, 4, 3)
    assert A.shape == (1, 4)
    assert W.shape == (1, 4)
