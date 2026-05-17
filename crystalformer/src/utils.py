import ast
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import spglib
from pyxtal.lattice import Lattice as PyXtalLattice
from pymatgen.core import Structure

from crystalformer.src.layergroup import WYCKOFF_TYPES, wmax_table


def letter_to_number(letter: str) -> int:
    if not letter or not letter.isalpha():
        raise ValueError(f"Invalid Wyckoff letter: {letter!r}")
    return ord(letter.lower()) - ord("a") + 1


def _structure_from_cell_dict(value):
    if isinstance(value, Structure):
        return value
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
    else:
        parsed = value
    return Structure.from_dict(parsed)

def shuffle(key, data):
    """
    shuffle data along batch dimension
    """
    G, L, XYZ, A, W = data
    idx = jax.random.permutation(key, jnp.arange(len(L)))
    return G[idx], L[idx], XYZ[idx], A[idx], W[idx]


def process_layer_structure(structure, atom_types, wyck_types, n_max, tol=1e-5):
    if wyck_types != WYCKOFF_TYPES:
        raise ValueError(f"CrystalFormer-2D expects wyck_types={WYCKOFF_TYPES}")

    cell = (
        np.asarray(structure.lattice.matrix),
        np.asarray(structure.frac_coords),
        np.asarray([site.specie.Z for site in structure]),
    )
    dataset = spglib.get_layergroup(cell, aperiodic_dir=2, symprec=tol)
    if dataset is None:
        raise ValueError("spglib.get_layergroup could not assign a layer group")

    g = int(dataset.number)
    std_lattice = np.asarray(dataset.std_lattice)
    std_positions = np.asarray(dataset.std_positions)
    std_types = np.asarray(dataset.std_types)
    equivalent_atoms = np.asarray(dataset.equivalent_atoms)
    wyckoffs = np.asarray(dataset.wyckoffs)

    site_rows = []
    for orbit in sorted(set(equivalent_atoms.tolist())):
        indices = np.where(equivalent_atoms == orbit)[0]
        representative = int(indices[0])
        wyckoff_index = letter_to_number(str(wyckoffs[representative]))
        if wyckoff_index > int(wmax_table[g - 1]):
            raise ValueError(f"Layer group {g} does not contain Wyckoff index {wyckoff_index}")
        atom_number = int(std_types[representative])
        if atom_number >= atom_types:
            raise ValueError(f"Atomic number {atom_number} is outside atom_types={atom_types}")
        site_rows.append((representative, atom_number, wyckoff_index, std_positions[representative]))

    if len(site_rows) >= n_max:
        raise ValueError(
            f"Layer structure has {len(site_rows)} sites and requires n_max>{len(site_rows)} "
            "to reserve a padded lattice-token slot"
        )

    site_rows = sorted(
        site_rows,
        key=lambda row: (row[2], row[3][0], row[3][1], row[3][2], row[1], row[0]),
    )

    pyxtal_lattice = PyXtalLattice.from_matrix(std_lattice)
    natoms = max(len(std_types), 1)
    lengths = np.array([pyxtal_lattice.a, pyxtal_lattice.b, pyxtal_lattice.c]) / natoms ** (1.0 / 3.0)
    angles = np.array([pyxtal_lattice.alpha, pyxtal_lattice.beta, pyxtal_lattice.gamma])
    lattice = np.concatenate([lengths, angles])

    xyz = np.zeros((n_max, 3), dtype=float)
    atoms = np.zeros((n_max,), dtype=int)
    wyckoff_indices = np.zeros((n_max,), dtype=int)
    for row_index, (_representative, atom_number, wyckoff_index, position) in enumerate(site_rows):
        atoms[row_index] = atom_number
        wyckoff_indices[row_index] = wyckoff_index
        xyz[row_index] = position

    return g, lattice, xyz, atoms, wyckoff_indices


def GLXYZAW_from_file(csv_file, atom_types, wyck_types, n_max, num_workers=1):
    df = pd.read_csv(csv_file)
    if "structure" not in df.columns and "cif" not in df.columns:
        raise ValueError("Input CSV must contain either a 'structure' or 'cif' column")

    rows = []
    for _idx, row in df.iterrows():
        if "structure" in df.columns and not pd.isna(row["structure"]):
            structure = _structure_from_cell_dict(row["structure"])
        else:
            structure = Structure.from_str(row["cif"], fmt="cif")
        rows.append(process_layer_structure(structure, atom_types, wyck_types, n_max))

    G, L, XYZ, A, W = zip(*rows)
    return (
        np.asarray(G, dtype=int),
        np.asarray(L, dtype=float),
        np.asarray(XYZ, dtype=float),
        np.asarray(A, dtype=int),
        np.asarray(W, dtype=int),
    )
