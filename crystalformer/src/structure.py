from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure

from crystalformer.src.layergroup import mult_table, symops


def _as_array(value, dtype=float):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return np.asarray(value, dtype=dtype)


def canonicalize_layer_z(coords):
    coords = np.asarray(coords, dtype=float).copy()
    coords[:, :2] %= 1.0
    coords[:, 2] += 0.5 - np.mean(coords[:, 2])
    return coords


def _validate_site_arrays(G, xyz, atoms, wyckoffs, multiplicities):
    expected_shape = wyckoffs.shape
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("X must be a two-dimensional array with three coordinates per site")
    if atoms.shape != expected_shape or xyz.shape[0] != expected_shape[0]:
        raise ValueError("X, A, and W lengths must agree")
    if multiplicities is not None and multiplicities.shape != expected_shape:
        raise ValueError("M length must agree with W length")

    expected_multiplicities = np.asarray(mult_table[int(G) - 1, wyckoffs], dtype=int)
    if multiplicities is not None and not np.array_equal(multiplicities, expected_multiplicities):
        raise ValueError("supplied multiplicities do not match layer-group multiplicity table")
    return expected_multiplicities


def expand_layer_sites(G, X, A, W, M=None):
    xyz = _as_array(X, dtype=float)
    atoms = _as_array(A, dtype=int)
    wyckoffs = _as_array(W, dtype=int)
    supplied_multiplicities = None if M is None else _as_array(M, dtype=int)
    multiplicities = _validate_site_arrays(G, xyz, atoms, wyckoffs, supplied_multiplicities)

    expanded_coords = []
    expanded_atoms = []
    for site_xyz, atom_number, wyckoff_index, multiplicity in zip(xyz, atoms, wyckoffs, multiplicities):
        if atom_number == 0 or wyckoff_index == 0:
            continue
        ops = np.asarray(symops[int(G) - 1, int(wyckoff_index), : int(multiplicity)])
        affine = np.array([site_xyz[0], site_xyz[1], site_xyz[2], 1.0])
        coords = ops @ affine
        coords[:, :2] %= 1.0
        expanded_coords.extend(coords.tolist())
        expanded_atoms.extend([int(atom_number)] * len(coords))

    if not expanded_coords:
        raise ValueError("sample row contains no non-padding sites")

    return np.asarray(expanded_coords, dtype=float), expanded_atoms


def row_to_structure(G, L, X, A, W, M=None):
    lattice_params = _as_array(L, dtype=float)
    coords, atoms = expand_layer_sites(int(G), X, A, W, M)
    coords = canonicalize_layer_z(coords)
    lattice = Lattice.from_parameters(*lattice_params.tolist(), pbc=[True, True, False])
    return Structure(lattice, atoms, coords)


def _structure_signature(structure, precision=8):
    lattice = tuple(np.round(structure.lattice.parameters, precision))
    sites = sorted(
        (
            int(site.specie.Z),
            *np.round(site.frac_coords, precision).tolist(),
        )
        for site in structure
    )
    return lattice, tuple(sites)


def write_structures_from_samples(sample_csv, output_dir, deduplicate=False):
    sample_csv = Path(sample_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "structures.csv"
    accepted_signatures = set()
    accepted_count = 0

    with sample_csv.open(newline="") as input_handle, output_csv.open("w", newline="") as output_handle:
        reader = csv.DictReader(input_handle)
        writer = csv.DictWriter(output_handle, fieldnames=["index", "cif", "structure_json", "source_row"])
        writer.writeheader()
        for row_index, row in enumerate(reader):
            structure = row_to_structure(row["G"], row["L"], row["X"], row["A"], row["W"], row.get("M"))
            signature = _structure_signature(structure)
            if deduplicate and signature in accepted_signatures:
                continue
            accepted_signatures.add(signature)
            cif_path = output_dir / f"structure_{accepted_count:06d}.cif"
            json_path = output_dir / f"structure_{accepted_count:06d}.json"
            structure.to(filename=str(cif_path), fmt="cif")
            with json_path.open("w") as json_handle:
                json.dump(structure.as_dict(), json_handle)
            writer.writerow(
                {
                    "index": accepted_count,
                    "cif": str(cif_path),
                    "structure_json": str(json_path),
                    "source_row": row_index,
                }
            )
            accepted_count += 1

    return output_csv
