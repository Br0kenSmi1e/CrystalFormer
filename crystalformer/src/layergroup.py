from __future__ import annotations

import ast
import csv
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


LAYER_GROUP_COUNT = 80
WYCKOFF_TYPES = 19
MULTIPLICITY_LCM = 48

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LAYER_LIST_CSV = DATA_DIR / "layer_list.csv"


def from_xyz_str(xyz_str: str) -> jnp.ndarray:
    rot_matrix = np.zeros((3, 3), dtype=float)
    trans = np.zeros(3, dtype=float)
    tokens = xyz_str.strip().replace(" ", "").lower().split(",")
    re_rot = re.compile(r"([+-]?)([\d\.]*)/?([\d\.]*)([x-z])")
    re_trans = re.compile(r"([+-]?)([\d\.]+)/?([\d\.]*)(?![x-z])")

    for i, token in enumerate(tokens):
        for match in re_rot.finditer(token):
            factor = -1.0 if match.group(1) == "-" else 1.0
            if match.group(2):
                numerator = float(match.group(2))
                denominator = float(match.group(3)) if match.group(3) else 1.0
                factor *= numerator / denominator
            axis = ord(match.group(4)) - ord("x")
            rot_matrix[i, axis] += factor

        for match in re_trans.finditer(token):
            sign = -1.0 if match.group(1) == "-" else 1.0
            numerator = float(match.group(2))
            denominator = float(match.group(3)) if match.group(3) else 1.0
            trans[i] += sign * numerator / denominator

    return jnp.array(np.concatenate([rot_matrix, trans[:, None]], axis=1))


def _load_wyckoff_positions(csv_path: Path = LAYER_LIST_CSV) -> list[list[list[str]]]:
    rows: list[list[list[str]]] = []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(ast.literal_eval(row["Wyckoff Positions"]))

    if len(rows) != LAYER_GROUP_COUNT:
        raise ValueError(f"Expected 80 layer groups in {csv_path}, found {len(rows)}")
    return rows


def build_layer_tables(csv_path: Path = LAYER_LIST_CSV):
    wyckoff_positions = _load_wyckoff_positions(csv_path)
    symop_array = np.zeros((LAYER_GROUP_COUNT, WYCKOFF_TYPES, MULTIPLICITY_LCM, 3, 4), dtype=float)
    mult_array = np.zeros((LAYER_GROUP_COUNT, WYCKOFF_TYPES), dtype=int)
    wmax_array = np.zeros((LAYER_GROUP_COUNT,), dtype=int)
    dof0_array = np.ones((LAYER_GROUP_COUNT, WYCKOFF_TYPES), dtype=bool)
    fc_mask_array = np.zeros((LAYER_GROUP_COUNT, WYCKOFF_TYPES, 3), dtype=bool)

    for group_index, group_wyckoffs in enumerate(wyckoff_positions):
        wyckoffs = []
        for wyckoff in group_wyckoffs:
            wyckoffs.append([np.array(from_xyz_str(op)) for op in wyckoff])

        wyckoffs = wyckoffs[::-1]
        if len(wyckoffs) >= WYCKOFF_TYPES:
            raise ValueError(f"Layer group {group_index + 1} has too many Wyckoff positions")

        wmax_array[group_index] = len(wyckoffs)
        for w_index, wyckoff in enumerate(wyckoffs, start=1):
            ops = np.array(wyckoff)
            repeats = MULTIPLICITY_LCM // ops.shape[0]
            if repeats * ops.shape[0] != MULTIPLICITY_LCM:
                raise ValueError(
                    f"Layer group {group_index + 1}, Wyckoff index {w_index} "
                    f"has multiplicity {ops.shape[0]}, which does not divide {MULTIPLICITY_LCM}"
                )

            symop_array[group_index, w_index] = np.tile(ops, (repeats, 1, 1))
            mult_array[group_index, w_index] = ops.shape[0]
            dof0_array[group_index, w_index] = np.linalg.matrix_rank(ops[0, :3, :3]) == 0
            fc_mask_array[group_index, w_index] = np.abs(ops[0, :3, :3]).sum(axis=1) != 0

    return (
        jnp.array(symop_array),
        jnp.array(mult_array),
        jnp.array(wmax_array),
        jnp.array(dof0_array),
        jnp.array(fc_mask_array),
    )


symops, mult_table, wmax_table, dof0_table, fc_mask_table = build_layer_tables()


def symmetrize_atoms_padded(g: int, w: int, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    w_max = wmax_table[g - 1].astype(int)
    general_ops = symops[g - 1, w_max]
    affine_point = jnp.array([x[0], x[1], x[2], 1.0])
    coords = general_ops @ affine_point
    coords = coords - jnp.floor(coords)

    def dist_to_first_op(coord):
        projected = symops[g - 1, w, 0] @ jnp.array([coord[0], coord[1], coord[2], 1.0])
        diff = projected - coord
        diff = diff - jnp.rint(diff)
        return jnp.sum(diff**2)

    loc = jnp.argmin(jax.vmap(dist_to_first_op)(coords))
    generator = coords[loc].reshape(3)
    m = mult_table[g - 1, w].astype(int)
    ops = symops[g - 1, w]
    expanded = ops @ jnp.array([generator[0], generator[1], generator[2], 1.0])
    mask = jnp.arange(MULTIPLICITY_LCM) < m
    return expanded - jnp.floor(expanded), mask


def symmetrize_atoms(g: int, w: int, x: jnp.ndarray) -> jnp.ndarray:
    coords, _ = symmetrize_atoms_padded(g, w, x)
    g_index = int(g) - 1
    w_index = int(w)
    m = int(mult_table[g_index, w_index])
    return coords[:m]
