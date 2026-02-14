from __future__ import annotations

import sys
import math
import re
import numpy as np
from collections import deque
from typing import List, Tuple, Optional, TextIO, Dict, Union

NumberOrVar = Union[float, str]
def geom_rmsd(R1, R2, save=None):
    """
        構造R1と構造R2を比較し、各原子間の距離のRMSDが最小になるように重ね合わせるように形状を変えずにR2を動かす。
        R1-R2間の最小のRMSDと、並進・回転させたR2の新たなR2の構造をxyzで出力する。
        R1とR2は分子構造(string)かxyzファイルのパス。
        saveはオプションで、ファイルパスが指定してあればそのファイルにxyzを書き込む。
    """
    try:
        # xyz file
        R1 = _read_xyz_as_string(R1)
    except:
        # raw data
        R1 = R1
    try:
        R1_elements, R1_coordinates = _get_coord_from_R(R1)
    except:
        raise TypeError('Failed to load R1')
            
    try:
        # xyz file
        R2 = _read_xyz_as_string(R2)
    except:
        # raw data
        R2 = R2
    try:
        R2_elements, R2_coordinates = _get_coord_from_R(R2)
    except:
        raise TypeError('Failed to load R2')

    if R1_elements != R2_elements:
        raise ValueError(f'Mismatch.\nElements in R1: {R1_elements}\nElements in R2: {R2_elements}')
    # R1 and R2 successfully loaded
    R2_aligned = _superimpose(R1_coordinates, R2_coordinates)
    # RMSDを計算
    rmsd_value = _compute_rmsd(R1_coordinates, R2_aligned)
    R2_aligned = _create_R_from_coords(R2_elements, R2_aligned)
    xyz = f"{len(R2_elements)}\n"
    xyz += f"RMSD:  {rmsd_value}\n"
    xyz += f"{R2_aligned}"
    print(xyz)
    if save is not None:
        try:
            with open(save, 'w') as file: 
                file.write(xyz)
            print(f"xyz saved in {save}.")
        except:
            print(f"{save} is not a valid path.")

def _read_xyz_as_string(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        # 1行目の原子数と2行目の空行をスキップして座標部分のみ取得
        data_lines = lines[2:]
        
        # すべての行を1つの文字列に結合
        Q = "".join(data_lines)
        return Q


def _get_coord_from_R(R):
    elements = []
    coordinates = []
    for line in R.strip().split('\n'):
        parts = line.split()
        elements.append(parts[0])
        coordinates.append([float(parts[1]),
                            float(parts[2]),
                            float(parts[3])])

    return elements, np.array(coordinates)

def _create_R_from_coords(elements, coordinates):
    R = ""
    for element, (x, y, z) in zip(elements, coordinates):
        R += f"{element:<2} {x:>15.10f} {y:>15.10f} {z:>15.10f}\n"
    return R.strip()


def _compute_rmsd(V, W):
    """2つの構造のRMSDを計算"""
    return np.sqrt(np.mean(np.sum((V - W) ** 2, axis=1)))

def _run_kebsch(P, Q):
    """
    Kebschアルゴリズムを用いて2つの点群を回転させて重ね合わせる
    P: 参照構造の座標 (N x 3)
    Q: 重ね合わせたい構造の座標 (N x 3)
    """

    # 並進：質量中心を原点に移動
    P_centroid = np.mean(P, axis=0)
    Q_centroid = np.mean(Q, axis=0)
    P -= P_centroid
    Q -= Q_centroid

    # 回転：最適な回転行列を計算
    H = np.dot(P.T, Q)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # 回転行列の行列式が-1の場合、反射を防ぐために修正
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    return R, P_centroid, Q_centroid

def _superimpose(P, Q):
    """
    PをQに重ね合わせるための並進と回転を行う
    P: 参照構造の座標 (N x 3)
    Q: 重ね合わせたい構造の座標 (N x 3)
    """

    # Kabschアルゴリズムで最適な回転と並進を求める
    R, P_centroid, Q_centroid = _run_kebsch(P.copy(), Q.copy())

    # QをPに重ね合わせる
    Q_aligned = np.dot(Q - Q_centroid, R) + P_centroid

    return Q_aligned

# tlab/geom.py
# -*- coding: utf-8 -*-
"""
tlab.geom
=========
Geometry utilities for XYZ <-> Z-matrix conversion.

Public entry points (recommended)
---------------------------------
- xyz2zmat(xyz_path, ..., out=None) -> (zmat_lines, varmap)
- zmat2xyz(zmat_path, ..., out=None) -> (elems_array, coords)

Core functionality
------------------
- read_xyz / write_xyz
- read_zmatrix / write_zmatrix
- zmatrix_to_xyz
- xyz_to_zmatrix (heuristic)
- xyz_to_zmatrix_with_targets (force multiple bonds/angles/dihedrals)

Forced targets conventions (0-based in ORIGINAL XYZ)
----------------------------------------------------
- Bond (A,B): enforce distance(A-B) on atom B line with bond ref A
- Angle (A,B,C): enforce angle(A-B-C) on atom C line with bond ref B and angle ref A
- Dihedral (A,B,C,D): enforce dihedral(A-B-C-D) on atom D line with refs (C,B,A)

If constraints conflict / create cycles / impossible ordering, ValueError is raised.

Z-matrix file format (Gaussian-like subset)
-------------------------------------------
Atom lines:
  Elem
  Elem  i  R
  Elem  i  R  j  Angle
  Elem  i  R  j  Angle  k  Dihedral
where i,j,k are 1-based indices in the ZMAT order.

R/Angle/Dihedral may be float or variable name.
Variable definitions may appear later:
  VAR=number
"""


# ------------------------------------------------------------
# Covalent radii (Å) — extended set (heuristic for bond guessing)
# ------------------------------------------------------------
DEFAULT_RAD = 0.77

COV_RAD: Dict[str, float] = {
    # 1st row
    "H": 0.31, "He": 0.28,
    # 2nd row
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58,
    # 3rd row
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    # 4th row (incl. 3d)
    "K": 2.03, "Ca": 1.76,
    "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24,
    "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16,
    # 5th row (incl. 4d)
    "Rb": 2.20, "Sr": 1.95,
    "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54, "Tc": 1.47, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39,
    "Ag": 1.45, "Cd": 1.44,
    "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40,
    # 6th row (incl. 5d + lanthanides rep)
    "Cs": 2.44, "Ba": 2.15,
    "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01, "Pm": 1.99, "Sm": 1.98, "Eu": 1.98, "Gd": 1.96,
    "Tb": 1.94, "Dy": 1.92, "Ho": 1.92, "Er": 1.89, "Tm": 1.90, "Yb": 1.87, "Lu": 1.87,
    "Hf": 1.78, "Ta": 1.70, "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41, "Pt": 1.36,
    "Au": 1.36, "Hg": 1.32,
    "Tl": 1.45, "Pb": 1.46, "Bi": 1.48, "Po": 1.40, "At": 1.50, "Rn": 1.50,
}

def cov_rad(sym: str) -> float:
    return COV_RAD.get(sym, DEFAULT_RAD)

# ------------------------------------------------------------
# XYZ I/O
# ------------------------------------------------------------
def read_xyz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r") as f:
        n = int(f.readline().strip())
        _ = f.readline()
        elems, coords = [], []
        for _ in range(n):
            parts = f.readline().split()
            if len(parts) < 4:
                raise ValueError(f"Invalid XYZ line: {' '.join(parts)}")
            elems.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(elems, dtype=str), np.array(coords, dtype=float)

def write_xyz(
    elems: np.ndarray,
    coords: np.ndarray,
    file: Optional[TextIO] = None,
    comment: str = "Generated by tlab.geom",
) -> None:
    if file is None:
        file = sys.stdout
    file.write(f"{len(elems)}\n")
    file.write(f"{comment}\n")
    for e, c in zip(elems, coords):
        file.write(f"{e:2s}  {c[0]: .10f}  {c[1]: .10f}  {c[2]: .10f}\n")

# ------------------------------------------------------------
# Geometry primitives
# ------------------------------------------------------------
def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-15:
        return 0.0
    cosang = float(np.dot(ba, bc) / denom)
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))

def dihedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    b0 = a - b
    b1 = c - b
    b2 = d - c
    n1 = np.linalg.norm(b1)
    if n1 < 1e-15:
        return 0.0
    b1n = b1 / n1
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1n, v), w))
    return math.degrees(math.atan2(y, x))

# ------------------------------------------------------------
# Bond guessing
# ------------------------------------------------------------
def build_bond_graph(elems: np.ndarray, coords: np.ndarray, scale: float = 1.25) -> List[List[int]]:
    n = len(elems)
    G: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        ri = cov_rad(str(elems[i]))
        for j in range(i + 1, n):
            rj = cov_rad(str(elems[j]))
            cutoff = scale * (ri + rj)
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d <= cutoff:
                G[i].append(j)
                G[j].append(i)
    return G

# ------------------------------------------------------------
# Z-matrix parsing / writing
# ------------------------------------------------------------
_var_def_re = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([-+0-9.eE]+)\s*$")

def write_zmatrix(zmat_lines: List[str], file: Optional[TextIO] = None) -> None:
    if file is None:
        file = sys.stdout
    for ln in zmat_lines:
        file.write(ln.rstrip() + "\n")

def _to_number_or_var(tok: str) -> NumberOrVar:
    try:
        return float(tok)
    except ValueError:
        return tok

def read_zmatrix(path: str) -> Tuple[List[str], List[List[int]], List[List[NumberOrVar]], Dict[str, float]]:
    with open(path, "r") as f:
        raw_lines = f.readlines()

    elems: List[str] = []
    refs: List[List[int]] = []
    vals: List[List[NumberOrVar]] = []
    varmap: Dict[str, float] = {}

    for raw in raw_lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        m = _var_def_re.match(line)
        if m:
            varmap[m.group(1)] = float(m.group(2))
            continue

        parts = line.split()
        if len(parts) not in (1, 3, 5, 7):
            raise ValueError(f"Unsupported Z-matrix line:\n{raw}")

        elems.append(parts[0])
        if len(parts) == 1:
            refs.append([])
            vals.append([])
        elif len(parts) == 3:
            refs.append([int(parts[1]) - 1])
            vals.append([_to_number_or_var(parts[2])])
        elif len(parts) == 5:
            refs.append([int(parts[1]) - 1, int(parts[3]) - 1])
            vals.append([_to_number_or_var(parts[2]), _to_number_or_var(parts[4])])
        else:
            refs.append([int(parts[1]) - 1, int(parts[3]) - 1, int(parts[5]) - 1])
            vals.append([_to_number_or_var(parts[2]), _to_number_or_var(parts[4]), _to_number_or_var(parts[6])])

    return elems, refs, vals, varmap

def _resolve(x: NumberOrVar, varmap: Optional[Dict[str, float]]) -> float:
    if isinstance(x, float) or isinstance(x, int):
        return float(x)
    if varmap is None:
        raise ValueError(f"Encountered variable '{x}' but varmap is None")
    if x not in varmap:
        raise ValueError(f"Variable '{x}' not found in varmap")
    return float(varmap[x])

# ------------------------------------------------------------
# Z-matrix -> XYZ
# ------------------------------------------------------------
def zmatrix_to_xyz(
    elems: List[str],
    refs: List[List[int]],
    vals: List[List[NumberOrVar]],
    varmap: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    n = len(elems)
    coords: List[np.ndarray] = []

    for i in range(n):
        if i == 0:
            coords.append(np.array([0.0, 0.0, 0.0], dtype=float))
        elif i == 1:
            r = _resolve(vals[i][0], varmap)
            coords.append(np.array([0.0, 0.0, r], dtype=float))
        elif i == 2:
            r = _resolve(vals[i][0], varmap)
            ang = math.radians(_resolve(vals[i][1], varmap))
            ia, ib = refs[i]
            a = coords[ia]
            b = coords[ib]
            ab = b - a
            nab = np.linalg.norm(ab)
            if nab < 1e-15:
                coords.append(np.array([r * math.sin(ang), 0.0, a[2] - r * math.cos(ang)], dtype=float))
            else:
                e1 = ab / nab
                tmp = np.array([1.0, 0.0, 0.0], dtype=float)
                if abs(np.dot(tmp, e1)) > 0.9:
                    tmp = np.array([0.0, 1.0, 0.0], dtype=float)
                ex = np.cross(e1, tmp)
                ex /= np.linalg.norm(ex)
                ez = e1
                v = r * (math.cos(ang) * (-ez) + math.sin(ang) * ex)
                coords.append(a + v)
        else:
            r = _resolve(vals[i][0], varmap)
            ang = math.radians(_resolve(vals[i][1], varmap))
            dih = math.radians(_resolve(vals[i][2], varmap))

            ia, ib, ic = refs[i]
            a = coords[ia]
            b = coords[ib]
            c = coords[ic]

            ba = a - b
            n_ba = np.linalg.norm(ba)
            if n_ba < 1e-15:
                coords.append(a + np.array([0.0, 0.0, r], dtype=float))
                continue
            e1 = ba / n_ba

            bc = c - b
            e2 = np.cross(bc, e1)
            n_e2 = np.linalg.norm(e2)
            if n_e2 < 1e-15:
                tmp = np.array([1.0, 0.0, 0.0], dtype=float)
                if abs(np.dot(tmp, e1)) > 0.9:
                    tmp = np.array([0.0, 1.0, 0.0], dtype=float)
                e2 = np.cross(tmp, e1)
                e2 /= np.linalg.norm(e2)
            else:
                e2 /= n_e2

            e3 = np.cross(e1, e2)

            dvec = r * (
                math.cos(ang) * (-e1)
                + math.sin(ang) * (math.cos(dih) * e3 + math.sin(dih) * e2)
            )
            coords.append(a + dvec)

    return np.vstack(coords)

# ------------------------------------------------------------
# XYZ -> Z-matrix (heuristic; kept for convenience)
# ------------------------------------------------------------
def xyz_to_zmatrix(
    elems: np.ndarray,
    coords: np.ndarray,
    scale: float = 1.25,
    header: bool = False,
) -> List[str]:
    G = build_bond_graph(elems, coords, scale=scale)
    n = len(G)
    visited = [False] * n
    order: List[int] = []

    for start in range(n):
        if visited[start]:
            continue
        q = deque([start])
        visited[start] = True
        while q:
            u = q.popleft()
            order.append(u)
            for v in G[u]:
                if not visited[v]:
                    visited[v] = True
                    q.append(v)

    lines: List[str] = []
    if header:
        lines += [
            "# Z-matrix generated from XYZ (heuristic bonding via covalent radii)",
            "# Format: Elem  ref1  R  ref2  Angle  ref3  Dihedral",
            "",
        ]

    for idx, i in enumerate(order):
        if idx == 0:
            lines.append(f"{elems[i]}")
        elif idx == 1:
            j = order[0]
            r = distance(coords[i], coords[j])
            lines.append(f"{elems[i]}  {j+1}  {r:.6f}")
        elif idx == 2:
            j = order[1]
            k = order[0]
            r = distance(coords[i], coords[j])
            ang = angle(coords[i], coords[j], coords[k])
            lines.append(f"{elems[i]}  {j+1}  {r:.6f}  {k+1}  {ang:.6f}")
        else:
            j = order[idx - 1]
            k = order[idx - 2]
            l = order[idx - 3]
            r = distance(coords[i], coords[j])
            ang = angle(coords[i], coords[j], coords[k])
            dih = dihedral(coords[i], coords[j], coords[k], coords[l])
            lines.append(f"{elems[i]}  {j+1}  {r:.6f}  {k+1}  {ang:.6f}  {l+1}  {dih:.6f}")

    return lines

# ------------------------------------------------------------
# Forced targets (multiple): bonds, angles, dihedrals
# ------------------------------------------------------------
class _ForcedSpec:
    def __init__(self):
        self.bond_ref: Optional[int] = None   # original index
        self.angle_ref: Optional[int] = None  # original index
        self.dihed_ref: Optional[int] = None  # original index
        self.r_name: Optional[str] = None
        self.a_name: Optional[str] = None
        self.d_name: Optional[str] = None

def _default_var_R(a: int, b: int) -> str:
    return f"R_{a+1}_{b+1}"

def _default_var_A(a: int, b: int, c: int) -> str:
    return f"A_{a+1}_{b+1}_{c+1}"

def _default_var_D(a: int, b: int, c: int, d: int) -> str:
    return f"PHI_{a+1}_{b+1}_{c+1}_{d+1}"

def _check_indices(n: int, idxs: Tuple[int, ...], label: str) -> None:
    for x in idxs:
        if not (0 <= x < n):
            raise ValueError(f"{label}: index out of range: {x} (n={n})")
    if len(set(idxs)) != len(idxs):
        raise ValueError(f"{label}: indices must be distinct: {idxs}")

def _toposort(n: int, edges: List[Tuple[int, int]]) -> List[int]:
    indeg = [0] * n
    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1

    q = deque([i for i in range(n) if indeg[i] == 0])
    out: List[int] = []
    while q:
        u = q.popleft()
        out.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(out) != n:
        raise ValueError("Specified internal-coordinate constraints create a cycle (impossible dependencies).")
    return out

def xyz_to_zmatrix_with_targets(
    elems: np.ndarray,
    coords: np.ndarray,
    bonds: Optional[List[Tuple[int, int]]] = None,
    angles: Optional[List[Tuple[int, int, int]]] = None,
    dihedrals: Optional[List[Tuple[int, int, int, int]]] = None,
    bond_names: Optional[List[str]] = None,
    angle_names: Optional[List[str]] = None,
    dihedral_names: Optional[List[str]] = None,
    scale: float = 1.25,
    header: bool = True,
) -> Tuple[List[str], Dict[str, float]]:
    n = len(elems)
    bonds = bonds or []
    angles = angles or []
    dihedrals = dihedrals or []

    if bond_names is not None and len(bond_names) != len(bonds):
        raise ValueError("bond_names length must match bonds length")
    if angle_names is not None and len(angle_names) != len(angles):
        raise ValueError("angle_names length must match angles length")
    if dihedral_names is not None and len(dihedral_names) != len(dihedrals):
        raise ValueError("dihedral_names length must match dihedrals length")

    forced: Dict[int, _ForcedSpec] = {}
    edges: List[Tuple[int, int]] = []
    varmap: Dict[str, float] = {}

    def get_spec(atom: int) -> _ForcedSpec:
        if atom not in forced:
            forced[atom] = _ForcedSpec()
        return forced[atom]

    # bonds: (A,B) -> enforce on B with bond_ref A
    for t, (A, B) in enumerate(bonds):
        _check_indices(n, (A, B), "bond")
        name = bond_names[t] if bond_names is not None else _default_var_R(A, B)
        spec = get_spec(B)
        if spec.bond_ref is not None and spec.bond_ref != A:
            raise ValueError(f"Conflicting bond_ref for atom {B+1}: {spec.bond_ref+1} vs {A+1}")
        spec.bond_ref = A
        spec.r_name = name
        edges.append((A, B))
        varmap[name] = distance(coords[A], coords[B])

    # angles: (A,B,C) -> enforce on C with bond_ref B, angle_ref A
    for t, (A, B, C) in enumerate(angles):
        _check_indices(n, (A, B, C), "angle")
        name = angle_names[t] if angle_names is not None else _default_var_A(A, B, C)
        spec = get_spec(C)
        if spec.bond_ref is not None and spec.bond_ref != B:
            raise ValueError(f"Conflicting bond_ref for atom {C+1}: {spec.bond_ref+1} vs {B+1}")
        if spec.angle_ref is not None and spec.angle_ref != A:
            raise ValueError(f"Conflicting angle_ref for atom {C+1}: {spec.angle_ref+1} vs {A+1}")
        spec.bond_ref = B
        spec.angle_ref = A
        spec.a_name = name
        edges.append((A, C))
        edges.append((B, C))
        varmap[name] = angle(coords[A], coords[B], coords[C])

    # dihedrals: (A,B,C,D) -> enforce on D with bond_ref C, angle_ref B, dihed_ref A
    for t, (A, B, C, D) in enumerate(dihedrals):
        _check_indices(n, (A, B, C, D), "dihedral")
        name = dihedral_names[t] if dihedral_names is not None else _default_var_D(A, B, C, D)
        spec = get_spec(D)
        if spec.bond_ref is not None and spec.bond_ref != C:
            raise ValueError(f"Conflicting bond_ref for atom {D+1}: {spec.bond_ref+1} vs {C+1}")
        if spec.angle_ref is not None and spec.angle_ref != B:
            raise ValueError(f"Conflicting angle_ref for atom {D+1}: {spec.angle_ref+1} vs {B+1}")
        if spec.dihed_ref is not None and spec.dihed_ref != A:
            raise ValueError(f"Conflicting dihed_ref for atom {D+1}: {spec.dihed_ref+1} vs {A+1}")
        spec.bond_ref = C
        spec.angle_ref = B
        spec.dihed_ref = A
        spec.d_name = name
        edges.append((A, D))
        edges.append((B, D))
        edges.append((C, D))
        varmap[name] = dihedral(coords[A], coords[B], coords[C], coords[D])

    order = _toposort(n, edges)
    pos_of: Dict[int, int] = {atom: i for i, atom in enumerate(order)}

    # early-line feasibility checks for forced atoms
    for atom, spec in forced.items():
        p = pos_of[atom]
        if spec.angle_ref is not None and p < 2:
            raise ValueError(f"Angle constraint on atom {atom+1} requires ZMAT position >=3.")
        if spec.dihed_ref is not None and p < 3:
            raise ValueError(f"Dihedral constraint on atom {atom+1} requires ZMAT position >=4.")

    # bond graph for fallback ref choices
    G = build_bond_graph(elems, coords, scale=scale)

    def _choose_fallback_refs(i_orig: int, emitted: List[int]) -> Tuple[int, int, int]:
        if len(emitted) < 3:
            raise RuntimeError("Need at least 3 emitted atoms for fallback refs")
        bonded = [x for x in G[i_orig] if x in emitted]
        bond_ref = bonded[0] if bonded else emitted[-1]
        cand2 = [x for x in G[bond_ref] if x in emitted and x != bond_ref]
        angle_ref = cand2[0] if cand2 else emitted[-2]
        dihed_ref = None
        for x in reversed(emitted):
            if x != bond_ref and x != angle_ref:
                dihed_ref = x
                break
        if dihed_ref is None:
            dihed_ref = emitted[-3]
        return bond_ref, angle_ref, dihed_ref

    lines: List[str] = []
    if header:
        lines += [
            "# Z-matrix generated from XYZ with forced internal coordinates (variables)",
            "# Indices in variable names are 1-based ORIGINAL XYZ indices.",
            "# ZMAT indices used on each line are 1-based in the ZMAT order.",
            "",
        ]

    emitted: List[int] = []

    for idx, i_orig in enumerate(order):
        spec = forced.get(i_orig)

        if idx == 0:
            lines.append(f"{elems[i_orig]}")
            emitted.append(i_orig)
            continue

        if idx == 1:
            j = spec.bond_ref if (spec and spec.bond_ref is not None) else order[0]
            r_tok: NumberOrVar = spec.r_name if (spec and spec.r_name) else float(distance(coords[i_orig], coords[j]))
            lines.append(f"{elems[i_orig]}  {pos_of[j]+1}  {r_tok}")
            emitted.append(i_orig)
            continue

        if idx == 2:
            j = spec.bond_ref if (spec and spec.bond_ref is not None) else order[1]
            k = spec.angle_ref if (spec and spec.angle_ref is not None) else order[0]
            r_tok = spec.r_name if (spec and spec.r_name) else float(distance(coords[i_orig], coords[j]))
            a_tok = spec.a_name if (spec and spec.a_name) else float(angle(coords[i_orig], coords[j], coords[k]))
            lines.append(f"{elems[i_orig]}  {pos_of[j]+1}  {r_tok}  {pos_of[k]+1}  {a_tok}")
            emitted.append(i_orig)
            continue

        # idx >= 3
        if spec:
            if spec.bond_ref is None:
                raise ValueError(f"Forced atom {i_orig+1} missing bond_ref (impossible).")
            j = spec.bond_ref
            k = spec.angle_ref
            l = spec.dihed_ref

            if (spec.a_name is not None or spec.d_name is not None) and k is None:
                raise ValueError(f"Forced atom {i_orig+1} needs angle_ref but missing.")
            if spec.d_name is not None and l is None:
                raise ValueError(f"Forced atom {i_orig+1} needs dihed_ref but missing.")

            if k is None or l is None:
                fbj, fbk, fbl = _choose_fallback_refs(i_orig, emitted)
                if k is None:
                    k = fbk if fbk != j else emitted[-2]
                if l is None:
                    cand = [x for x in reversed(emitted) if x != j and x != k]
                    if not cand:
                        raise ValueError(f"Cannot choose dihedral ref for atom {i_orig+1}.")
                    l = cand[0]

            r_tok = spec.r_name if spec.r_name else float(distance(coords[i_orig], coords[j]))
            a_tok = spec.a_name if spec.a_name else float(angle(coords[i_orig], coords[j], k))
            d_tok = spec.d_name if spec.d_name else float(dihedral(coords[i_orig], coords[j], k, l))

            lines.append(
                f"{elems[i_orig]}  {pos_of[j]+1}  {r_tok}  {pos_of[k]+1}  {a_tok}  {pos_of[l]+1}  {d_tok}"
            )
            emitted.append(i_orig)
            continue

        j, k, l = _choose_fallback_refs(i_orig, emitted)
        r_val = distance(coords[i_orig], coords[j])
        a_val = angle(coords[i_orig], coords[j], coords[k])
        d_val = dihedral(coords[i_orig], coords[j], coords[k], coords[l])
        lines.append(
            f"{elems[i_orig]}  {pos_of[j]+1}  {r_val:.6f}  {pos_of[k]+1}  {a_val:.6f}  {pos_of[l]+1}  {d_val:.6f}"
        )
        emitted.append(i_orig)

    lines.append("")
    for k, v in varmap.items():
        lines.append(f"{k}={v:.12f}")

    return lines, varmap

# ------------------------------------------------------------
def xyz2zmat(
    xyz_path: str,
    *,
    bonds: Optional[List[Tuple[int, int]]] = None,
    angles: Optional[List[Tuple[int, int, int]]] = None,
    dihedrals: Optional[List[Tuple[int, int, int, int]]] = None,
    bond_names: Optional[List[str]] = None,
    angle_names: Optional[List[str]] = None,
    dihedral_names: Optional[List[str]] = None,
    scale: float = 1.25,
    header: bool = True,
    out: Optional[str] = None,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Convert an XYZ file to a (Gaussian-like) Z-matrix, optionally forcing
    *multiple* internal coordinates (bonds/angles/dihedrals) to appear
    explicitly as named variables.

    This is the recommended high-level entry point for building a Z-matrix
    that you can later edit (e.g., change a torsion) and convert back to XYZ.

    Parameters
    ----------
    xyz_path
        Path to input XYZ file (standard 2-line header XYZ).
    bonds
        Optional list of (A, B) pairs (0-based indices in the ORIGINAL XYZ).
        Each pair enforces the distance(A-B) as a *variable* on the atom B line,
        i.e., atom B is defined with bond reference A:
            B  A  R_A_B  ...
        This means you control R(A,B) by editing that variable in the Z-matrix.

        Example:
            bonds=[(0,1), (1,2)]  enforces R(1-2) and R(2-3) (in 1-based labels).

    angles
        Optional list of (A, B, C) triples (0-based indices in the ORIGINAL XYZ)
        enforcing angle(A-B-C) as a *variable* on the atom C line, with references
        (bond_ref=B, angle_ref=A). This works because:
            angle(C, B, A) == angle(A, B, C)

        Z-matrix line pattern for atom C:
            C  B  R  A  A_A_B_C  ...

    dihedrals
        Optional list of (A, B, C, D) quadruples (0-based indices in the ORIGINAL XYZ)
        enforcing dihedral(A-B-C-D) as a *variable* on the atom D line, with references
        (bond_ref=C, angle_ref=B, dihed_ref=A). This works because:
            dihedral(D, C, B, A) == dihedral(A, B, C, D)

        Z-matrix line pattern for atom D:
            D  C  R  B  A  A  PHI_A_B_C_D

        This is the key for your use-case:
            dihedrals=[(C2,C1,C8,C9)]  => enforce torsion on atom C9 line.

    bond_names / angle_names / dihedral_names
        Optional lists of variable names (strings). If provided, each list must have
        the same length as its corresponding targets list.
        If omitted, default names are generated:
            bond:     R_<A+1>_<B+1>
            angle:    A_<A+1>_<B+1>_<C+1>
            dihedral: PHI_<A+1>_<B+1>_<C+1>_<D+1>
        Note: these indices in names are 1-based ORIGINAL XYZ indices.

    scale
        Scale factor for covalent-radius-based bond guessing, used only when choosing
        fallback references for *non-forced* atoms (and to pick reasonable refs when
        some refs are not specified). Typical values: 1.20–1.35.
        This does NOT affect forced targets themselves.
    header
        If True, prepend comment lines explaining format/conventions.
    out
        If provided, write the resulting Z-matrix text to this file path.
        If None, nothing is written (you still get the lines returned).

    Returns
    -------
    zmat_lines
        List of lines (strings) representing the Z-matrix file content.
        The end of the Z-matrix includes a variable-definition block:
            VAR=number
        for every forced target variable.
    varmap
        dict mapping variable name -> default numeric value taken from the input XYZ.

    Raises
    ------
    ValueError
        If any of the following occurs:
        - Index out of range or duplicated indices in a single target
        - Conflicting constraints on the same defined atom:
            * e.g., specifying bonds (A,B) and (C,B) would demand two different
              bond references for atom B
            * similarly for angle/dihedral references
        - Cyclic dependency in constraints (impossible Z-matrix dependency graph):
            * e.g., forcing both (A,B) and (B,A)
        - A constraint would require an atom to appear too early in the Z-matrix:
            * angle requires the constrained atom to appear at position >= 3
            * dihedral requires position >= 4
          (this is usually guaranteed by dependencies; if not, you get an error)

    Notes
    -----
    - The Z-matrix ordering is chosen to satisfy dependency constraints strictly.
      It is not guaranteed to be “chemically pretty” (though it is valid).
    - Forced targets are guaranteed to appear as variables in the output.
      That is the point of this wrapper.
    - For editing only torsions/bonds and converting back, this is typically sufficient.

    Examples
    --------
    1) Force two bond lengths and one torsion (stilbene-style):

    >>> zlines, varmap = xyz2zmat(
    ...     "stilbene_cis.xyz",
    ...     bonds=[(0,1), (1,2)],                 # R_1_2, R_2_3
    ...     dihedrals=[(1,0,7,8)],                # PHI_2_1_8_9
    ...     out="stilbene_cis_forced.zmat",
    ... )

    2) Provide custom variable names:

    >>> zlines, varmap = xyz2zmat(
    ...     "mol.xyz",
    ...     bonds=[(0,1)],
    ...     bond_names=["R12"],
    ...     dihedrals=[(1,0,7,8)],
    ...     dihedral_names=["PHI_TARGET"],
    ... )
    """
    elems, coords = read_xyz(xyz_path)

    if (bonds or angles or dihedrals):
        zlines, varmap = xyz_to_zmatrix_with_targets(
            elems, coords,
            bonds=bonds, angles=angles, dihedrals=dihedrals,
            bond_names=bond_names, angle_names=angle_names, dihedral_names=dihedral_names,
            scale=scale, header=header
        )
    else:
        zlines = xyz_to_zmatrix(elems, coords, scale=scale, header=header)
        varmap = {}

    if out is not None:
        with open(out, "w") as f:
            write_zmatrix(zlines, file=f)

    return zlines, varmap


def zmat2xyz(
    zmat_path: str,
    *,
    overrides: Optional[Dict[str, float]] = None,
    comment: str = "Generated by tlab.geom (zmat2xyz)",
    out: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a Z-matrix file (Gaussian-like subset, optionally with variables)
    back to an XYZ geometry.

    This is the recommended high-level entry point for:
      1) taking a Z-matrix produced by xyz2zmat(),
      2) overriding one or more internal-coordinate variables (e.g., a torsion),
      3) reconstructing Cartesian coordinates.

    Parameters
    ----------
    zmat_path
        Path to input Z-matrix file.
        The file may contain variable definitions (VAR=number) anywhere after the
        atom lines (blank lines/comments allowed).
    overrides
        Optional dict {varname: value} to override variables read from the file.
        Typical use: change a single torsion variable to match another method.

        Example:
            overrides={"PHI_2_1_8_9": -143.036717}

        If a variable appears in overrides but not in the file, it is added
        to the internal varmap for resolution; however, if the Z-matrix atom lines
        never reference that variable name, it has no effect.
    comment
        Comment line to write in XYZ (2nd line of XYZ).
    out
        If provided, write the resulting XYZ to this file path.
        If None, nothing is written (you still get arrays returned).

    Returns
    -------
    elems
        np.ndarray of element symbols (shape (N,), dtype=str).
    coords
        np.ndarray of Cartesian coordinates in Å (shape (N,3), dtype=float).

    Raises
    ------
    ValueError
        If:
        - The Z-matrix format is unsupported (wrong number of tokens per line)
        - A variable name is used in an atom line but is not defined in the file
          and not provided in overrides
        - Conversion encounters a pathological reference geometry (rare; e.g.,
          collinear references causing numerical degeneracy). The implementation
          includes fallbacks, but extreme cases may still error.

    Notes
    -----
    - This function reconstructs one valid Cartesian embedding of the internal
      coordinates. Z-matrices can admit multiple embeddings (mirror images)
      depending on conventions; the algorithm here follows a consistent convention.
    - If your workflow is:
          xyz2zmat -> edit PHI -> zmat2xyz
      then the resulting structure will change primarily around the targeted torsion,
      while all other internal coordinates are kept as specified by the Z-matrix
      (either numeric constants or other variables).

    Examples
    --------
    1) Change one torsion and write XYZ:

    >>> elems, xyz = zmat2xyz(
    ...     "stilbene_cis_forced.zmat",
    ...     overrides={"PHI_2_1_8_9": -143.036717},
    ...     out="stilbene_phi_to_ecis.xyz",
    ... )

    2) No overrides (use file’s own variables/constants), print to stdout:

    >>> elems, xyz = zmat2xyz("mol.zmat")
    >>> write_xyz(elems, xyz)
    """
    elems, refs, vals, varmap = read_zmatrix(zmat_path)
    if overrides:
        varmap.update(overrides)

    coords = zmatrix_to_xyz(elems, refs, vals, varmap=varmap)
    elems_arr = np.array(elems, dtype=str)

    if out is not None:
        with open(out, "w") as f:
            write_xyz(elems_arr, coords, file=f, comment=comment)

    return elems_arr, coords

# ------------------------------------------------------------
# CLI: python -m tlab.geom xyz2zmat ... / zmat2xyz ...
# ------------------------------------------------------------
def _parse_int_tuple(s: str) -> Tuple[int, ...]:
    # "1,2" or "1 2" in CLI => 1-based input -> convert to 0-based
    parts = re.split(r"[,\s]+", s.strip())
    parts = [p for p in parts if p]
    return tuple(int(p) - 1 for p in parts)

def _main(argv: List[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(description="tlab.geom: xyz2zmat / zmat2xyz")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("xyz2zmat", help="Convert XYZ to Z-matrix (optionally with forced targets)")
    p1.add_argument("xyz", help="input xyz")
    p1.add_argument("--out", default=None, help="output zmat path (default: stdout)")
    p1.add_argument("--scale", type=float, default=1.25, help="bond cutoff scale for bond guessing")
    p1.add_argument("--no-header", action="store_true", help="do not print header comments")

    p1.add_argument("--bond", action="append", default=[], help="force bond as 'i,j' (1-based indices)")
    p1.add_argument("--angle", action="append", default=[], help="force angle as 'i,j,k' (1-based)")
    p1.add_argument("--dihedral", action="append", default=[], help="force dihedral as 'i,j,k,l' (1-based)")

    p2 = sub.add_parser("zmat2xyz", help="Convert Z-matrix to XYZ (with optional overrides)")
    p2.add_argument("zmat", help="input zmat")
    p2.add_argument("--out", default=None, help="output xyz path (default: stdout)")
    p2.add_argument("--set", action="append", default=[], help="override variable like 'PHI_2_1_8_9=-143.0'")
    p2.add_argument("--comment", default="Generated by tlab.geom (zmat2xyz)", help="XYZ comment line")

    args = p.parse_args(argv)

    if args.cmd == "xyz2zmat":
        bonds = [tuple(_parse_int_tuple(x)) for x in args.bond]
        angles = [tuple(_parse_int_tuple(x)) for x in args.angle]
        dihedrals = [tuple(_parse_int_tuple(x)) for x in args.dihedral]

        # validate tuple lengths
        for t in bonds:
            if len(t) != 2:
                raise ValueError(f"--bond must have 2 indices, got {t}")
        for t in angles:
            if len(t) != 3:
                raise ValueError(f"--angle must have 3 indices, got {t}")
        for t in dihedrals:
            if len(t) != 4:
                raise ValueError(f"--dihedral must have 4 indices, got {t}")

        zlines, _ = xyz2zmat(
            args.xyz,
            bonds=bonds or None,
            angles=angles or None,
            dihedrals=dihedrals or None,
            scale=args.scale,
            header=(not args.no_header),
            out=args.out,
        )
        if args.out is None:
            write_zmatrix(zlines)
        return 0

    if args.cmd == "zmat2xyz":
        overrides: Dict[str, float] = {}
        for s in args.set:
            if "=" not in s:
                raise ValueError(f"--set must be like VAR=VALUE, got: {s}")
            k, v = s.split("=", 1)
            overrides[k.strip()] = float(v)

        elems, coords = zmat2xyz(
            args.zmat,
            overrides=overrides or None,
            comment=args.comment,
            out=args.out,
        )
        if args.out is None:
            write_xyz(elems, coords, comment=args.comment)
        return 0

    return 1

if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))

