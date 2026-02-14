import numpy as np
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
Geometry utilities for XYZ <-> Z-matrix conversion, plus small helpers.

Features
--------
- Read/write XYZ (standard 2-line header format)
- Read/write simple Z-matrix format:
    Elem
    Elem  i  R
    Elem  i  R  j  Angle
    Elem  i  R  j  Angle  k  Dihedral
  where i,j,k are 1-based indices in the Z-matrix order.
- XYZ -> Z-matrix:
    - heuristic bond-guessing via covalent radii and distance cutoff
    - BFS order generation to follow connectivity
- Z-matrix -> XYZ:
    - deterministic construction of Cartesian coords from internal coordinates

Public API
----------
- read_xyz(path) -> (elems, coords)
- write_xyz(elems, coords, file=None, comment="...")
- build_bond_graph(elems, coords, scale=1.25) -> adjacency list
- xyz_to_zmatrix(elems, coords, scale=1.25) -> list[str] (zmat lines)
- write_zmatrix(zmat_lines, file=None, header=True)

- read_zmatrix(path) -> (elems, refs, values)
- zmatrix_to_xyz(elems, refs, values) -> coords (N,3)
- zmat_lines_to_internal(zmat_lines) -> (elems, refs, values)

Notes
-----
- Covalent radii are used ONLY for bond-guessing in xyz_to_zmatrix().
- For Z-matrix -> XYZ conversion, covalent radii are NOT needed.
"""

from __future__ import annotations

import sys
import math
import numpy as np
from collections import deque
from typing import List, Tuple, Optional, TextIO, Dict, Any


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
    # 6th row (incl. 5d + lanthanides representative)
    "Cs": 2.44, "Ba": 2.15,
    "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01, "Pm": 1.99, "Sm": 1.98, "Eu": 1.98, "Gd": 1.96,
    "Tb": 1.94, "Dy": 1.92, "Ho": 1.92, "Er": 1.89, "Tm": 1.90, "Yb": 1.87, "Lu": 1.87,
    "Hf": 1.78, "Ta": 1.70, "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41, "Pt": 1.36,
    "Au": 1.36, "Hg": 1.32,
    "Tl": 1.45, "Pb": 1.46, "Bi": 1.48, "Po": 1.40, "At": 1.50, "Rn": 1.50,
}


def cov_rad(sym: str) -> float:
    """Return covalent radius (Å) for bond guessing; fallback to DEFAULT_RAD."""
    return COV_RAD.get(sym, DEFAULT_RAD)


# ------------------------------------------------------------
# XYZ I/O
# ------------------------------------------------------------
def read_xyz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read standard XYZ file. Returns (elems[str], coords[float] shape (N,3))."""
    with open(path, "r") as f:
        n = int(f.readline().strip())
        _ = f.readline()
        elems: List[str] = []
        coords: List[List[float]] = []
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
    """Write XYZ to file (stdout if None)."""
    if file is None:
        file = sys.stdout
    n = len(elems)
    file.write(f"{n}\n")
    file.write(f"{comment}\n")
    for e, c in zip(elems, coords):
        file.write(f"{e:2s}  {c[0]: .10f}  {c[1]: .10f}  {c[2]: .10f}\n")


# ------------------------------------------------------------
# Geometry primitives
# ------------------------------------------------------------
def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle a-b-c in degrees."""
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-15:
        return 0.0
    cosang = float(np.dot(ba, bc) / denom)
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))


def dihedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Dihedral a-b-c-d in degrees, range (-180, 180]."""
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
# Bond guessing + ordering
# ------------------------------------------------------------
def build_bond_graph(elems: np.ndarray, coords: np.ndarray, scale: float = 1.25) -> List[List[int]]:
    """
    Build adjacency list by distance <= scale*(ri+rj).
    Heuristic only; meant for xyz_to_zmatrix().
    """
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


def generate_bfs_order(G: List[List[int]]) -> List[int]:
    """
    Generate an atom order by BFS, restarting for disconnected components.
    Starts from atom 0, then smallest unvisited index.
    """
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
    return order


# ------------------------------------------------------------
# XYZ -> Z-matrix
# ------------------------------------------------------------
def xyz_to_zmatrix(
    elems: np.ndarray,
    coords: np.ndarray,
    scale: float = 1.25,
    header: bool = False,
) -> List[str]:
    """
    Convert XYZ arrays to a simple Z-matrix lines list.
    The references are chosen as the previous 1/2/3 atoms in the BFS order.
    """
    G = build_bond_graph(elems, coords, scale=scale)
    order = generate_bfs_order(G)

    lines: List[str] = []
    if header:
        lines.append("# Z-matrix generated from XYZ (heuristic bonding via covalent radii)")
        lines.append("# Format: Elem  ref1  R  ref2  Angle  ref3  Dihedral")
        lines.append("")

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


def write_zmatrix(zmat_lines: List[str], file: Optional[TextIO] = None, header: bool = False) -> None:
    """Write Z-matrix lines to file (stdout if None)."""
    if file is None:
        file = sys.stdout
    if header:
        file.write("# Z-matrix\n")
        file.write("# Format: Elem  ref1  R  ref2  Angle  ref3  Dihedral\n\n")
    for ln in zmat_lines:
        file.write(ln.rstrip() + "\n")


# ------------------------------------------------------------
# Z-matrix parsing
# ------------------------------------------------------------
def zmat_lines_to_internal(zmat_lines: List[str]) -> Tuple[List[str], List[List[int]], List[List[float]]]:
    """
    Parse Z-matrix lines into (elems, refs, values).
    refs are 0-based indices into already-defined atoms in Z-matrix order.
    values are floats [R], [R,Angle], or [R,Angle,Dihedral].
    """
    elems: List[str] = []
    refs: List[List[int]] = []
    values: List[List[float]] = []

    for raw in zmat_lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) not in (1, 3, 5, 7):
            raise ValueError(f"Unsupported Z-matrix line:\n{raw}")

        elems.append(parts[0])

        if len(parts) == 1:
            refs.append([])
            values.append([])
        elif len(parts) == 3:
            refs.append([int(parts[1]) - 1])
            values.append([float(parts[2])])
        elif len(parts) == 5:
            refs.append([int(parts[1]) - 1, int(parts[3]) - 1])
            values.append([float(parts[2]), float(parts[4])])
        else:  # 7
            refs.append([int(parts[1]) - 1, int(parts[3]) - 1, int(parts[5]) - 1])
            values.append([float(parts[2]), float(parts[4]), float(parts[6])])

    return elems, refs, values


def read_zmatrix(path: str) -> Tuple[List[str], List[List[int]], List[List[float]]]:
    """Read Z-matrix file into (elems, refs, values)."""
    with open(path, "r") as f:
        lines = f.readlines()
    return zmat_lines_to_internal(lines)


# ------------------------------------------------------------
# Z-matrix -> XYZ
# ------------------------------------------------------------
def rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix."""
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-15:
        return np.eye(3)
    axis = axis / n
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=float,
    )


def zmatrix_to_xyz(
    elems: List[str],
    refs: List[List[int]],
    values: List[List[float]],
) -> np.ndarray:
    """
    Build Cartesian coords from Z-matrix definition.
    Convention:
      atom0 at origin
      atom1 on +z
      atom2 in xz-plane
      atoms>=3: general placement using internal coords
    """
    n = len(elems)
    coords: List[np.ndarray] = []

    for i in range(n):
        if i == 0:
            coords.append(np.array([0.0, 0.0, 0.0], dtype=float))
        elif i == 1:
            r = values[i][0]
            coords.append(np.array([0.0, 0.0, r], dtype=float))
        elif i == 2:
            r, ang_deg = values[i]
            ang = math.radians(ang_deg)

            a = coords[refs[i][0]]  # bond reference
            b = coords[refs[i][1]]  # angle reference (for i==2 usually atom0)

            # Place in xz-plane relative to atom a; keep y=0
            # We use a simple construction anchored to a and b.
            # Direction from a to b defines -z-like axis in this local frame.
            ab = b - a
            nab = np.linalg.norm(ab)
            if nab < 1e-15:
                # fallback: just place with respect to global frame
                coords.append(np.array([r * math.sin(ang), 0.0, a[2] - r * math.cos(ang)], dtype=float))
            else:
                e1 = ab / nab  # axis along a->b
                # choose an arbitrary perpendicular for x axis
                tmp = np.array([1.0, 0.0, 0.0], dtype=float)
                if abs(np.dot(tmp, e1)) > 0.9:
                    tmp = np.array([0.0, 1.0, 0.0], dtype=float)
                ex = np.cross(e1, tmp)
                ex /= np.linalg.norm(ex)
                ez = e1  # local z
                # want point at distance r making angle ang with (a-b) direction, i.e. with -ez
                # vector = r*(cos(ang)*(-ez) + sin(ang)*ex)
                v = r * (math.cos(ang) * (-ez) + math.sin(ang) * ex)
                coords.append(a + v)
        else:
            r, ang_deg, dih_deg = values[i]
            ang = math.radians(ang_deg)
            dih = math.radians(dih_deg)

            ia, ib, ic = refs[i]  # a: bond ref, b: angle ref, c: dihedral ref
            a = coords[ia]
            b = coords[ib]
            c = coords[ic]

            # Build local orthonormal basis at a using (a-b) and plane defined by (b,c)
            ba = a - b
            n_ba = np.linalg.norm(ba)
            if n_ba < 1e-15:
                # pathological; place arbitrarily
                coords.append(a + np.array([0.0, 0.0, r], dtype=float))
                continue
            e1 = ba / n_ba  # points from b to a

            bc = c - b
            e2 = np.cross(bc, e1)
            n_e2 = np.linalg.norm(e2)
            if n_e2 < 1e-15:
                # if collinear, choose any perpendicular
                tmp = np.array([1.0, 0.0, 0.0], dtype=float)
                if abs(np.dot(tmp, e1)) > 0.9:
                    tmp = np.array([0.0, 1.0, 0.0], dtype=float)
                e2 = np.cross(tmp, e1)
                e2 /= np.linalg.norm(e2)
            else:
                e2 /= n_e2

            e3 = np.cross(e1, e2)

            # Displacement from a:
            # r * [cos(ang)*(-e1) + sin(ang)*(cos(dih)*e3 + sin(dih)*e2)]
            dvec = r * (
                math.cos(ang) * (-e1)
                + math.sin(ang) * (math.cos(dih) * e3 + math.sin(dih) * e2)
            )
            coords.append(a + dvec)

    return np.vstack(coords)


# ------------------------------------------------------------
# CLI helper (optional)
# ------------------------------------------------------------
def _main(argv: List[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(description="tlab.geom: XYZ <-> Z-matrix converter")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("xyz2zmat", help="Convert XYZ to Z-matrix")
    p1.add_argument("xyz", help="input xyz")
    p1.add_argument("--scale", type=float, default=1.25, help="bond cutoff scale for covalent radii")
    p1.add_argument("--header", action="store_true", help="print header comments")

    p2 = sub.add_parser("zmat2xyz", help="Convert Z-matrix to XYZ")
    p2.add_argument("zmat", help="input zmat")
    p2.add_argument("--comment", default="Generated by tlab.geom", help="XYZ comment line")

    args = p.parse_args(argv)

    if args.cmd == "xyz2zmat":
        elems, coords = read_xyz(args.xyz)
        zlines = xyz_to_zmatrix(elems, coords, scale=args.scale, header=args.header)
        write_zmatrix(zlines, header=False)
        return 0

    if args.cmd == "zmat2xyz":
        elems, refs, values = read_zmatrix(args.zmat)
        coords = zmatrix_to_xyz(elems, refs, values)
        write_xyz(np.array(elems, dtype=str), coords, comment=args.comment)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))

