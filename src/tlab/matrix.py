from __future__ import annotations
import re
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

_INT_COL_RE = re.compile(r"^\s*(\d+(?:\s+\d+)+)\s*$")
_FLOAT_RE = re.compile(r"""
    [+\-]?                      # sign
    (?:\d+(?:\.\d*)?|\.\d+)     # body
    (?:[EDed][+\-]?\d+)?        # exponent with E or D
""", re.VERBOSE)

def _is_int_columns(line: str) -> Optional[list[int]]:
    m = _INT_COL_RE.match(line)
    if not m:
        return None
    try:
        cols = [int(t) for t in m.group(1).split()]
    except ValueError:
        return None
    return cols if cols else None

def _try_parse_row(line: str, ncols: int) -> Optional[Tuple[int, list[float]]]:
    toks = line.strip().split()
    if not toks:
        return None
    if re.fullmatch(r"\d+", toks[0]):
        row = int(toks[0])
        vals_str = toks[1:]
        if len(vals_str) >= ncols:
            vals = [float(v.replace("D","E").replace("d","e")) for v in vals_str[:ncols]]
            return row, vals
        floats = _FLOAT_RE.findall(line)
        if len(floats) >= ncols:
            vals = [float(s.replace("D","E").replace("d","e")) for s in floats[-ncols:]]
            return row, vals
    return None

def _looks_like_section_end(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if "Density matrix done" in s:
        return True
    if s.startswith("***"):
        return True
    return False

def _parse_matrix_from(lines: list[str], start_idx: int) -> Tuple[np.ndarray, int]:
    k = start_idx
    while k < len(lines) and _is_int_columns(lines[k]) is None:
        if _looks_like_section_end(lines[k]):
            raise RuntimeError("Matrix header ended before integer column header.")
        k += 1
    if k >= len(lines):
        raise RuntimeError("Integer column header not found.")

    data: dict[tuple[int,int], float] = {}
    max_row = 0
    max_col = 0
    col_idx: Optional[list[int]] = None
    captured = False

    while k < len(lines):
        line = lines[k]
        cols = _is_int_columns(line)
        if cols is not None:
            col_idx = cols
            max_col = max(max_col, max(cols))
            k += 1
            continue

        if col_idx is not None:
            parsed = _try_parse_row(line, ncols=len(col_idx))
            if parsed is not None:
                row, vals = parsed
                max_row = max(max_row, row)
                for j, c in enumerate(col_idx):
                    data[(row, c)] = vals[j]
                captured = True
                k += 1
                continue

        if not line.strip():
            k += 1
            continue

        if _looks_like_section_end(line):
            k += 1
            break

        if re.match(r"^\s*[A-Za-z].*$", line) and captured:
            break

        if captured:
            break
        k += 1

    if not data:
        raise RuntimeError("No matrix data captured.")

    N = max(max_row, max_col)
    M = np.zeros((N, N), dtype=float)
    for (r, c), v in data.items():
        if 1 <= r <= N and 1 <= c <= N:
            M[r-1, c-1] = v
    return M, k

def getmat(out_path: str, string: str) -> List[np.ndarray]:
    """
    Get matrix (matrices) named `string` from `out_path`. 
    `out_path`のファイル内を検索し、`string`の文字列から始まる行列を抜き出してnumpyの配列に入れます。
    """
    lines = Path(out_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    mats: list[np.ndarray] = []
    i = 0
    while i < len(lines):
        found = None
        for j in range(i, len(lines)):
            if string in lines[j]:
                found = j
                break
        if found is None:
            break
        try:
            M, next_idx = _parse_matrix_from(lines, found + 1)
            mats.append(M)
            i = next_idx
        except RuntimeError:
            i = found + 1
    return mats

def printmat(M, title="", col=10, precision=7, width=12, file=None, flush=False):
    """
    Return a string that prints matrix `M` in a block format:
      - Column indices shown in blocks (default 10 per block)
      - Row index at the left
      - Fixed-width floats with given precision
    numpyの２次元配列で与えられた行列`M`を見やすく表示します。
    以下オプション：
    `title`で名前をつけます。
    `col`は1ブロックあたりのカラムの数でデフォルトは10です。
    `precision`は桁数です。
    `width`はどれくらい幅を取るかです。
    `file`を設定するとそのファイルに書き出します。
    `flush`をするとバッファがたまる前にすぐに書き出します。
    """
    A = np.asarray(M, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("M must be a square 2D array")
    N = A.shape[0]

    if file is None:
        file = sys.stdout

    roww = max(5, len(str(N)) + 3)  # width for the left row index

    # Title
    file.write(f" {title}\n\n")

    # Blocks of columns
    for start in range(0, N, col):
        end = min(N, start + col)

        # Column header
        file.write(" " * roww)
        for j in range(start, end):
            file.write(f"{j+1:>{width}d}")
        file.write("\n")

        # Rows
        for i in range(N):
            file.write(f"{i+1:>{roww}d}")
            for j in range(start, end):
                file.write(f"{A[i, j]: {width}.{precision}f}")
            file.write("\n")
        file.write("\n")

    if flush:
        file.flush()
    return None
