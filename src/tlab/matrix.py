from __future__ import annotations
import sys
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np

_INT_COL_RE = re.compile(r"^\s*(\d+(?:\s+\d+)*)\s*$")
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

    # --- ここを正方行列→長方形（非正方）対応に変更 ---
    nrows = max_row
    ncols = max_col
    M = np.zeros((nrows, ncols), dtype=float)
    for (r, c), v in data.items():
        if 1 <= r <= nrows and 1 <= c <= ncols:
            M[r-1, c-1] = v
    return M, k

def getmat(out_path: str, string: str, all: bool = False
           ) -> Union[List[np.ndarray], Optional[np.ndarray]]:
    """
    Get matrix (matrices) named `string` from `out_path`.

    - 列数 ≠ 行数の長方形行列にも対応します（出力は (max_row, max_col)）。

    Parameters
    ----------
    out_path : str
        出力ファイルのパス
    string : str
        見出し（この文字列を含む行の直後から行列ブロックを探す）
    all : bool, default True
        True のときは見つかったすべての行列を List[np.ndarray] で返す。
        False のときは最初の行列のみ np.ndarray を返す（見つからなければ None）。

    Returns
    -------
    List[np.ndarray] | np.ndarray | None
    """
    lines = Path(out_path).read_text(encoding="utf-8", errors="ignore").splitlines()

    mats: List[np.ndarray] = []
    i = 0
    while i < len(lines):
        found = None
        for j in range(i, len(lines)):
            if string in lines[j]:
                found = j
                break

        if found is None:
            break  # これ以上見出しが無い

        try:
            M, next_idx = _parse_matrix_from(lines, found + 1)
        except RuntimeError:
            # この見出しの直後に行列ブロックが無い/壊れている → 次行から再探索
            i = found + 1
            continue

        if all:
            mats.append(M)
            i = next_idx  # 次のブロック以降を探索
        else:
            print(f'Matrix "{string}" found. shape={M.shape}')
            return M  # 最初の1件だけ返して終了

    if mats == [] or not all:
        print(f'No matrix "{string}" found.')
    else:
        print(f'{len(mats)} "{string}" matrices found. shapes={[m.shape for m in mats]}')
    return mats if all else None
def printmat(M, title: str = "", col: int = 10, precision: int = 7,
             width: int = 12, file=None, flush: bool = False):
    """
    Return a string that prints matrix `M` in a block format:
      - Column indices shown in blocks (default 10 per block)
      - Row index at the left
      - Fixed-width floats with given precision

    numpy の 2 次元配列で与えられた行列 `M` を見やすく表示します（非正方可）。
    オプションは従来どおり。

    Parameters
    ----------
    M : array-like, shape (nrows, ncols)
        行列（非正方も可）
    title : str
        タイトル行
    col : int
        1 ブロックあたりの列数
    precision : int
        小数点以下桁数
    width : int
        各数値フィールド幅
    file : IO or None
        出力先（未指定なら sys.stdout）
    flush : bool
        出力後に flush する
    """
    A = np.asarray(M, dtype=float)
    if A.ndim != 2:
        raise ValueError("M must be a 2D array")
    nrows, ncols = A.shape

    if file is None:
        file = sys.stdout

    # 左端の行番号の幅（行数に応じて可変）
    roww = max(5, len(str(nrows)) + 3)

    # Title
    file.write(f" {title}\n\n")

    # 列ブロックごとに出力
    for start in range(0, ncols, col):
        end = min(ncols, start + col)

        # Column header
        file.write(" " * roww)
        for j in range(start, end):
            file.write(f"{j+1:>{width}d}")
        file.write("\n")

        # Rows
        for i in range(nrows):
            file.write(f"{i+1:>{roww}d}")
            for v in A[i, start:end]:
                file.write(f"{v: {width}.{precision}f}")
            file.write("\n")
        file.write("\n")

    if flush:
        file.flush()
    return None

