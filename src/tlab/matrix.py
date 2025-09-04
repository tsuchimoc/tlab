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

def _parse_matrix_from(lines: list[str], start_idx: int) -> Tuple[np.ndarray, int]:
    def _looks_like_section_end(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        if s.startswith("***"):
            return True
        return False

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

    def _is_int_columns(line: str) -> Optional[list[int]]:
        m = _INT_COL_RE.match(line)
        if not m:
            return None
        try:
            cols = [int(t) for t in m.group(1).split()]
        except ValueError:
            return None
        return cols if cols else None

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

import sys
def printmat(A, title=None, eig=None, mmax=5, n=None, m=None, format="12.7f", ao_labels=None,  file=None):
    """Function:
    Print out A in a readable format.

        A         :  1D or 2D numpy array of dimension
        eig       :  Given eigenvectros A[:,i], eig[i] are corresponding eigenvalues (ndarray or list)
        file      :  file to be printed
        mmax      :  maxixmum number of columns to print for each block
        title     :  Name to be printed
        n,m       :  Need to be specified if A is a matrix,
                     but loaded as a 1D array
        format    :  Printing format
        ao_labels :  AO labels instead of integers for rows.

    Author(s): Takashi Tsuchimochi
    """
    if isinstance(A, list):
        dimension = 1
    elif isinstance(A, np.ndarray):
        dimension = A.ndim
    if dimension == 0 or dimension > 2:
        error("Neither scalar nor tensor is printable with printmat.")
    
    if file is None:
        file = sys.stdout
    if True:
        if title is not None:
            file.write(f" {title}\n\n")

        if format.find('f') != -1:
            ### Float
            style = "f"
            idx_f = format.find('f') 
            idx = format.find('.')
            digits = int(format[:idx]) 
            decimal = int(format[idx+1:idx_f])
        elif format.find('d') != -1:
            style = "d"
            idx = format.find('d')
            digits = int(format[:idx])
            decimal = '0'
            format = format[:idx]+'.0f'
        else:
            raise TypeError(f'format={format} not supported in printmat.')
        len_ = digits
        if format[0] != ' ':
            format = ' ' + format
        if format.find('>') == -1:
            format = '>' + format
        if dimension == 2:
            n, m = A.shape
            imax = 0
            while imax < m:
                imin = imax + 1
                imax = imax + mmax
                if imax > m:
                    imax = m
                file.write('\n')
                if ao_labels is not None:
                    space_ao = " "*7
                else:
                    space_ao = ""
                file.write(f"{space_ao}")#file=f)
                if eig is None:
                    space = " "*(5)
                    file.write(f"{space}")#file=f)
                else:
                    file.write(" eig |")#file=f)
                    
                for i in range(imin-1, imax):
                    if eig is None:
                        space = " "*(len_)
                        file.write(f"{i:>{len_}d} ")#file=f)
                    else:
                        file.write(f"{eig[i]:{format}}|")
                file.write('\n')
                file.write(f"{space_ao}")
                if eig is not None:
                    hyphen = '-'*len_
                    for i in range(imin-1, imax+1):
                        file.write(f"{hyphen}")
                else:
                    file.write(f"{space}")
                print()#file=f)
                for j in range(n):
                    if ao_labels is None:
                        file.write(f" {j:4d} ")
                    else:
                        file.write(f" {ao_labels[j]:12s}")
                    for i in range(imin-1, imax):
                        file.write(f"{A[j][i]:{format}} ")
                    file.write('\n')
        elif dimension == 1:
            if n is None or m is None:
                if isinstance(A, list):
                    n = len(A)
                    m = 1
                elif isinstance(A, np.ndarray):
                    n = A.size
                    m = 1
            imax = 0
            while imax < m:
                imin = imax + 1
                imax = imax + mmax
                if imax > m:
                    imax = m
                if eig is None:
                    file.write("           ")
                else:
                    file.write(" eig:  ")
                    
                for i in range(imin-1, imax):
                    if eig is None:
                        file.write(f"{i:{digits-6}d}          ")
                    else:
                        file.write(f"  {eig[i]:{format}}  ")
                file.write('\n')
                for j in range(n):
                    if n > 1:
                        file.write(f" {j:4d}  ")
                    else:
                        file.write(f"       ")
                    for i in range(imin-1, imax):
                        file.write(f"  {A[j + i*n]:{format}}  ")
                    file.write('\n')
        file.write('\n')
        file.flush()
