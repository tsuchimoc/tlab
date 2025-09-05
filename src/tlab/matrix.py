from __future__ import annotations
import sys
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union, NamedTuple
import numpy as np

_INT_COL_RE = re.compile(r"^\s*(\d+(?:\s+\d+)*)\s*$")
_FLOAT_RE = re.compile(r"""
    [+\-]?                      # sign
    (?:\d+(?:\.\d*)?|\.\d+)     # body
    (?:[EDed][+\-]?\d+)?        # exponent with E or D
""", re.VERBOSE)

class MatBlock(NamedTuple):
    M: np.ndarray
    eig: Optional[List[float]]
    ao_labels: Optional[List[str]]

def _parse_matrix_from(lines: list[str], start_idx: int):
    """
    返り値: (M: np.ndarray, next_idx: int, eigs: Optional[List[float]], ao_labels: Optional[List[str]])
    """
    def _looks_like_section_end(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        if s.startswith("***"):
            return True
        return False

    _INT_COL_RE = re.compile(r"^\s*(\d+(?:\s+\d+)*)\s*$")
    _FLOAT_RE = re.compile(r"""
        [+\-]?                      # sign
        (?:\d+(?:\.\d*)?|\.\d+)     # body
        (?:[EDed][+\-]?\d+)?        # exponent with E or D
    """, re.VERBOSE)
    _EIG_HDR_RE = re.compile(r"\beig\b", re.IGNORECASE)

    def _is_int_columns(line: str) -> Optional[list[int]]:
        m = _INT_COL_RE.match(line)
        if not m:
            return None
        try:
            cols = [int(t) for t in m.group(1).split()]
        except ValueError:
            return None
        return cols if cols else None

    def _parse_eig_values_from_line(line: str) -> Optional[List[float]]:
        if not _EIG_HDR_RE.search(line):
            return None
        floats = _FLOAT_RE.findall(line)
        if not floats:
            return None
        return [float(s.replace("D","E").replace("d","e")) for s in floats]

    def _try_parse_numeric_row(line: str, ncols: int) -> Optional[Tuple[int, List[float]]]:
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

    def _try_parse_labeled_row(line: str, ncols: int) -> Optional[Tuple[str, List[float]]]:
        m = re.match(r"^\s*(\S+)\s+(.*)$", line)
        if not m:
            return None
        label, rest = m.group(1), m.group(2)
        # 先頭トークンが純数字ならラベルではない
        if re.fullmatch(r"\d+", label):
            return None
        floats = _FLOAT_RE.findall(rest)
        if len(floats) >= ncols:
            vals = [float(s.replace("D","E").replace("d","e")) for s in floats[:ncols]]
            return label, vals
        return None

    # 見出し直後から「整数列ヘッダ」または「eig ヘッダ」を探す
    k = start_idx
    header_mode = None  # "intcols" or "eig"
    pending_cols = None
    pending_eigs = None

    while k < len(lines):
        if _looks_like_section_end(lines[k]):
            raise RuntimeError("Matrix header ended before a recognizable header.")
        cols = _is_int_columns(lines[k])
        if cols is not None:
            header_mode = "intcols"
            pending_cols = cols
            k += 1
            break
        eigs_block = _parse_eig_values_from_line(lines[k])
        if eigs_block:
            header_mode = "eig"
            pending_eigs = eigs_block
            k += 1
            # 区切り線があれば消費
            if k < len(lines) and re.match(r"^\s*[-=]{3,}\s*$", lines[k]):
                k += 1
            break
        k += 1

    if header_mode is None:
        raise RuntimeError("No matrix header found (neither integer columns nor eig header).")

    # ---------------- A) 整数列ヘッダモード ----------------
    if header_mode == "intcols":
        data_num: dict[tuple[int, int], float] = {}
        row_min = None
        row_max = None
        col_min = None
        col_max = None
        captured = False

        # ラベル対応（必要になったらONにする）
        ao_labels: Optional[List[str]] = None
        label_to_row: dict[str, int] = {}
        expected_row_idx: Optional[int] = None
        label_mode: Optional[bool] = None  # 最初のデータ行で True(ラベル) / False(数値) に固定

        current_cols = pending_cols

        while k < len(lines):
            line = lines[k]

            # 次ブロックの列ヘッダ？
            new_cols = _is_int_columns(line)
            if new_cols is not None:
                # ブロック切替時の検証（ラベルモード時は行数が揃っているか）
                if label_mode is True and ao_labels is not None and expected_row_idx is not None:
                    if expected_row_idx != len(ao_labels):
                        raise RuntimeError("AO label list mismatch across blocks (row count differs).")
                current_cols = new_cols
                # 列範囲更新
                col_min = min(new_cols) if col_min is None else min(col_min, min(new_cols))
                col_max = max(new_cols) if col_max is None else max(col_max, max(new_cols))
                # 新ブロック開始
                expected_row_idx = 0 if label_mode else None
                k += 1
                continue

            if current_cols is not None:
                # まず“現在のモード”に従ってパース（未確定なら数値優先→失敗ならラベル）
                parsed_num = None
                parsed_lab = None

                if label_mode is not True:
                    parsed_num = _try_parse_numeric_row(line, ncols=len(current_cols))
                    if parsed_num is not None and label_mode is None:
                        label_mode = False  # 最初のデータで固定
                        expected_row_idx = None

                if parsed_num is None and label_mode is not False:
                    parsed_lab = _try_parse_labeled_row(line, ncols=len(current_cols))
                    if parsed_lab is not None and label_mode is None:
                        label_mode = True
                        ao_labels = []
                        label_to_row = {}
                        expected_row_idx = 0

                # 数値行
                if parsed_num is not None:
                    row, vals = parsed_num
                    row_min = row if row_min is None else min(row_min, row)
                    row_max = row if row_max is None else max(row_max, row)
                    if col_min is None:
                        col_min, col_max = min(current_cols), max(current_cols)
                    else:
                        col_min = min(col_min, min(current_cols))
                        col_max = max(col_max, max(current_cols))
                    for j, c in enumerate(current_cols):
                        data_num[(row, c)] = vals[j]
                    captured = True
                    k += 1
                    continue

                # ラベル行
                if parsed_lab is not None:
                    label, vals = parsed_lab
                    assert label_mode is True
                    assert ao_labels is not None

                    if expected_row_idx is None:
                        expected_row_idx = 0
                    if label not in label_to_row:
                        if expected_row_idx != len(ao_labels):
                            # 途中で未知ラベルが出現 → 不一致
                            raise RuntimeError("AO label list mismatch across blocks (unknown label).")
                        label_to_row[label] = len(ao_labels)
                        ao_labels.append(label)

                    # 順序チェック
                    if expected_row_idx >= len(ao_labels) or label != ao_labels[expected_row_idx]:
                        raise RuntimeError("AO label list mismatch across blocks (order differs).")

                    rowid = label_to_row[label]
                    for j, c in enumerate(current_cols):
                        data_num[(rowid, c)] = vals[j]

                    expected_row_idx += 1
                    captured = True
                    k += 1
                    continue

            # 空行はスキップ
            if not line.strip():
                k += 1
                continue

            # セクション終端
            if _looks_like_section_end(line):
                k += 1
                break

            # ここに来た非空行はデータでない
            # 既に何か取れていればブロック終了
            if captured:
                break
            # まだ何も取れていない場合は次の行へ（ノイズ行を飛ばす）
            k += 1

        # ブロック末尾の検証
        if label_mode is True and ao_labels is not None and expected_row_idx is not None:
            if expected_row_idx != len(ao_labels):
                raise RuntimeError("AO label list mismatch across blocks (row count differs at end).")

        if not data_num:
            raise RuntimeError("No matrix data captured.")

        # 出力配列を構築
        if label_mode is True and ao_labels is not None:
            nrows = len(ao_labels)
            assert col_min is not None and col_max is not None
            ncols = col_max - col_min + 1
            M = np.zeros((nrows, ncols), dtype=float)
            for (r, c), v in data_num.items():
                # ラベル行の r は 0 起点の内部インデックス
                M[r, c - col_min] = v
            return M, k, None, ao_labels

        # 数値行モード
        assert row_min is not None and row_max is not None
        assert col_min is not None and col_max is not None
        nrows = row_max - row_min + 1
        ncols = col_max - col_min + 1
        M = np.zeros((nrows, ncols), dtype=float)
        for (r, c), v in data_num.items():
            M[r - row_min, c - col_min] = v
        return M, k, None, None

    # ---------------- B) eig ヘッダモード（横連結） ----------------
    eigs_all: List[float] = []
    ao_labels: Optional[List[str]] = None
    label_to_row: dict[str, int] = {}
    data_eig: dict[tuple[int, int], float] = {}
    col_offset = 0

    current_eigs = pending_eigs
    while current_eigs:
        ne = len(current_eigs)
        eigs_all.extend(current_eigs)
        expected_row_idx = 0

        while k < len(lines):
            line = lines[k]

            if not line.strip():
                k += 1
                continue

            # 次の eig ヘッダ？
            next_eigs = _parse_eig_values_from_line(line)
            if next_eigs:
                if ao_labels is None or expected_row_idx != len(ao_labels):
                    raise RuntimeError("AO label list mismatch or missing rows before next eig header.")
                k += 1
                if k < len(lines) and re.match(r"^\s*[-=]{3,}\s*$", lines[k]):
                    k += 1
                col_offset += ne
                current_eigs = next_eigs
                break

            if _looks_like_section_end(line):
                if ao_labels is None or expected_row_idx != len(ao_labels):
                    raise RuntimeError("AO label list mismatch or missing rows before section end.")
                current_eigs = None
                k += 1
                break

            parsed = _try_parse_labeled_row(line, ncols=ne)
            if parsed is None:
                raise RuntimeError("Unexpected line inside eigenvector block.")
            label, vals = parsed

            if ao_labels is None:
                ao_labels = []
                label_to_row = {}

            if label not in label_to_row:
                if len(ao_labels) != expected_row_idx:
                    raise RuntimeError("AO label list mismatch across eig blocks (unknown label).")
                label_to_row[label] = len(ao_labels)
                ao_labels.append(label)

            if expected_row_idx >= len(ao_labels) or label != ao_labels[expected_row_idx]:
                raise RuntimeError("AO label list mismatch across eig blocks (order differs).")

            rowid = label_to_row[label]
            for j, v in enumerate(vals):
                data_eig[(rowid, col_offset + j)] = v

            expected_row_idx += 1
            k += 1
        else:
            if ao_labels is None or expected_row_idx != len(ao_labels):
                raise RuntimeError("AO label list mismatch or missing rows at EOF.")
            current_eigs = None

    if not data_eig:
        raise RuntimeError("No eigenvector data captured after eig header.")

    nrows = len(ao_labels) if ao_labels is not None else 0
    ncols = len(eigs_all)
    if nrows == 0 or ncols == 0:
        raise RuntimeError("Eigenvector block malformed (no labels or no eigenvalues).")

    M = np.zeros((nrows, ncols), dtype=float)
    for (r, c), v in data_eig.items():
        M[r, c] = v

    return M, k, eigs_all, ao_labels


def getmat(out_path: str, string: str, all: bool = False
           ) -> Union[List[np.ndarray], Optional[np.ndarray]]:
    """
    Get matrices named `string` from `out_path`.
    `getgenmat`のシンプル版.

    - 行がでも文字列でも行列のみを返す。
    - 固有値・固有ベクトル（eigヘッダ）を検出した場合でも行列のみ返す。
    これらが必要な場合は`getgenmat`を使うこと。

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
            M, next_idx, eig, ao_labels = _parse_matrix_from(lines, found + 1)
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


def getmatgen(out_path: str, string: str, all: bool = False
           ) -> Union[List[MatBlock], Optional[MatBlock]]:
    """
    Get matrices with a more complicated format named `string` from `out_path`.

    `getmat`の拡張版です。以下の情報をが追加で拾ってきます。
    - 行が数値なら数値インデックスで、行が文字列なら `ao_labels` として返す。
      ただし文字列行（ラベル）の場合、全ブロックで同じ並びでないとエラー。
    - 固有値・固有ベクトル（eigヘッダ）を検出した場合は `eig` を併せて返す。
    これらは返り値(result = getmatgen(...))のインスタンスとして以下のように保存されています。

    result.M : 行列（固有ベクトル）
    result.eig : 固有値
    result.ao_labels : AOラベル


    Parameters
    ----------
    out_path : str
        出力ファイルのパス
    string : str
        見出し（この文字列を含む行の直後から行列ブロックを探す）
    all : bool, default False
        True: 見つかったすべてを List[MatBlock] で返す。
        False: 最初の 1 件のみ MatBlock を返す（無ければ None）。

    Returns
    -------
    List[MatBlock] | MatBlock | None

    MatBlock.M : 行列情報
    MatBlock.eig : 固有値情報
    MatBlock.ao_labels : AOラベル情報

    """
    lines = Path(out_path).read_text(encoding="utf-8", errors="ignore").splitlines()

    results: List[MatBlock] = []
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
            M, next_idx, eig, ao_labels = _parse_matrix_from(lines, found + 1)
        except RuntimeError:
            # この見出しの直後に行列ブロックが無い/壊れている → 次行から再探索
            i = found + 1
            continue

        block = MatBlock(M=M, eig=eig, ao_labels=ao_labels)

        if all:
            results.append(block)
            i = next_idx  # 次のブロック以降を探索
        else:
            print(f'Matrix "{string}" found. shape={M.shape}'
                  + (f", eig={len(eig)}" if eig else "")
                  + (f", labels={len(ao_labels)}" if ao_labels else ""))
            return block  # 最初の1件だけ返して終了

    if not results or not all:
        print(f'No matrix "{string}" found.')
        return None
    else:
        shapes = [blk.M.shape for blk in results]
        print(f'{len(results)} "{string}" matrices found. shapes={shapes}')
        return results

import sys
def printmat(M: np.array, title: str = None, eig: np.array = None, mmax: int = 5, n: int = None, m: int = None, format: str = "12.7f", ao_labels: list = None,  file: str = None):
    """Function:
    Print out A in a readable format.

        M         :  1D or 2D numpy array of dimension
        eig       :  Given eigenvectros M[:,i], eig[i] are corresponding eigenvalues (ndarray or list)
        file      :  file to be printed
        mmax      :  maxixmum number of columns to print for each block
        title     :  Name to be printed
        n,m       :  Need to be specified if M is a matrix,
                     but loaded as a 1D array
        format    :  Printing format
        ao_labels :  AO labels instead of integers for rows.

    Author(s): Takashi Tsuchimochi
    """
    if isinstance(M, list):
        dimension = 1
    elif isinstance(M, np.ndarray):
        dimension = M.ndim
    if dimension == 0 or dimension > 2:
        error("Neither scalar nor tensor is printable with printmat.")
    
    if file is None:
        file = sys.stdout
        should_close = False
    elif isinstance(file, str):   # ファイル名が渡された場合
        file = open(file, "w")
        should_close = True
    else:
        raise TypeError("'file' should be None or str.")

    if True:
        if title is not None:
            file.write(f" {title}\n")

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
            n, m = M.shape
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
                file.write('\n')
                for j in range(n):
                    if ao_labels is None:
                        file.write(f" {j:4d} ")
                    else:
                        file.write(f" {ao_labels[j]:12s}")
                    for i in range(imin-1, imax):
                        file.write(f"{M[j][i]:{format}} ")
                    file.write('\n')
        elif dimension == 1:
            if n is None or m is None:
                if isinstance(M, list):
                    n = len(M)
                    m = 1
                elif isinstance(M, np.ndarray):
                    n = M.size
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
                        file.write(f"  {M[j + i*n]:{format}}  ")
                    file.write('\n')
        file.write('\n')
        file.flush()
    if should_close:
        file.close()        
