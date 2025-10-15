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

