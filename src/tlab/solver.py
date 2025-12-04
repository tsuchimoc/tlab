import numpy as np
from .matrix import printmat, MatBlock
from .linalg import symm, Lowdin_orthonormalization

def davidson(H, S=None, *, nroots=1, diag=None, Sdiag=None, init_guess=None, threshold=1e-5, maxiter=100, verbose=0, shift_operator=None):
    """
    Solve the (generalized) eigenvalue problem using the Davidson algorithm:

        H c = E c
        (or H c = E S c if S is given)

    Parameters
    ----------
    H : (N, N) ndarray or callable function to compute sigma-vectors H(c) = H@c
        Hamiltonian (or coefficient) matrix.
    S : (N, N) ndarray or callable function to compute sigma-vectors S(c) = S@c, optional
        Overlap matrix. If None, the standard eigenproblem H c = E c is solved.
    nroots : int, optional
        Number of lowest eigenpairs to compute (default: 1).
    diag : (N,) ndarray, optional
        Precomputed diagonal elements of H (used as preconditioner).
        If None, they are extracted from H.
    Sdiag : (N,) ndarray, optional
        Precomputed diagonal elements of S (used as preconditioner).
        If None, they are extracted from S.
    init_guess : (N, k) ndarray, optional
        Initial guess vectors for the subspace (columns). If None, random
        or canonical basis vectors are used.
    threshold : float, optional
        Convergence tolerance on the residual norm (default: 1e-5).
    maxiter : int, optional
        Maximum number of Davidson iterations (default: 100).
    verbose : int, optional
        Verbosity level. If >0, residual norms and eigenvalues are printed
        during the iterations.
    shift_operator : 

    Returns
    -------
    MatBlock
        Result object with attribute .M containing the eigenvectors 
        and attirbute .eig containing the eigenvalues.

    Notes
    -----
    - The Davidson method builds a Krylov-like subspace iteratively and
      accelerates convergence using diagonal preconditioning.
    - The residual for an approximate eigenpair (E, c) is
        r = H c - E S c
      and convergence is achieved when ||r|| < threshold.
    - For large sparse problems, `H` and `S` may be provided as
      `callable` function to avoid explicit matrix storage.
    - If the overlap matrix S is supplied, it should be symmetric positive-definite.
    """
    
    if isinstance(H, np.ndarray):
        H_mat = H
        print('Found full matrix H')
        if H_mat.shape[0] != H_mat.shape[1]:
            raise TypeError(f"H({H_mat.shape[0]}, {H_mat.shape[1]}) is not a square matrix.")
        if np.linalg.norm(H_mat - H_mat.T) > 1e-8:
            raise TypeError("H is not a symmetric matrix.")
        NDim = H_mat.shape[0]

        def H_op(c):
            return H_mat @ c

    elif callable(H):
        print('Direct Davidson')
        H_mat = None
        if diag is None:
            raise Exception("diag has to be defined when H is given as a callable.")
        NDim = len(diag)

        # H 自体が H(c) を返す演算子であるとみなす
        H_op = H

    else:
        raise TypeError("H must be either a numpy array or a callable(c)->Hc.")

    print('Dimension of H:', NDim)

    # --- S の扱いを統一 ---
    if S is None:
        S_mat = None
        Sdiag_list = np.ones(NDim, float)
        def S_op(c):
            # 単位行列として働く
            return c

    elif isinstance(S, np.ndarray):
        S_mat = S
        print('Dimension of S:', S_mat.shape)
        if S_mat.shape[0] != S_mat.shape[1]:
            raise TypeError("S is not a square matrix.")
        if np.linalg.norm(S_mat - S_mat.T) > 1e-8:
            raise TypeError("S is not a symmetric matrix.")
        if S_mat.shape[0] != NDim:
            raise TypeError("Dimensions of H and S do not match.")
        Sdiag_list = np.diag(S)

        def S_op(c):
            return S_mat @ c

    elif callable(S):
        S_mat = None
        if Sdiag is None:
            raise Exception("Sdiag has to be defined when S is given as a callable.")
        if NDim != len(Sdiag):
            raise TypeError("Dimensions of H and S do not match.")
        Sdiag_list = Sdiag

        # S 自体が S(c) を返す演算子
        S_op = S
    else:
        raise TypeError("S must be None, a numpy array, or a callable(c)->Sc.")

#    if S is None:
#        S = np.diag(np.ones(NDim))
        
    if diag is None:
        Hdiag_list = np.diag(H)
    else:
        try:
            Hdiag_list = diag.flatten()
        except:
            Hdiag_list = diag
    if Sdiag is not None:
        try:
            Sdiag_list = Sdiag.flatten()
        except:
            Sdiag_list = Sdiag
    Ssub = np.zeros(0, float)
    norms = np.zeros(nroots)
    converge = [False for x in range(nroots)]
    Hstates = []
    Sstates = []
    icyc = 0
    new_state = np.zeros(NDim, float)
    ioff = 0
    
    if init_guess is None:
        ### Find the nroots lowest diagonals
        ### Get lowest nroots states according to the diagonals
        from heapq import nsmallest
        result = [(value, k) for k, value in enumerate(Hdiag_list)]
        min_result = nsmallest(nroots, result)
        
        states = []
        if verbose>=0:
            print("Initial guess estimated as: ", end='')
        for k in range(nroots):
            if verbose>=0:
                print(f"{min_result[k][1]}", end='')
            v = np.zeros(NDim, float)
            v[min_result[k][1]] = 1
            states.append(v)
            if verbose>=0:
                if k != nroots-1:
                    print(', ',end='')
                else:
                    print('')
#    else:
#        states = _normalize_init_guess(init_guess, NDim)
#    print(states)
#    ## Orthogonalize
#    for i in range(len(states)):
#        Sstates.append(S@states[i])
#    for i in range(len(states)):
#        for j in range(i+1):
#            Sij = states[j] @ Sstates[i]
#            Ssub = np.append(Ssub, Sij)
#        
#        Ssub_symm = symm(Ssub)
#    X = Lowdin_orthonormalization(Ssub_symm, thres=1e-6)
#     
#    for i in range(len(states)):
#        vec[i] *= 0
#        for j in range(X.shape[0]):
#            vec[i] += states[j] * X[j, i]
#    states = list(vec.copy())
    else:
        states = _normalize_init_guess(init_guess, NDim)
    
    # states: list of np.ndarray, each shape=(NDim,)
    
    ## Orthogonalize
    nvec = len(states)

    vec = np.zeros((nvec, NDim), float)
    Hvec = np.zeros((nvec, NDim), float)
    
    # Sstates = S @ state
    Sstates = [S_op(v) for v in states] 
    
    # Ssub_ij = <state_i | S | state_j>
    for i in range(nvec):
        for j in range(i+1):
            Sij = states[i] @ Sstates[j]
            Ssub = np.append(Ssub, Sij)
    
    Ssub_symm = symm(Ssub)
    
    X = Lowdin_orthonormalization(Ssub_symm, thres=1e-6)
    
    # 直交化された線形結合を作る
    for i in range(nvec):
        # vec[i] = Σ_j states[j] * X[j,i]
        vec[i] *= 0
        for j in range(nvec):
            vec[i] += states[j] * X[j, i]
    
    states = list(vec.copy()) 
        
    #printmat(np.array(states).T, 'orthogonalized states')
    Sstates = []
    Hsub = np.zeros(0, float)
    Ssub = np.zeros(0, float)  
    nroots_ = len(states)
    if verbose >= 0:
        print('Cycle    State       Energy        Grad')
    while icyc < maxiter:
        ### Subspace Hamiltonian
        ntargets = len(states) - len(Hstates) 
        len_states = len(states)
        for i in range(ioff, ioff+ntargets):
            if verbose >= 2:
                printmat(states[i], 'Trial vector')
            Hstates.append(H_op(states[i]))
            Sstates.append(S_op(states[i]))
            for j in range(i+1):
                Hij = states[j] @ Hstates[i]
                Hsub = np.append(Hsub, Hij)
                Sij = states[j] @ Sstates[i]
                Ssub = np.append(Ssub, Sij)

        Hsub_symm = symm(Hsub)
        Ssub_symm = symm(Ssub)
        E, V = np.linalg.eigh(Hsub_symm)
        if verbose >= 1:
            printmat(Hsub_symm, title='Subspace Hamiltonian')
            printmat(V, eig=E, title='Eigen-set')
        reset = False 
        for i in range(min(nroots, len_states)):
            vec[i] *= 0
            Hvec[i] *= 0
            for j in range(V.shape[0]):
                vec[i] += states[j] * V[j, i]
                Hvec[i] += Hstates[j] * V[j, i] 
            residual = Hvec[i] - E[i] * S_op(vec[i])
            if verbose >= 2:
                printmat(residual, 'Residual vector')
            #print('state',i)
            #printmat(Hvec.T, 'AX vector')
            #printmat(residual, 'Residual vector')
            norms[i] = np.linalg.norm(residual)
            #norms[i] = np.sqrt(residual.T@S@residual)
            if norms[i] < threshold:
                converge[i] = True
            else:
                converge[i] = False
                new_state *= 0 
                for k in range(NDim):
                    if abs(Hdiag_list[k] - E[i] *Sdiag_list[k])   > 1e-8:
                        new_state[k] = - residual[k] / (Hdiag_list[k] - E[i]*Sdiag_list[k])
                        #if verbose >= 3:
                        #    print(f"{k}  {residual[k]:16.10f} / {Hdiag_list[k] - E[i]*Sdiag_list[k]:16.10f}")
                    else:
                        # Changed 1e14 to 1e4 (just perturb a little bit...)
                        new_state[k] = - residual[k] / 1e4
                if verbose >= 2:
                    printmat(new_state, 'Updated vector (not orthogonal)')
                #printmat(new_state,'new_state (unorthonormal')
                # Gram-Schmidt orthogonalization
                state = new_state.copy()
                Sstate = S_op(state)
                norm2 = np.sqrt(state.T@Sstate)
                #X = (np.array(states)).T
                state /= norm2
                if norm2 < 1e-6:
                    reset = True
                for old_state in states:
                    state -= old_state * (old_state @ S_op(state))
                    #norm2 = np.linalg.norm(state)
                    norm2 = np.sqrt(state.T@S_op(state))
                    #state.normalize(norm2)
                if norm2 < 1e-6:
                        ### This means the new vector is spanned by other vectors in the subspace. 
                        ### Skip this state.
                        break
                else:
                    norm2 = np.sqrt(state.T@S_op(state))
                    print('state.T @ S @ state',norm2)
                    state /= norm2
                    #print(state.T@S@state)
                    states.append(state)
                    for old_state in states[:min(len_states, nroots)]:
                        #print_state(old_state)
                        if abs(old_state @ state) > 1e-8:
                            reset = True
                if verbose >= 2:
                    printmat(state, 'Updated vector (orthogonal)')
        #printmat(s)
        if verbose >= 0:
            print(f'[{icyc:3d}]        0:  {E[0]:+.10f}   {norms[0]:.2e}  ', end='')
            if converge[0]:
                print('converged')
            else:
                print('')
            for k in range(1, min(nroots, len_states)):
                print(f'          {k:4d}:  {E[k]:+.10f}   {norms[k]:.2e}  ', end='')
                if converge[k]:
                    print('converged')
                else:
                    print('')
        if all (converge): 
            print(f'\nAll {nroots} states converged.')
            for k in range(nroots):
                print(f'State {k:5d}:  {E[k]:+.10f}')
            break
        if len(states) <= nroots and all (converge[:len(states)]):
            if verbose >= 0:
                print(f'{len(states)} states converged.')
                print(f'Rest {nroots - len(states)} states not found.')
            nroots = len(states)
            converge = converge[:len(states)]
            break

        ioff += ntargets
        icyc += 1
    nret = min(nroots, len(E))
    result = MatBlock(M=vec[:nret].T, eig=E[:nret], ao_labels=None)
    return result


# ---- helpers ---------------------------------------------------------------

import numpy as np

def _normalize_init_guess(init_guess, NDim: int):
    """
    init_guess を「list of np.ndarray(shape=(NDim,))」に正規化する。

    受け付ける例:
      - list:
          各要素が 1 本のベクトル
          ・(NDim,) の ndarray → そのまま
          ・(NDim,1) や (1,NDim) など → flatten して (NDim,) に
      - ndarray:
          ・shape = (NDim,)       → 1 本 → [ (NDim,) ]
          ・shape = (NDim, n)     → n 本 → 各列を (NDim,) に
          ・shape = (n, NDim)     → n 本 → 各行を (NDim,) に
      - それ以外:
          np.asarray で ndarray 化できるなら ndarray のケースへ
    """

    def _to_vec1d(x):
        """x を np.ndarray(shape=(NDim,)) に変換する小ヘルパー"""
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            if arr.size != NDim:
                raise ValueError(
                    f"init_guess element has size {arr.size}, expected {NDim}."
                )
            return arr.copy()
        elif arr.ndim == 2:
            if arr.size != NDim:
                raise ValueError(
                    f"init_guess element has shape {arr.shape}, "
                    f"total size {arr.size}, expected {NDim}."
                )
            # (NDim,1) / (1,NDim) / その他 2次元だが要素数 NDim → フラットにして 1D
            return arr.reshape(NDim,)
        else:
            raise ValueError(
                f"init_guess element has ndim={arr.ndim}, cannot convert to 1D vector."
            )

    # --- 1) list の場合 ---
    if isinstance(init_guess, list):
        states = []
        for v in init_guess:
            states.append(_to_vec1d(v))
        return states

    # --- 2) ndarray の場合 ---
    if isinstance(init_guess, np.ndarray):
        arr = np.asarray(init_guess, dtype=float)

        # (NDim,) → 1 本
        if arr.ndim == 1:
            if arr.size != NDim:
                raise ValueError(
                    f"init_guess has size {arr.size}, expected {NDim}."
                )
            return [arr.copy()]

        # (NDim, n) → 列方向に n 本のベクトル
        if arr.ndim == 2:
            if arr.shape[0] == NDim:
                n = arr.shape[1]
                return [arr[:, i].copy() for i in range(n)]
            # (n, NDim) → 行方向に n 本のベクトルとして解釈
            if arr.shape[1] == NDim:
                n = arr.shape[0]
                return [arr[i, :].copy() for i in range(n)]
            raise ValueError(
                f"init_guess has shape {arr.shape}, "
                f"cannot interpret as (NDim, n) or (n, NDim) with NDim={NDim}."
            )

        # それ以外
        raise ValueError(
            f"init_guess ndarray ndim={arr.ndim} not supported."
        )

    # --- 3) list でも ndarray でもない → ndarray 化を試みる ---
    try:
        arr = np.asarray(init_guess, dtype=float)
    except Exception as e:
        raise TypeError(
            "init_guess must be list, ndarray, or convertible to ndarray."
        ) from e

    # ndarray として再処理（上の分岐に入る）
    return _normalize_init_guess(arr, NDim)

def _build_projector(Null, N):
    if Null is None:
        return None
    Y = np.asarray(Null)
    G = Y.T @ Y
    return np.eye(N) - Y @ np.linalg.inv(G) @ Y.T

def _jacobi_update(R, diag):
    N, nvec = R.shape
    dX = np.empty_like(R)
    for k in range(N):
        d = diag[k]
        dX[k, :] = - R[k, :] / d if abs(d) > 1e-12 else - R[k, :] / 1e4
    return dX

def _final_project(X, Null):
    if Null is None:
        return X
    N = X.shape[0]
    P = _build_projector(Null, N)
    return P @ X

# ---- DIIS (multi-RHS). Nullは最終解にのみ適用 -----------------------------

def _solve_diis(A, B, *, diag=None, init_guess=None,
                threshold=1e-6, maxiter=200, verbose=0,
                Null=None, mmax=100, b_reg_scale=1e-14):

    N, nvec = B.shape
    diag = np.diag(A).astype(float) if diag is None else np.asarray(diag, float).ravel()

    X = (-B / diag.reshape(-1,1)) if init_guess is None else init_guess.copy()
    trials = [X.copy()]
    residuals = [A @ X + B]

    print("Cycle    Residual Norm (DIIS)")
    for it in range(maxiter):
        R = A @ X + B
        rn = np.linalg.norm(R)
        print(f"[{it:2d}]   {rn:.3e}")
        if rn < threshold:
            X = _final_project(X, Null)
            print("converged")
            return MatBlock(M=X, eig=None, ao_labels=None)

        # Jacobi 前進
        X_new = X + _jacobi_update(R, diag)

        # DIIS 空間更新（上限 mmax）
        trials.append(X_new.copy())
        residuals.append(A @ X_new + B)
        if len(trials) > mmax:
            trials.pop(0); residuals.pop(0)

        # DIIS 係数 → 外挿
        m = len(residuals)
        if m >= 2:
            Bmat = np.zeros((m+1, m+1), float)
            flats = [r.reshape(-1) for r in residuals]
            for i in range(m):
                for j in range(m):
                    Bmat[i, j] = np.dot(flats[i], flats[j])
            # 微小正則化で安定化
            tr = np.trace(Bmat[:m,:m])
            Bmat[:m,:m] += (b_reg_scale * (tr / max(m,1) if tr != 0 else 1.0)) * np.eye(m)
            Bmat[m, :m] = 1.0
            Bmat[:m, m] = 1.0

            rhs = np.zeros((m+1,1)); rhs[m,0] = 1.0
            try:
                coeff = np.linalg.solve(Bmat, rhs)[:m]
                X = sum(coeff[i,0] * trials[i] for i in range(m))
            except np.linalg.LinAlgError:
                if verbose: print("B ill-conditioned; skip DIIS step.")
                X = X_new
        else:
            X = X_new

    print("Warning: DIIS not converged")
    X = _final_project(X, Null)
    return MatBlock(M=X, eig=None, ao_labels=None)

# ---- MINRES（symmetric, single RHS）。Nullは最終解にのみ適用 --------------

def _solve_minres(A, b, *, diag=None, init_guess=None,
                        threshold=1e-6, maxiter=200, verbose=0,
                        Null=None):
    N = A.shape[0]
    diag = np.diag(A).astype(float) if diag is None else np.asarray(diag, float).ravel()
    eps = 1e-14
    s = 1.0 / np.sqrt(np.clip(diag, eps, None))   # S = diag(1/sqrt(diag))

    def K_mv(v):            # K v = S A S v
        return s * (A @ (s * v))

    fprime = - (s * b)      # f' = - S b
    y0 = np.zeros(N) if init_guess is None else (init_guess / np.where(s!=0, s, 1.0))

    # MINRES on operator K
    Nloc = fprime.shape[0]
    r0 = fprime - K_mv(y0)
    beta1 = np.linalg.norm(r0)
    if beta1 == 0.0:
        x = (s * y0).reshape(N,1)
        x = _final_project(x, Null)
        return x

    v_prev = np.zeros(Nloc)
    v = r0 / beta1
    V = [v.copy()]
    alphas, betas = [], []

    y_best = y0.copy()
    res_best = beta1

    print("Cycle    Residual Norm (MINRES)")
    for k in range(1, maxiter+1):
        Kv = K_mv(v)
        alpha = np.dot(v, Kv); alphas.append(alpha)
        r = Kv - (betas[-1] * v_prev if k > 1 else 0.0) - alpha * v
        beta = np.linalg.norm(r)
        if k == 1: betas.append(beta)
        v_next = r / beta if beta > 0 else np.zeros_like(r)
        if beta > 0 and k < maxiter: V.append(v_next.copy())

        kdim = k
        T = np.diag(alphas[:kdim])
        for i in range(kdim-1):
            T[i, i+1] = betas[i]
            T[i+1, i] = betas[i]
        e1 = np.zeros(kdim); e1[0] = beta1
        yk, *_ = np.linalg.lstsq(T, e1, rcond=None)

        yk = y0 + np.column_stack(V[:kdim]) @ yk
        res = np.linalg.norm(K_mv(yk) - fprime)
        if verbose: print(f"[{k:3d}]   ||r_y||={res:.3e}")
        else:       print(f"[{k:3d}]   {res:.3e}")

        if res < res_best: res_best, y_best = res, yk.copy()
        if res < threshold:
            x = (s * yk).reshape(N,1)
            x = _final_project(x, Null)
            print("converged"); return x

        if k == maxiter or beta == 0.0: break
        v_prev, v = v, v_next
        if k >= 2: betas.append(beta)

    print("Warning: MINRES not converged")
    x = (s * y_best).reshape(N,1)
    x = _final_project(x, Null)
    return x

# ---- public API ------------------------------------------------------------

def linearsolver(A, B, *, diag=None, init_guess=None,
                 threshold=1e-6, maxiter=200, verbose=0,
                 Null=None, solver="DIIS",
                 diis_mmax=100, diis_breg_scale=1e-14):
    """
    Solve the linear system(s)
        A X + B = 0

    Parameters
    ----------
    A : (N, N) ndarray
        Coefficient matrix (symmetric for MINRES).
    B : (N, n) ndarray
        Right-hand side(s). Each column is one vector b.
    diag : (N,) ndarray, optional
        Diagonal elements of A (used for Jacobi preconditioning in DIIS,
        or for preconditioner in MINRES). If None, np.diag(A) is used.
    init_guess : (N, n) ndarray, optional
        Initial guess for the solution(s). If None, a Jacobi-based guess is used.
    threshold : float, optional
        Convergence tolerance for the residual norm (default: 1e-6).
    maxiter : int, optional
        Maximum number of iterations (default: 200).
    verbose : int, optional
        Verbosity level. If >0, residual norms are printed every iteration.
    Null : (N, k) ndarray, optional
        Basis for null space. If provided, the final solution is projected
        once at the end. No projection is applied during iterations.
    solver : {"DIIS", "MINRES"}, optional
        Solver to use:
            - "DIIS"   : Direct Inversion in the Iterative Subspace.
                          Supports multiple RHS. Uses Jacobi updates and
                          residual minimization with DIIS extrapolation.
            - "MINRES" : Minimum Residual Method preconditioned with diagonal preconditioner.
                          Symmetric A, single RHS only. Lanczos iteration
                          with exact least-squares solve for residual minimization.
                          Uses M = diag(diag).
    diis_mmax : int, optional
        Maximum number of stored vectors in DIIS subspace (default: 100).
    diis_breg_scale : float, optional
        Regularization strength added to the DIIS B-matrix for stability.

    Returns
    -------
    MatBlock
        Result object with attribute .M containing the solution vector(s).

    Notes
    -----
    - DIIS supports multiple right-hand sides (columns of B).
    - MINRES supports only a single RHS.
    - Null projection is applied only once at the final solution,
      since residuals are guaranteed to be orthogonal to the null space.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    if B.ndim == 1:
        B = B.reshape(-1,1)
    N, nvec = B.shape

    if solver.upper() == "DIIS":
        return _solve_diis(A, B, diag=diag, init_guess=init_guess,
                           threshold=threshold, maxiter=maxiter, verbose=verbose,
                           Null=Null, mmax=diis_mmax, b_reg_scale=diis_breg_scale)

    elif solver.upper() == "MINRES":
        if nvec != 1:
            raise ValueError("MINRES は単一 RHS のみ対応です。B.shape=(N,1) にしてください。")
        x0 = None if init_guess is None else init_guess[:,0]
        x = _solve_minres(A, B[:,0], diag=diag, init_guess=x0,
                                threshold=threshold, maxiter=maxiter,
                                verbose=verbose, Null=Null)
        return MatBlock(M=x, eig=None, ao_labels=None)

    else:
        raise ValueError("solver must be 'DIIS' or 'MINRES'")

