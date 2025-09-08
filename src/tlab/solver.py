import numpy as np
from .matrix import printmat, MatBlock
from .linalg import symm, Lowdin_orthonormalization

def davidson(H, S=None, *, nroots=1, diag=None, init_guess=None, threshold=1e-5, maxiter=100, verbose=0, shift_operator=None):
    """
    Solve the (generalized) eigenvalue problem using the Davidson algorithm:

        H c = E c
        (or H c = E S c if S is given)

    Parameters
    ----------
    H : (N, N) ndarray or LinearOperator
        Hamiltonian (or coefficient) matrix.
    S : (N, N) ndarray or LinearOperator, optional
        Overlap matrix. If None, the standard eigenproblem H c = E c is solved.
    nroots : int, optional
        Number of lowest eigenpairs to compute (default: 1).
    diag : (N,) ndarray, optional
        Precomputed diagonal elements of H (used as preconditioner).
        If None, they are extracted from H.
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
      `scipy.sparse.linalg.LinearOperator` to avoid explicit matrix storage.
    - If the overlap matrix S is supplied, it should be symmetric positive-definite.
    """

    print('Dimension of H:',H.shape)
    
    if H.shape[0] != H.shape[1]:
        raise TypeError("H is not a square matrix.")
    if np.linalg.norm(H-H.T) > 1e-8:
        raise TypeError("H is not a symmetric matrix.")
    if S is not None:
        print('Dimension of S:',S.shape)
        if S.shape[0] != S.shape[1]:
            raise TypeError("S is not a square matrix.")
        if np.linalg.norm(S-S.T) > 1e-8:
            raise TypeError("S is not a symmetric matrix.")

    NDim = H.shape[0]
    if S is None:
        S = np.diag(np.ones(NDim))
        
    if diag is None:
        Hdiag_list = np.diag(H)
    else:
        Hdiag_list = diag
    Sdiag_list = np.diag(S)
    Hsub = np.zeros(0, float)
    Ssub = np.zeros(0, float)
    norms = np.zeros(nroots)
    converge = [False for x in range(nroots)]
    Hstates = []
    Sstates = []
    icyc = 0
    vec = np.zeros((nroots, NDim), float)
    Hvec = np.zeros((nroots, NDim), float)
    new_state = np.zeros(NDim, float)
    ioff = 0
    
    if init_guess is None:
        ### Find the nroots lowest diagonals
        ### Get lowest nroots states according to the diagonals
        from heapq import nsmallest
        result = [(value, k) for k, value in enumerate(Hdiag_list)]
        min_result = nsmallest(nroots, result)
        
        states = []
        print("Initial guess estimated as: ", end='')
        for k in range(nroots):
            print(f"{min_result[k][1]}", end='')
            v = np.zeros(NDim, float)
            v[min_result[k][1]] = 1
            states.append(v)
            if k != nroots-1:
                print(', ',end='')
            else:
                print('')
    else:
        states = init_guess
    ## Orthogonalize
    for i in range(len(states)):
        Sstates.append(S@states[i])
    for i in range(len(states)):
        for j in range(i+1):
            Sij = states[j] @ Sstates[i]
            Ssub = np.append(Ssub, Sij)
        
        Ssub_symm = symm(Ssub)
    X = Lowdin_orthonormalization(Ssub_symm, thres=1e-6)
     
    for i in range(len(states)):
        vec[i] *= 0
        for j in range(X.shape[0]):
            vec[i] += states[j] * X[j, i]
    states = list(vec.copy())
        
    #printmat(np.array(states).T, 'orthogonalized states')
    Sstates = []
    Ssub *= 0 
    nroots_ = len(states)
    print('Cycle  State       Energy      Grad')
    while icyc < maxiter:
        ### Subspace Hamiltonian
        ntargets = len(states) - len(Hstates) 
        len_states = len(states)
        for i in range(ioff, ioff+ntargets):
            if verbose:
                printmat(states[i], 'Trial vector')
            Hstates.append(H@states[i])
            Sstates.append(S@states[i])
            for j in range(i+1):
                Hij = states[j] @ Hstates[i]
                Hsub = np.append(Hsub, Hij)
                Sij = states[j] @ Sstates[i]
                Ssub = np.append(Ssub, Sij)

        Hsub_symm = symm(Hsub)
        Ssub_symm = symm(Ssub)
        #printmat(Hsub_symm, 'HSub')
        E, V = np.linalg.eigh(Hsub_symm)
        #printmat(V, eig=E, mmax=5)
        if verbose:
            printmat(Hsub_symm)
            printmat(V, eig=E)
        reset = False 
        for i in range(min(nroots, len_states)):
            vec[i] *= 0
            Hvec[i] *= 0
            for j in range(V.shape[0]):
                vec[i] += states[j] * V[j, i]
                Hvec[i] += Hstates[j] * V[j, i] 
            residual = Hvec[i] - E[i] * S@vec[i]
            if verbose:
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
                    #if abs(Hdiag_list[k] - E[i] *Sdiag_list[k])   > 1e-6 and Sdiag_list[k]>:
                    if abs(Hdiag_list[k] - E[i] *Sdiag_list[k])   > 1e-8:
                        new_state[k] = - residual[k] / (Hdiag_list[k] - E[i]*Sdiag_list[k])
                        if verbose:
                            print(f"{k}  {residual[k]:16.10f}, {Hdiag_list[k] - E[i]*Sdiag_list[k]:16.10f}")
                    else:
                        # Changed 1e14 to 1e4 (just perturb a little bit...)
                        new_state[k] = - residual[k] / 1e4
                #printmat(new_state,'new_state (unorthonormal')
                # Gram-Schmidt orthogonalization
                state = new_state.copy()
                Sstate = S@state
                norm2 = np.sqrt(state.T@Sstate)
                #X = (np.array(states)).T
                #printmat(X.T@S@X)
                state /= norm2
                if norm2 < 1e-6:
                    reset = True
                for old_state in states:
                    state -= old_state * (old_state @ S @ state)
                    #norm2 = np.linalg.norm(state)
                    norm2 = np.sqrt(state.T@S@state)
                    #state.normalize(norm2)
                if norm2 < 1e-6:
                        ### This means the new vector is spanned by other vectors in the subspace. 
                        ### Skip this state.
                        break
                else:
                    norm2 = np.sqrt(state.T@S@state)
                    state /= norm2
                    #print(state.T@S@state)
                    states.append(state)
                    for old_state in states[:min(len_states, nroots)]:
                        #print_state(old_state)
                        if abs(old_state @ state) > 1e-8:
                            reset = True
                        #print(abs(old_state @ state))
                if verbose:
                    printmat(state, 'Updated vector (orthogonal)')
                #printmat(state, 'Updated vector (orthogonal)')
        # Project out the range space
        s = np.zeros((len(states), len(states)))
        for j in range(len(states)):
            for i in range(len(states)):
                s[j,i] = (states[j] @ states[i]).real
        #printmat(s)
        if True:
                
            print(f'[{icyc:2d}]      0:  {E[0]:+.10f}   {norms[0]:.2e}  ', end='')
            if converge[0]:
                print('converged')
            else:
                print('')
            for k in range(1, min(nroots, len_states)):
                print(f'          {k}:  {E[k]:+.10f}   {norms[k]:.2e}  ', end='')
                if converge[k]:
                    print('converged')
                else:
                    print('')
        if all (converge): 
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
    result = MatBlock(M=vec.T, eig=E, ao_labels=None)
    return result

import numpy as np

# ---- helpers ---------------------------------------------------------------

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

