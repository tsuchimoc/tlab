import numpy as np
from .matrix import printmat, MatBlock
from .linalg import symm, Lowdin_orthonormalization

def davidson(H, S=None, nroots=1, diag = None, init_guesses=None, threshold=1e-5, maxiter=100, verbose=0, shift_operator=None, parallel=False):

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
    
    if init_guesses is None:
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
        states = init_guesses
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
    shift_expectation = 0
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
 
