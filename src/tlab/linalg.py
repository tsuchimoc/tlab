import numpy as np
import scipy as sp
def symm(B):
    """
    Given a vector B with the dimension n*(n+1)//2,
    form a symmetric matrix,

                                         |  0   1   3   6  10  ... |
                                         |  1   2   4   7  11  ... |
    B = [0,1,2,3,4,5,...]   -->     A =  |  3   4   5   8  12  ... |
                                         |  6   7   8   9  13  ... |
                                         | 10  11  12  13  14  ... |
                                         | ..  ..  ..  ..  ..  ... |
                              
    Args:
        B (1darray): Lower-triangular of A
    Returns:
        A (2darray): Symmetric matrix of (n x n)

    Author(s): Takashi Tsuchimochi
    """
    lenB = len(B)
    n = (-1+np.sqrt(1+8*lenB))//2
    if not n.is_integer():
        raise ValueError(f'Vector is not [N*(N+1)] in symm (lenB = {lenB})')
    n = int(n)
    dtype = 'float'
    for element in B:
        if 'complex' in str(type(element)):
            dtype = 'complex'
    if dtype == 'complex':
        A = np.zeros((n,n), complex)
    else:
        A = np.zeros((n,n))

    ij = 0
    for i in range(n):
        for j in range(i+1):
            A[i,j] = A[j,i] = B[ij]
            ij += 1
    if dtype == 'complex':
        for i in range(n):
            for j in range(i+1):
                A[i,j] = np.conjugate(A[i,j])
    return A

def skew(B):
    """
    Given a vector B with the dimension n*(n-1)//2,
    form a skew matrix,

                                       |  0  -1  -2  -4  -7  ... |  
                                       |  1   0  -3  -5  -8  ... |
    B = [1,2,3,4,5,...]   -->     A =  |  2   3  -0  -6  -9  ... |
                                       |  4   5   6   0 -10  ... |
                                       |  7   8   9  10   0  ... |
                                       | ..  ..  ..  ..  ..  ... |
                              
    Args:
        B (1darray): Lower-triangular of A
    Returns:
        A (2darray): Skew matrix of (n x n)

    Author(s): Takashi Tsuchimochi
    """
    lenB = len(B)
    n = (1+np.sqrt(1+8*lenB))//2
    if not n.is_integer():
        raise ValueError(f'Vector is not [N*(N-1)] in skew (lenB = {lenB})')
    n = int(n)
    A = np.zeros((n,n))
    

    ij = n*(n-1)//2 -1
    for i in range(n-1, 0, -1):
        for j in range(i, 0, -1):
            A[i, j-1] = B[ij]
            ij -= 1
    for i in range(n):
        for j in range(i):
           A[j,i] = - A[i,j]
    return A
    
def vectorize_symm(A):
    """
    Given the nxn square matrix A (symmetric),
    vectrize its lower-triangular part (diagonals+off-diagonals), 
    return it as a vector. 

         |  0   1   3   6  10  ... |
         |  1   2   4   7  11  ... |
    A =  |  3   4   5   8  12  ... |   -->  B = [1,2,3,4,5,...]
         |  6   7   8   9  13  ... |
         | 10  11  12  13  14  ... |
         | ..  ..  ..  ..  ..  ... |

    Args:
        A (2darray): Symmetric matrix
    Returns:
        B (1darray): Lower-triangular of A

    Author(s): Takashi Tsuchimochi
    """
    N = A.shape[0]
    if A.shape[1] != N:
        raise ValueError(f'Matrix is not square in vectorize_symm ({A.shape[0], A.shape[1]})')
    NTT = N*(N+1)//2
    if 'complex' in A.dtype:
        B = np.zeros((NTT), complex)
    else:
        B = np.zeros((NTT))

    ij = 0
    for i in range(0, N):
        for j in range(i+1):
            B[ij] = A[i,j]
            ij += 1
    return B

def vectorize_skew(A):
    """
    Given the nxn square matrix A (skew),
    vectorize its lower-triangular part (off-diagonals), 
    return it as a vector. 

         |  0  -1  -2  -4  -7  ... |
         |  1   0  -3  -5  -8  ... |
    A =  |  2   3  -0  -6  -9  ... |   -->  B = [1,2,3,4,5,...]
         |  4   5   6   0 -10  ... |
         |  7   8   9  10   0  ... |
         | ..  ..  ..  ..  ..  ... |

    Args:
        A (2darray): Skew matrix
    Returns:
        B (1darray): Lower-triangular of A

    Author(s): Takashi Tsuchimochi
    """

    N = A.shape[0]
    if A.shape[1] != N:
        raise ValueError(f'Matrix is not square in vectorize_skew ({A.shape[0], A.shape[1]})')

    NTT = N*(N-1)//2
    B = np.zeros((NTT))

    ij = 0
    for i in range(1, N):
        for j in range(i):
            B[ij] = A[i,j]
            ij += 1
    return B


def root(A, eps=1e-8):
    """Function:
    Get canonical (non-symmetric) A^1/2. 

    Author(s): Takashi Tsuchimochi
    """

    #u, s, vh = np.linalg.svd(A, hermitian=True)
    s,u = np.linalg.eigh(A)
    mask = s >= eps
    red_u = np.compress(mask, u, axis=1)
    # Diagonal matrix of s**1/2
    s2 = np.diag([np.sqrt(i) for i in s if i > eps])
    S2 = red_u @ s2 @ np.conjugate(red_u).T
    return S2

def root_inv(A, eps=1e-8):
    """Function:
    Get canonical (non-symmetric) A^-1/2. 
    Dimensions may be reduced.

    Author(s): Takashi Tsuchimochi
    """

    #u, s, vh = np.linalg.svd(A, hermitian=True)
    s,u = np.linalg.eigh(A)
    mask = s >= eps
    red_u = np.compress(mask, u, axis=1)
    # Diagonal matrix of s**-1/2
    sinv2 = np.diag([1/np.sqrt(i) for i in s if i > eps])
    Sinv2 = red_u@sinv2
    return Sinv2
    
def Lowdin_orthonormalization(S, thres=1.0e-8):
    """Function:
    Get symmetric A^-1/2 based on Lowdin's orthonormalization. 
    Dimensions may be reduced.

    Author(s): Takashi Tsuchimochi
    """
    eig,u = np.linalg.eigh(S)
    #s^(-1/2)
    for i in range(len(eig)):
        if eig[i] < thres:
            eig[i] = 0.0e0
    eig_2 = np.diag(eig)
    eig_2 = np.linalg.pinv(eig_2)
    eig_2 = sp.linalg.sqrtm(eig_2)
    return u@eig_2@np.conjugate(u.T)
    
def pinv(A, eps=1e-6, hermitian=False):
    """Function:
    Get Moore-Penrose Pseudo-Inverse of the matrix A.

    Author(s): Takashi Tsuchimochi
    """
    if hermitian:
        a, v = np.linalg.eigh(A)
        a_reg = np.zeros(len(a), float)
        for k in range(a.shape[0]):
            if abs(a[k]) > eps:
                a_reg[k] = 1/a[k]
        return v @ np.diag(a_reg) @ np.conjugate(v.T)

    else:
        u, a, v = np.linalg.svd(A)
        a_reg = np.zeros(len(a), float)
        for k in range(a.shape[0]):
            if abs(a[k]) > eps:
                a_reg[k] = 1/a[k]
        return v.T @ np.diag(a_reg) @ u.T

def nullspace(A, eps=1e-8):
    """Function:
    Get the nullspace and range of A. 

    Author(s): Takashi Tsuchimochi
    """

    s,u = np.linalg.eigh(A)
    mask = s >= eps
    Range = np.compress(mask, u, axis=1)
    mask = s < eps
    Null = np.compress(mask, u, axis=1)
    return Null, Range
