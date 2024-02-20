import pandas as pd
import numpy as np
from decimal import Decimal

def vector_norm(x):
    return Decimal(np.dot(x, x)).sqrt()


def qr_decomposition(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = vector_norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R


def rank_of_matrix(mat):
    m = len(mat)
    n = len(mat[0])

    rank = min(m, n)

    for row in range(rank):
        if mat[row][row] != 0:
            for col in range(row + 1, m):
                factor = mat[col][row] / mat[row][row]
                for i in range(row, n):
                    mat[col][i] -= factor * mat[row][i]
        else:
            reduce_rank = True
            for i in range(row + 1, m):
                if mat[i][row] != 0:
                    mat[row], mat[i] = mat[i], mat[row]
                    reduce_rank = False
                    break
            if reduce_rank:
                rank -= 1
                for i in range(row, m):
                    mat[i][row] = mat[i][rank]

    return rank


def eig(A):
    max_iter = 100
    tol = 1e-6

    m, n = A.shape
    eigvecs = np.random.randn(n, n)

    for i in range(max_iter):
        eigvecs_new = A @ eigvecs
        eigvecs_new, _ = qr_decomposition(eigvecs_new)
        if np.allclose(eigvecs_new, eigvecs, rtol=tol):
            break
        eigvecs = eigvecs_new

    eigvals = np.diag(eigvecs.T @ A @ eigvecs)

    return eigvals, eigvecs


def SVD(A):
    if A.shape[0] < A.shape[1]:
        S = A @ A.T
        k = rank_of_matrix(S.copy())
    else:
        S = A.T @ A
        k = rank_of_matrix(S.copy())

    print(f"svd of {A}:{S}")
    eigvals, eigvecs = eig(S)
    print(eigvals)
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:,sorted_indices]


    s = np.sqrt(eigvals)
    s = s[:k]
    s_inv = np.zeros_like(A.T)
    np.fill_diagonal(s_inv, 1.0 / s)

    if(A.shape[0] > A.shape[1]):
        U = np.dot(A, np.dot(eigvecs, s_inv))
        V_T = eigvecs.T
        if(len(s) != V_T.shape[0]): V_T = V_T[:len(s) - V_T.shape[0], :] 

    else:
        U = eigvecs
        V_T = np.dot(s_inv, np.dot(U.T, A))
        if(len(s) != U.shape[1]): U = U[:, :len(s) - U.shape[1]]

    sigma = np.zeros([U.shape[1], V_T.shape[0]])
    for i in range(len(s)):
        sigma[i, i] = s[i]

    return U, s, sigma, V_T


def ReducedSVD(A, threshold=0, to_remove=0):
    U, s, sigma, V_trans = SVD(A)
    print(f"reduced svd of {A}:{S}")
    if (to_remove < len(s) and to_remove > 0):
        s = s[:-to_remove]
        U = U[:, :-to_remove]
        V_trans = V_trans[:-to_remove, :]
        sigma = sigma[:-to_remove, :-to_remove]

    elif (to_remove < 0):
        print("The number of eigen values to be removed is Invalid!!")
        exit()

    if (threshold < s[0] and threshold > 0):
        s = s[s >= threshold]
        U = U[:, :len(s)]
        V_trans = V_trans[:len(s), :]
        sigma = sigma[:len(s), :len(s)]

    elif (threshold < 0):
        print("Invalid threshold value!!")
        exit()

    return U, s, sigma, V_trans


if __name__ == "__main__":
    tryy = np.array([[0.89411765, 0.84117647, 0.77058824, 0.54411765, 0.91176471,
                      0.784375, 0.7625, 0.61785714, 0.69444444, 0.76666667,
                      0.6875, 0.58181818, 0.62222222, 0.44, 0.6,
                      0.48333333, 0.45, 0.41666667, 0.48333333, 0.5,
                      0.28571429, 0.47166667, 0.52166667, 0.45, 0.72142857],
                     [0.75543947, 0.63583333, 0.56670589, 0.52787547, 0.82429796,
                      0.86477368, 0.69059243, 0.64201142, 0.6202247, 0.63628521,
                      0.65243902, 0.45059658, 0.59061035, 0.54418605, 0.61658815,
                      0.23793103, 0.24285714, 0.15517241, 0.21785714, 0.12962963,
                      0.33333333, 0.50517084, 0.43846154, 0.34615385, 0.57777778],
                     [0.81939394, 0.67545455, 0.65515152, 0.65939394, 0.83606061,
                      0.803, 0.809, 0.61724138, 0.5875, 0.675,
                      0.66666667, 0.42884615, 0.66086957, 0.6, 0.55,
                      0.19047619, 0.2, 0.225, 0.195, 0.15789474,
                      0.31666667, 0.4568, 0.455, 0.5, 0.45714286],
                     [0.77526316, 0.69210526, 0.61526316, 0.72684211, 0.9,
                      0.67692308, 0.58571429, 0.54545455, 0.53333333, 0.52222222,
                      0.75555556, 0.30111111, 0.47777778, 0.56666667, 0.75555556,
                      0.14571429, 0.11714286, 0.04857143, 0.12142857, 0.03333333,
                      0.33333333, 0.38, 0.35, 0.45, 0.52857143]])
    U, s, sigma, V_trans = SVD(tryy)
    print(U @ sigma @ V_trans)
