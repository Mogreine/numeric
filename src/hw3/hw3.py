import numpy as np
from scipy import sparse
from scipy.sparse.linalg import inv
from timeit import default_timer as timer
import plotly.graph_objs as go
import plotly


def u_val(x, y):
    return np.sin(np.pi * x) * np.sin(2 * np.pi * y)


def f_val(x, y):
    return np.sin(np.pi * x) * np.sin(2 * np.pi * y) * (5 * np.pi ** 2 + np.exp(x * y))


def generate_F(x, y):
    res = []
    for yi in y:
        res.append(f_val(x, yi))
    return np.array(res)


def generate_A(x, y, h, n):
    res = np.zeros(shape=(n ** 2, n ** 2))
    for i in range(n):
        for j in range(n):
            k = i + j * n

            res[k][k] = 4 / h ** 2 + np.exp(x[i] * y[j])

            val = -1 / h ** 2
            # if |i - i'| = 1, j = j'
            if i + 1 < n:
                res[k][k + 1] = val
            if i - 1 < n:
                res[k][k - 1] = val

            # if |j - j'| = 1, i = i'
            if j + 1 < n:
                res[k][k + n] = val
            if j - 1 < n:
                res[k][k - n] = val
    return res


def SOR_matrix(A, f, w=1.5, max_iter=100):
    D = sparse.csr_matrix(np.diagflat(np.diagonal(A)))
    L = sparse.csr_matrix(np.tril(A))
    U = sparse.csr_matrix(np.triu(A))

    x = np.zeros_like(f)
    for i in range(max_iter):
        x = inv(D + w * L) @ (w * f - (w * U + (w - 1) @ D) @ x)
    return x


def test(n):
    h = 1 / (n + 1)
    ix = [h * i for i in range(n + 1)]
    iy = [h * i for i in range(n + 1)]

    f = generate_F(ix, iy)
    A = generate_A(ix, iy, h, n)

    u_true = u_val(ix, iy)

    print(np.linalg.norm())