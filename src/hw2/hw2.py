import numpy as np


def numpy_solve(A, b):
    return np.linalg.solve(A, b)


def thomas_solve(A, b):
    n = len(b)
    c_i = A[0][0]
    a_i, b_i = -A[1][0], -A[0][1]
    alpha = [0]
    betta = [0]

    # getting alpha and betta
    for i in range(n):
        alpha.append(b_i / (c_i - alpha[-1] * a_i))
        betta.append((b[i] + betta[-1] * a_i) / (c_i - alpha[-2] * a_i))
    alpha.pop(0)
    betta.pop(0)

    # getting answer
    x = [betta[-1]]
    for i in range(n - 2, -1, -1):
        x.append(x[-1] * alpha[i] + betta[i])
    x.reverse()
    return np.array(x)


def func(x):
    return -(np.exp(2) * x - x + np.exp(1 - x) - np.exp(x + 1)) / (1 - np.exp(2))


def is_correct(x, u, tol=1e-5):
    u_true = func(x)
    return np.allclose(u, u_true, atol=tol)


if __name__ == "__main__":
    n = int(1e3)
    h = 1 / (n + 1)
    b = np.arange(1, n + 1) * h
    A = np.diagflat([2 * h ** (-2) + 1 for i in range(1, n + 1)])\
        + np.diagflat([-h ** (-2) for i in range(1, n)], k=-1)\
        + np.diagflat([-h ** (-2) for i in range(1, n)], k=1)

    sol_true = func(b)
    sol_np = numpy_solve(A, b)
    sol_thomas = thomas_solve(A, b)

    print(f'numpy solve correct: {is_correct(b, sol_np)}')
    print(f'diff numpy: {np.linalg.norm(sol_true - sol_np)}')
    print(f'diff thomas: {np.linalg.norm(sol_true - sol_thomas)}')
