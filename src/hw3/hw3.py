import numpy as np
from scipy import sparse
from scipy.sparse.linalg import inv
from timeit import default_timer as timer
import plotly.graph_objs as go
import plotly


def u_val(x, y):
    return np.sin(np.pi * x) * np.sin(2 * np.pi * y)


def f_val(x, y):
    return u_val(x, y) * (5 * np.pi ** 2 + np.exp(x * y))


def generate_F(x, y, func):
    res = []
    for yi in y:
        res.append(func(x, yi))
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
            if i - 1 >= 0:
                res[k][k - 1] = val

            # if |j - j'| = 1, i = i'
            if j + 1 < n:
                res[k][k + n] = val
            if j - 1 >= 0:
                res[k][k - n] = val
    return res


def SOR_matrix(A, f, w=1.5, max_iter=100):
    D = sparse.csr_matrix(np.diagflat(np.diagonal(A)))
    L = sparse.csr_matrix(np.tril(A, -1))
    U = sparse.csr_matrix(np.triu(A, 1))

    # D = np.diagflat(np.diagonal(A))
    # L = np.tril(A, -1)
    # U = np.triu(A, 1)

    x = np.zeros_like(f)
    for i in range(max_iter):
        x = inv(D + w * L) @ (w * f - (w * U + (w - 1) * D) @ x)
    return x


def SOR_dep(A, f, w=1.0, max_iter=100):
    x = np.zeros_like(f)
    hist = []
    kk = []
    for k in range(max_iter):
        for i in range(len(x)):
            x[i] = (1 - w) * x[i] + w / A[i][i] * (f[i] - A[i, :i] @ x[:i] - A[i, i + 1:] @ x[i + 1:])
        hist.append(x.copy())
        kk.append(k + 1)
    return x, kk, hist


def SOR(A, f, w=1.0, max_iter=1000, tol=1e-8):
    x = np.zeros_like(f)
    hist = []
    kk = []
    prev = x.copy()
    for k in range(max_iter):
        for i in range(len(x)):
            x[i] = (1 - w) * x[i] + w / A[i][i] * (f[i] - A[i, :i] @ x[:i] - A[i, i + 1:] @ x[i + 1:])
        hist.append(x.copy())
        kk.append(k + 1)

        if np.linalg.norm(x - prev) <= tol:
            break
        prev = x.copy()
    return x, kk, hist


def gen_color(n):
    res = []
    for i in range(n):
        res.append({'color': f'rgba({255 / n * (i + 1)},'
                             f' 0,'
                             f' 0,'
                             f' 0.8)'
                    })
    return res


def plot(data):
    traces = []
    colors = gen_color(len(data))
    for (w, (x, y)), c in zip(data.items(), colors):
        traces.append(
            go.Scatter(
                x=x,
                y=y,
                mode='lines+markers',
                marker=c,
                name=str(w)
            )
        )

    layout = go.Layout(
        title='Зависимость точности от количества итераций',
        xaxis={
            'title': 'k'
        },
        yaxis={
            'title': '$log_{10}|F(w_i) - F(w^*)|$'
        }
    )

    fig = go.Figure(data=traces, layout=layout)
    plotly.offline.iplot(fig)


def test(n, w_arr=(1, 1.2, 1.4, 1.6, 1.8, 1.9), tol=1e-3, max_iter=1000):
    h = 1 / (n + 1)
    ix = np.array([h * (i + 1) for i in range(n)])
    iy = np.array([h * (i + 1) for i in range(n)])

    f = generate_F(ix, iy, f_val).flatten()
    A = generate_A(ix, iy, h, n)

    u_true = generate_F(ix, iy, u_val).flatten()
    data = {}
    for w in w_arr:
        u_pred, x, y = SOR(A, f, w=w, max_iter=max_iter, tol=tol)
        data[w] = (x, np.log10(np.linalg.norm(np.absolute(y - u_true), ord=np.inf, axis=1)))
        # print(np.linalg.norm(np.abs(u_true - u_pred), ord=np.inf))
    plot(data)


def test2(w, n_arr, tol=1e-3, max_iter=1000):
    data = {}
    x = []
    y = []
    for n in n_arr:
        h = 1 / (n + 1)
        ix = np.array([h * (i + 1) for i in range(n)])
        iy = np.array([h * (i + 1) for i in range(n)])

        f = generate_F(ix, iy, f_val).flatten()
        A = generate_A(ix, iy, h, n)

        u_true = generate_F(ix, iy, u_val).flatten()

        u_pred, _, _ = SOR(A, f, w=w, tol=tol, max_iter=max_iter)
        y.append(np.log10(np.linalg.norm(np.absolute(u_pred - u_true), ord=np.inf)))
        x.append(n)
    data[w] = (x, y)
    plot(data)


# test(40)
# test2(w=1.9, n_arr=[10, 20, 30, 50, 70], alg=SOR2)
test(70, [1.9], tol=1e-4)
