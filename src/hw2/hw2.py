import numpy as np
from timeit import default_timer as timer
import plotly.graph_objs as go
import plotly


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


def get_accuracy(x, u):
    u_true = func(x)
    tol = 1e-1
    while np.allclose(u, u_true, atol=tol) and tol > 1e-17:
        tol /= 10
    return tol * 10


def generate_sys(n=1000):
    h = 1 / (n + 1)
    b = np.arange(1, n + 1) * h
    A = np.diagflat([2 * h ** (-2) + 1 for i in range(1, n + 1)]) \
        + np.diagflat([-h ** (-2) for i in range(1, n)], k=-1) \
        + np.diagflat([-h ** (-2) for i in range(1, n)], k=1)
    return A, b


def test_time(method, l=10, r=1000, step=1):
    sz = []
    elapsed_time = []
    for i in range(l, r + 1, step):
        A, b = generate_sys(i)
        start_time = timer()
        method(A, b)
        elapsed_time.append(timer() - start_time)
        sz.append(i)
    return sz, elapsed_time


def plot(x, y, names):
    traces = []
    method_colors = {
        'thomas': {'color': 'rgba(0, 255, 0, 0.8)'},
        'gauss': {'color': 'rgba(255, 0, 0, 0.8)'}
    }
    for i in range(len(names)):
        traces.append(
            go.Scatter(
                x=x[i],
                y=y[i],
                mode='lines+markers',
                marker=method_colors[names[i]],
                name=names[i]
            )
        )

    layout = go.Layout(
        title='Зависимость ремени работы от n',
        xaxis={
            'title': 'n'
        },
        yaxis={
            'title': 'time, ms'
        }
    )

    fig = go.Figure(data=traces, layout=layout)
    plotly.offline.iplot(fig)


if __name__ == "__main__":
    r = 1500
    step = 10
    x_g, y_g = test_time(numpy_solve, l=10, r=r, step=step)
    x_t, y_t = test_time(thomas_solve, l=10, r=r, step=step)
    plot([x_g, x_t], [y_g, y_t], names=['gauss', 'thomas'])
