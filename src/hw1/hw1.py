import numpy as np
import plotly.graph_objs as go
import plotly

exp = np.exp
fact = np.math.factorial


def exp_true(x):
    return 1 / exp(x)


def exp_teilor(x, float_type=64):
    fl = np.float64
    k = 60
    if float_type == 32:
        fl = np.float32
        k = 30
    exp_series = [fl(x ** i) / fl(fact(i)) for i in range(k)]
    exp_series.sort()
    res = fl(0)
    for i in exp_series:
        res += i
    return fl(1) / res


def exp_teilor_vec(x_arr, float_type=64):
    res = []
    for x in x_arr:
        res.append(exp_teilor(x, float_type))
    return np.array(res)


def plot(x, *errs):
    trace1 = go.Scatter(
        x=x,
        y=errs[0],
        mode='lines',
        marker={'color': 'rgba(0, 255, 0, 0.8)'},
        name='float32',
        xaxis='x1',
        yaxis='y1'
    )
    trace2 = go.Scatter(
        x=x,
        y=errs[1],
        mode='lines',
        marker=dict(color='rgba(255, 0, 0, 0.8)'),
        name='float64',
        xaxis='x2',
        yaxis='y2'
    )
    data = [trace1, trace2]

    layout = go.Layout(
        title='Зависимость относительной ошибки от x',
        xaxis=dict(
            domain=[0, 0.45]
        ),
        yaxis=dict(
            domain=[0, 0.65]
        ),
        xaxis2=dict(
            domain=[0.55, 1]
        ),
        yaxis2=dict(
            domain=[0, 0.65],
            anchor='x2'
        )
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig)


if __name__ == "__main__":
    x = np.linspace(0, 20, 500)
    y_true = exp_true(x)
    y_mine32 = exp_teilor_vec(x, 32)
    y_mine64 = exp_teilor_vec(x, 64)

    err32 = np.abs(y_mine32 - y_true) / y_true
    err64 = np.abs(y_mine64 - y_true) / y_true

    plot(x, err32, err64)
