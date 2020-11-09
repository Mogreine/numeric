import numpy as np
import plotly.graph_objs as go
import plotly

exp = np.exp
fact = np.math.factorial


def exp_true(x):
    return 1 / exp(x)


def exp_teilor(x, float_type=64):
    fl = np.float64
    k = 160
    if float_type == 32:
        fl = np.float32
        k = 50
    exp_series = [fl(1)]
    last = exp_series[-1]
    for i in range(1, k + 1):
        exp_series.append(last / fl(i) * fl(x))
        last = exp_series[-1]
    # exp_series = [fl(x ** i) / fl(fact(i)) for i in range(k)]
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


def plot(x32, x64, err32, err64):
    trace1 = go.Scatter(
        x=x32,
        y=err32,
        mode='lines',
        marker={'color': 'rgba(0, 255, 0, 0.8)'},
        name='float32',
        xaxis='x1',
        yaxis='y1'
    )
    trace2 = go.Scatter(
        x=x64,
        y=err64,
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
            domain=[0, 0.45],
            title='x'
        ),
        yaxis=dict(
            domain=[0, 0.65],
            showexponent='all',
            exponentformat='e',
            title='Относительная погрешность'
        ),
        xaxis2=dict(
            domain=[0.55, 1],
            title='x'
        ),
        yaxis2=dict(
            domain=[0, 0.65],
            anchor='x2',
            showexponent='all',
            exponentformat='e',
            title='Относительная погрешность'
        )
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig)


if __name__ == "__main__":
    x32 = np.linspace(0, 20, 1000)
    x64 = np.linspace(0, 80, 1000)
    y_true32 = exp_true(x32)
    y_true64 = exp_true(x64)
    y_mine32 = exp_teilor_vec(x32, 32)
    y_mine64 = exp_teilor_vec(x64, 64)

    err32 = np.abs(y_mine32 - y_true32) / y_true32
    err64 = np.abs(y_mine64 - y_true64) / y_true64

    plot(x32, x64, err32, err64)
