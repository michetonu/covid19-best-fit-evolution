import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.animation as animation
from sklearn.preprocessing import MinMaxScaler

from src import utils, config

import matplotlib
matplotlib.use('TkAgg')

MIN_POINTS = 5
ROLLING_MEAN_WINDOW = 3

plt.style.use('seaborn-pastel')


def run(country, save=False, path=None):
    data = utils.get_json_from_url(config.DATA_URL)
    df = pd.DataFrame(data[country])
    df = df[df['confirmed'] > config.MIN_CONFIRMED_CASES]
    df = df.reset_index(drop=True)

    x_future = [float(x) for x in list(np.linspace(0, config.MAX_DAYS_AHEAD,
                                                   num=config.MAX_DAYS_AHEAD))]

    fig = plt.figure()
    ax = plt.axes(xlim=(0, len(x_future)), ylim=(0, 100000))
    line, = ax.plot([], [], lw=2)
    scatter = ax.scatter([], [], s=3, color='black')
    date = ax.text(85, 101000, '')
    count = ax.text(78, 95000, '')

    plt.title("Evolving logistic best fit, confirmed cases\nCounty: Italy")

    def plot_animation():
        def init():
            line.set_data([], [])
            return [line, scatter, date, count],

        def run_until_index(i):
            x = np.array([float(x) for x in range(len(df))])[:i+MIN_POINTS]

            m = MinMaxScaler()
            confirmed = df['confirmed'].iloc[:i+MIN_POINTS]
            confirmed = confirmed.rolling(ROLLING_MEAN_WINDOW, min_periods=1, center=True).mean()

            y = m.fit_transform(confirmed.values.reshape(-1, 1))
            y = y.reshape(1, -1)[0]
            y_pred = utils.fit_predict(x, y, utils.logistic, x_pred=x_future)

            return m.inverse_transform(y_pred.reshape(-1, 1)).reshape(1, -1)[0]

        def get_date(i):
            return df['date'].values[i+MIN_POINTS-1]

        def get_count(i):
            return df['confirmed'].values[i+MIN_POINTS-1]

        def get_scatter_values(i):
            x = np.array([float(x) for x in range(len(df))])[:i+MIN_POINTS]

            y = df['confirmed'].iloc[:i + MIN_POINTS]

            return x, y

        def animate(i):
            y = run_until_index(i)
            line.set_data(x_future, y)
            line.label = i

            x_s, y_s = get_scatter_values(i)

            scatter_values = np.column_stack((x_s, y_s))
            scatter.set_offsets(scatter_values)

            date.set_text(get_date(i))
            count.set_text(f"# cases: {get_count(i)}")
            return [line, scatter, date, count],

        return animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=len(df)-MIN_POINTS, interval=1000, repeat=True)

    anim = plot_animation()
    if save:
        path = path or os.path.join(config.SRC_PATH, f'../examples/{country.lower()}_animated.gif')
        anim.save(path, writer='imagemagick')

    plt.show()
    plt.close()


if __name__ == "__main__":
    run("Italy", save=True)
