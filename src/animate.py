"""Create an animation of the best-fit logistic curve over time."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.animation as animation
from sklearn.preprocessing import MinMaxScaler

from src import utils, config
import argparse
import matplotlib
matplotlib.use('TkAgg')

plt.style.use('dark_background')
DOTS_COLOR = 'white'

# Uncomment this for dark style
# plt.style.use('seaborn-pastel')
# DOTS_COLOR = 'black'

matplotlib.rc('font', family='sans-serif')
matplotlib.rc('font', serif='Helvetica Neue')
matplotlib.rc('text', usetex='false')


def run(country, to_plot='confirmed', save=False, path=None):
    """Create the animation.

    Parameters
    ----------
    country: str
        Country to plot.
    to_plot: str, default 'confirmed'
        'confirmed' for confirmed cases, or 'deaths' for confirmed deaths.
    save: bool, default False
        Whether to save the plot.
    path: str, default None
        Path where to save the plot (if save == False). If None, save it in 'examples/'
    """
    if to_plot not in ['confirmed', 'deaths']:
        raise ValueError("'to_plot' must be in {'confirmed', 'deaths'}")

    # Download data and get dataframe for the given country
    data = utils.get_json_from_url(config.DATA_URL)
    df = pd.DataFrame(data[country])

    # Filter the data to only include data points after a certain number of cases
    if to_plot == 'confirmed':
        min_cases = config.MIN_CONFIRMED_CASES
    else:
        min_cases = config.MIN_DEATHS
    df = df[df[to_plot] > min_cases]
    df = df.reset_index(drop=True)

    # Plot limits
    y_max = df[to_plot].max() * 2
    x_max = config.MAX_DAYS_AHEAD + len(df)

    # X axis for future dates which are not in the data)
    x_future = [float(x) for x in list(np.linspace(0, x_max, num=x_max))]

    fig = plt.figure()
    ax = plt.axes(xlim=(0, len(x_future)), ylim=(0-(y_max*0.05), y_max))
    scatter = ax.scatter([], [], s=15, color=DOTS_COLOR)
    line, = ax.plot([], [], lw=2)
    date = ax.text(x_max - x_max*0.15, y_max + y_max*0.01, '')
    count = ax.text(x_max - x_max*0.23, y_max - y_max*0.05, '')

    plt.title(f"Logistic best fit over time, {to_plot} cases\nCountry: {country}")
    plt.xlabel(f"Days since {min_cases} {to_plot} cases")
    plt.ylabel(f"# {to_plot}")

    def plot_animation():
        def init():
            """Initialize the plot for the animation."""
            line.set_data([], [])
            scatter.set_offsets(np.empty(shape=(0, 2)))
            date.set_text('')
            count.set_text('')
            return [scatter, line, date, count],

        def fit_until_index(i):
            """Fit the logistic curve using data up until the current time <i>."""
            x = np.array([float(x) for x in range(len(df))])[:i+config.MIN_POINTS]
            cases = df[to_plot].iloc[:i+config.MIN_POINTS]

            # Apply smoothing via rolling average
            cases = cases.rolling(config.ROLLING_MEAN_WINDOW, min_periods=1, center=False).mean()

            # Scale data for fitting
            m = MinMaxScaler()
            y = m.fit_transform(cases.values.reshape(-1, 1))
            y = y.reshape(1, -1)[0]

            # Fit the logistic, then apply the inverse scaling to get actual values
            # Reshaping is needed for scipy fitting
            y_pred = utils.fit_predict(x, y, utils.logistic, x_pred=x_future).reshape(-1, 1)

            return m.inverse_transform(y_pred).reshape(1, -1)[0]

        def get_date(i):
            """Get the current date."""
            return df['date'].values[i+config.MIN_POINTS-1]

        def get_count(i):
            """Get the current case counts."""
            return df[to_plot].values[i+config.MIN_POINTS-1]

        def get_scatter_values(i):
            """Get actual values to plot as scatter points."""
            x = np.array([float(x) for x in range(len(df))])[:i+config.MIN_POINTS]

            y = df[to_plot].iloc[:i + config.MIN_POINTS]

            return x, y

        def animate(i):
            """Update the animation."""
            # Update scatter of actual values
            x_s, y_s = get_scatter_values(i)
            scatter_values = np.column_stack((x_s, y_s))
            scatter.set_offsets(scatter_values)

            # Update best fit line
            y = fit_until_index(i)
            line.set_data(x_future, y)
            line.label = i

            # Update texts
            date.set_text(get_date(i))
            count.set_text(f"# cases: {get_count(i)}")

            return [scatter, line, date, count],

        fig.tight_layout()

        return animation.FuncAnimation(fig, animate,
                                       init_func=init,
                                       frames=len(df)+1-config.MIN_POINTS,
                                       interval=500,
                                       repeat=True, repeat_delay=2)

    anim = plot_animation()
    if save:
        path = path or os.path.join(config.SRC_PATH, f'../examples/{country.lower()}_animated.gif')
        anim.save(path, writer='imagemagick', fps=1.5)

    plt.show()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Shows Fitting')
    parser.add_argument('--country', default='Italy', choices=['Italy'])
    parser.add_argument('--to-plot', default='confirmed', choices=['deaths', 'confirmed'])

    args = parser.parse_args()

    run(args.country, to_plot=args.to_plot, save=True)
