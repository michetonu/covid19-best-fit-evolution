import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


def scale(y):
    m = MinMaxScaler()
    y = m.fit_transform(y.reshape(-1, 1))
    return y.reshape(1, -1)[0]


def logistic(x, a, c, d):
    return a / (1. + np.exp(-c * (x - d)))


def first_derivative(x, a, c, d):
    return (a * c * np.exp(-c * (x - d))) / (np.exp(-c * (x - d)) + 1)**2


def second_derivative(x, a, c, d):
    return (
            a *
            (((2 * c**2 * np.exp(-2 * c * (x - d))) / (np.exp(-c * (x - d)) + 1)**3)
                - ((c**2 * np.exp(-c * (x - d))) / (np.exp(-c * (x - d)) + 1)**2))
    )


def third_derivative(x, a, c, d):
    return (
            a
            * (2 * c**2 * np.exp(-2*c * (x - d))) / (np.exp(-c * (x - d)) + 1)**3
            * -(c**2 * np.exp(-c * (x - d))) / (np.exp(-c * (x - d)) + 1)**2
    )


def fit_predict(x, y, f, x_pred=None):
    popt, pcov = opt.curve_fit(f, x, y, maxfev=100000)
    if x_pred is None:
        x_pred = x
    return f(x_pred, *popt)


def compute_derivative(df, column, x, f, x_pred=None):
    y = np.array([float(x) for x in (df[column] - df[column].shift(1)).fillna(0).values])
    y = scale(y)
    return fit_predict(x, y, f, x_pred)


def plot(x, y, ax, title, points=None):
    sns.lineplot(x, y, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('')
    if points is not None:
        sns.scatterplot(x, scale(points.values), ax=ax, color='black')


def plot_projection(x, y, ax):
    sns.lineplot(x, y, ax=ax, linewidth=1, color='red', alpha=0.5)


if __name__ == "__main__":
    max_days = 100
    min_confirmed = 40

    country = 'Italy'
    data_path = "/Users/michele/Documents/covid19/docs/timeseries.json"

    with open(data_path, 'r') as ff:
        data = json.load(ff)
    # print(data.keys())
    df = pd.DataFrame(data[country])

    df = df[df['confirmed'] > min_confirmed]
    df = df.reset_index(drop=True)

    future_dates = list(np.linspace(0, max_days, num=max_days))
    df_projected = pd.DataFrame(index=future_dates)

    x = np.array([float(x) for x in range(len(df))])
    x_future = np.array([float(x) for x in range(len(df_projected))])

    y = scale(df['confirmed'].values)
    df['confirmed_fit'] = fit_predict(x, y, logistic)
    df['first_dev_confirmed'] = compute_derivative(df, 'confirmed_fit', x, first_derivative)
    df['second_dev_confirmed'] = compute_derivative(df, 'first_dev_confirmed', x, second_derivative)
    df['third_dev_confirmed'] = compute_derivative(df, 'second_dev_confirmed', x, second_derivative)

    df_projected['confirmed_fit'] = fit_predict(x, y, logistic, x_pred=x_future)
    df_projected['first_dev_confirmed'] = compute_derivative(df, 'confirmed_fit', x,
                                                             first_derivative, x_pred=x_future)
    df_projected['second_dev_confirmed'] = compute_derivative(df, 'first_dev_confirmed',
                                                              x, second_derivative, x_pred=x_future)
    df_projected['third_dev_confirmed'] = compute_derivative(df, 'second_dev_confirmed',
                                                             x, second_derivative, x_pred=x_future)

    y = scale(df['deaths'].values)
    df['deaths_fit'] = fit_predict(x, y, logistic)
    df['first_dev_deaths'] = compute_derivative(df, 'deaths_fit', x, first_derivative)
    df['second_dev_deaths'] = compute_derivative(df, 'first_dev_deaths', x, second_derivative)
    df['third_dev_deaths'] = compute_derivative(df, 'second_dev_deaths', x, third_derivative)

    df_projected['deaths_fit'] = fit_predict(x, y, logistic, x_pred=x_future)
    df_projected['first_dev_deaths'] = compute_derivative(df_projected, 'deaths_fit', x_future,
                                                          first_derivative)
    df_projected['second_dev_deaths'] = compute_derivative(df_projected, 'first_dev_deaths',
                                                           x_future, second_derivative)
    df_projected['third_dev_deaths'] = compute_derivative(df_projected, 'second_dev_deaths',
                                                          x_future, third_derivative)

    fig, axs = plt.subplots(4, 2, figsize=(15, 8))
    date = df.index
    date_proj = df_projected.index

    plot(date, df['confirmed_fit'], ax=axs[0, 0], points=df['confirmed'], title='confirmed')
    plot(date, df['first_dev_confirmed'], ax=axs[1, 0], title='first_dev_confirmed')
    plot(date, df['second_dev_confirmed'], ax=axs[2, 0], title='second_dev_confirmed')
    plot(date, df['third_dev_confirmed'], ax=axs[3, 0], title='third_dev_confirmed')

    plot_projection(date_proj, df_projected['confirmed_fit'], ax=axs[0, 0])
    plot_projection(date_proj, df_projected['first_dev_confirmed'], ax=axs[1, 0])
    plot_projection(date_proj, df_projected['second_dev_confirmed'], ax=axs[2, 0])
    plot_projection(date_proj, df_projected['third_dev_confirmed'], ax=axs[3, 0])

    plot(date, df['deaths_fit'], ax=axs[0, 1], points=df['deaths'], title='deaths')
    plot(date, df['first_dev_deaths'], ax=axs[1, 1], title='first_dev_deaths')
    plot(date, df['second_dev_deaths'], ax=axs[2, 1], title='second_dev_deaths')
    plot(date, df['third_dev_deaths'], ax=axs[3, 1], title='third_dev_deaths')

    plot_projection(date_proj, df_projected['deaths_fit'], ax=axs[0, 1])
    plot_projection(date_proj, df_projected['first_dev_deaths'], ax=axs[1, 1])
    plot_projection(date_proj, df_projected['second_dev_deaths'], ax=axs[2, 1])
    plot_projection(date_proj, df_projected['third_dev_deaths'], ax=axs[3, 1])

    fig.autofmt_xdate()
    fig.text(0.5, 0.04, f'days since {min_confirmed} confirmed cases', ha='center')
    plt.suptitle(country)

    plt.show()
