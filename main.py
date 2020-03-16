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


def fit_predict(x, y, f, x_pred=None):
    popt, pcov = opt.curve_fit(f, x, y, maxfev=100000)
    if x_pred is None:
        x_pred = x
    return f(x_pred, *popt)


def compute_derivative(df, column, x, f):
    y = np.array([float(x) for x in (df[column] - df[column].shift(1)).bfill().values])
    return fit_predict(x, y, f)


def plot(x, y, ax, title, points=None):
    sns.lineplot(x, y, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('')
    if points is not None:
        sns.scatterplot(x, scale(points.values), ax=ax, color='black')


def plot_projection(x, y, ax):
    sns.lineplot(x, y, ax=ax, linewidth=1, color='red', alpha=0.5)


def compute_all_derivatives(df, column, times_pred):
    df[f'first_dev_{column}'] = compute_derivative(df, f'{column}_fit', times_pred,
                                                   first_derivative)
    df[f'second_dev_{column}'] = (
            df[f'first_dev_{column}'] - df[f'first_dev_{column}'].shift()
    )

    df[f'third_dev_{column}'] = (
            df[f'second_dev_{column}'] - df[f'second_dev_{column}'].shift()
    )
    return df


if __name__ == "__main__":
    max_days = 100
    min_confirmed = 40

    country = 'Korea, South'
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
    df = compute_all_derivatives(df, 'confirmed', times_pred=x)

    df_projected['confirmed_fit'] = fit_predict(x, y, logistic, x_pred=x_future)
    df_projected = compute_all_derivatives(df_projected, 'confirmed', times_pred=x_future)

    y = scale(df['deaths'].values)
    df['deaths_fit'] = fit_predict(x, y, logistic)
    df = compute_all_derivatives(df, 'deaths', times_pred=x)

    df_projected['deaths_fit'] = fit_predict(x, y, logistic, x_pred=x_future)
    df_projected = compute_all_derivatives(df_projected, 'deaths', times_pred=x_future)

    fig, axs = plt.subplots(4, 2, figsize=(15, 8))
    date = df.index
    date_proj = df_projected.index

    plot(date, df['confirmed_fit'], ax=axs[0, 0], points=df['confirmed'], title='confirmed')
    plot(date, df['first_dev_confirmed'], ax=axs[1, 0], title="f'(x)")
    plot(date, df['second_dev_confirmed'], ax=axs[2, 0], title="f''(x)")
    plot(date, df['third_dev_confirmed'], ax=axs[3, 0], title="f'''(x)")

    plot_projection(date_proj, df_projected['confirmed_fit'], ax=axs[0, 0])
    plot_projection(date_proj, df_projected['first_dev_confirmed'], ax=axs[1, 0])
    plot_projection(date_proj, df_projected['second_dev_confirmed'], ax=axs[2, 0])
    plot_projection(date_proj, df_projected['third_dev_confirmed'], ax=axs[3, 0])

    plot(date, df['deaths_fit'], ax=axs[0, 1], points=df['deaths'], title='deaths')
    plot(date, df['first_dev_deaths'], ax=axs[1, 1], title="f'(x)")
    plot(date, df['second_dev_deaths'], ax=axs[2, 1], title="f''(x)")
    plot(date, df['third_dev_deaths'], ax=axs[3, 1], title="f'''(x)")

    plot_projection(date_proj, df_projected['deaths_fit'], ax=axs[0, 1])
    plot_projection(date_proj, df_projected['first_dev_deaths'], ax=axs[1, 1])
    plot_projection(date_proj, df_projected['second_dev_deaths'], ax=axs[2, 1])
    plot_projection(date_proj, df_projected['third_dev_deaths'], ax=axs[3, 1])

    fig.autofmt_xdate()
    fig.text(0.5, 0.01, f'days since {min_confirmed} confirmed cases', ha='center')
    plt.suptitle(f"Country: {country}")

    plt.show()
