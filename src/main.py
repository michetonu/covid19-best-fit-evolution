import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation

from src import utils, config


def compute_all_derivatives(df, column, times_pred):
    df[f'first_dev_{column}'] = utils.compute_derivative(
        df, f'{column}_fit', times_pred, utils.first_derivative
    )

    df[f'second_dev_{column}'] = df[f'first_dev_{column}'] - df[f'first_dev_{column}'].shift()

    df[f'third_dev_{column}'] = df[f'second_dev_{column}'] - df[f'second_dev_{column}'].shift()

    return df


def run_fit_on_single_country(country, save=False, path=None):
    data = utils.get_json_from_url(config.DATA_URL)

    df = pd.DataFrame(data[country])
    date = df['date'].values[-1]

    df = df[df['confirmed'] > config.MIN_CONFIRMED_CASES]

    df = df.reset_index(drop=True)

    future_dates = list(np.linspace(0, config.MAX_DAYS_AHEAD, num=config.MAX_DAYS_AHEAD))
    df_projected = pd.DataFrame(index=future_dates)

    x = np.array([float(x) for x in range(len(df))])
    x_future = np.array([float(x) for x in range(len(df_projected))])

    y = utils.scale(df['confirmed'].values)
    df['confirmed_fit'] = utils.fit_predict(x, y, utils.logistic)
    df = compute_all_derivatives(df, 'confirmed', times_pred=x)

    df_projected['confirmed_fit'] = utils.fit_predict(x, y, utils.logistic, x_pred=x_future)
    df_projected = compute_all_derivatives(df_projected, 'confirmed', times_pred=x_future)

    y = utils.scale(df['deaths'].values)
    df['deaths_fit'] = utils.fit_predict(x, y, utils.logistic)
    df = compute_all_derivatives(df, 'deaths', times_pred=x)

    df_projected['deaths_fit'] = utils.fit_predict(x, y, utils.logistic, x_pred=x_future)
    df_projected = compute_all_derivatives(df_projected, 'deaths', times_pred=x_future)

    fig, axs = plt.subplots(2, 2, figsize=(15, 8))
    x = df.index
    x_proj = df_projected.index

    utils.plot(x, df['confirmed_fit'], ax=axs[0, 0], points=df['confirmed'], title='confirmed',
               label='best fit', scatter_label='actual')
    utils.plot(x, df['first_dev_confirmed'], ax=axs[1, 0], title="f'(x)")

    utils.plot_projection(x_proj, df_projected['confirmed_fit'], ax=axs[0, 0], label='projected')
    utils.plot_projection(x_proj, df_projected['first_dev_confirmed'], ax=axs[1, 0])

    utils.plot(x, df['deaths_fit'], ax=axs[0, 1], points=df['deaths'], title='deaths')
    utils.plot(x, df['first_dev_deaths'], ax=axs[1, 1], title="f'(x)")

    utils.plot_projection(x_proj, df_projected['deaths_fit'], ax=axs[0, 1])
    utils.plot_projection(x_proj, df_projected['first_dev_deaths'], ax=axs[1, 1])

    axs[0, 0].legend(loc=(0.8, 0.1))
    fig.autofmt_xdate()
    fig.text(0.5, 0.01, f'days since {config.MIN_CONFIRMED_CASES} confirmed cases', ha='center')
    plt.suptitle(f"Country: {country}. Last update: {date}")

    plt.show()
    if save:
        path = path or os.path.join(config.SRC_PATH, f'../examples/{country.lower()}.png')
        fig.savefig(path, bbox_inches='tight')

    plt.close()


if __name__ == "__main__":
    run_fit_on_single_country('China', save=True)
