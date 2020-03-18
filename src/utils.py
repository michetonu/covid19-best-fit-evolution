import json
from urllib.request import urlopen

import numpy as np
import scipy.optimize as opt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def get_json_from_url(url):
    """Get a json from a URL."""
    response = urlopen(url)

    return json.loads(response.read())

def scale(y):
    """Scale the data between 0 and 1 using a min-max scaler"""
    m = MinMaxScaler()
    y = m.fit_transform(y.reshape(-1, 1))
    return y.reshape(1, -1)[0]


def logistic(x, a, c, d):
    """Fit a logistic function."""
    return a / (1. + np.exp(-c * (x - d)))


def first_derivative(x, a, c, d):
    """Fit the first derivative of a logistic function"""
    return (a * c * np.exp(-c * (x - d))) / (np.exp(-c * (x - d)) + 1)**2


def fit_predict(x, y, f, x_pred=None):
    """Fit a function and predict on some input"""
    popt, pcov = opt.curve_fit(f, x, y, maxfev=100000)
    if x_pred is None:
        x_pred = x
    return f(x_pred, *popt)


def compute_derivative(df, column, x, f):
    """Compute the derivative by calculating the rate of change (per day) of some values."""
    y = np.array([float(x) for x in (df[column] - df[column].shift(1)).bfill().values])
    return fit_predict(x, y, f)


def plot(x, y, ax, title, points=None):
    """Plot some data points on a line (and optionally a scatter plot)"""
    sns.lineplot(x, y, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('')
    if points is not None:
        sns.scatterplot(x, scale(points.values), ax=ax, color='black')


def plot_projection(x, y, ax):
    """Plot the projected future line."""
    sns.lineplot(x, y, ax=ax, linewidth=1, color='red', alpha=0.5)
