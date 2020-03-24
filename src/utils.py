import json
from urllib.request import urlopen

import numpy as np
import scipy.optimize as opt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def get_json_from_url(url):
    """Get a json from a URL."""
    response = urlopen(url)

    return json.loads(response.read().decode())


def logistic(x, a, c, d):
    """Fit a logistic function."""
    return a / (1. + np.exp(-c * (x - d)))


def fit_predict(x, y, f, x_pred=None):
    """Fit a function and predict on some input"""
    popt, pcov = opt.curve_fit(f, x, y, maxfev=100000)
    if x_pred is None:
        x_pred = x
    return f(x_pred, *popt)
