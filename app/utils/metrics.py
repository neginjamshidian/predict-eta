from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_squared_error
def rmse_compat(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
