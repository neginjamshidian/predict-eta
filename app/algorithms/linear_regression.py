from ..utils.metrics import rmse_compat
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def train(X, y):
    m = LinearRegression().fit(X,y)
    p = m.predict(X)
    rmse = rmse_compat(y, p)
    mae = mean_absolute_error(y, p)
    r2 = r2_score(y, p)
    return m, {"rmse": rmse, "mae": mae, "r2": r2}
