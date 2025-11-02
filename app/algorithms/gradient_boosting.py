from ..utils.metrics import rmse_compat
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def train(X, y):
    m = GradientBoostingRegressor(random_state=42).fit(X,y)
    p = m.predict(X)
    rmse = rmse_compat(y, p)
    mae = mean_absolute_error(y, p)
    r2 = r2_score(y, p)
    return m, {"rmse": rmse, "mae": mae, "r2": r2}, m.get_params()
