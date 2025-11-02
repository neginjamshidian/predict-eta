from ..utils.metrics import rmse_compat
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def train(X, y, n_estimators=200, random_state=42):
    m = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1).fit(X,y)
    p = m.predict(X)
    rmse = rmse_compat(y, p)
    mae = mean_absolute_error(y, p)
    r2 = r2_score(y, p)
    return m, {"rmse": rmse, "mae": mae, "r2": r2}
