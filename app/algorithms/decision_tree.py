from ..utils.metrics import rmse_compat
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def train(X, y, random_state: int = 42):
    m = DecisionTreeRegressor(random_state=random_state).fit(X,y)
    p = m.predict(X)
    rmse = rmse_compat(y, p)
    mae = mean_absolute_error(y, p)
    from sklearn.metrics import r2_score
    r2 = r2_score(y, p)
    return m, {"rmse": rmse, "mae": mae, "r2": r2}
