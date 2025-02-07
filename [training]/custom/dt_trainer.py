from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np


if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def transform_custom(data, data_2, *args, **kwargs):
    if "dt" not in data_2:
        # return "Decision tree model isn't being trained."
        return None


    X_train, X_validation, y_train, y_test = data

    dt = GridSearchCV(DecisionTreeRegressor(random_state=0), data_2.get("dt"), cv=5)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_validation)
    dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))

    return {"model": "dt", "rmse": dt_rmse, "trained_model": dt}

