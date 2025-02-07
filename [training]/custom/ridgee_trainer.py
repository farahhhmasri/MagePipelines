from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np


if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def transform_custom(data, data_2, *args, **kwargs):
    if "ridge" not in data_2:
        # return "Ridge model isn't being trained."
        return None


    X_train, X_validation, y_train, y_test = data

    ridge = GridSearchCV(Ridge(random_state=0), data_2.get("ridge"), cv=5)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_validation)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))

    return {"model": "ridge", "rmse": ridge_rmse, "trained_model": ridge}
