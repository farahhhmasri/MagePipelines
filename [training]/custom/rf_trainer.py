from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np


if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def transform_custom(data, data_2, *args, **kwargs):
    
    if "rf" not in data_2:
        # return "Random forest model isn't being trained."
        return None


    X_train, X_validation, y_train, y_test = data

    rf = GridSearchCV(RandomForestRegressor(random_state=0), data_2.get("rf"), cv=5)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_validation)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))


    return {"model": "rf", "rmse": rf_rmse, "trained_model": rf}