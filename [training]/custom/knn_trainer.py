from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np


if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def transform_custom(data, data_2, *args, **kwargs):
    
    if "knn" not in data_2:
        # return "knn model isn't being trained."
        return None


    X_train, X_validation, y_train, y_test = data

    knn = GridSearchCV(KNeighborsRegressor(), data_2.get("knn"), cv=5)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_validation)
    knn_rmse = np.sqrt(mean_squared_error(y_test, knn_pred))


    return {"model": "knn", "rmse": knn_rmse, "trained_model": knn}