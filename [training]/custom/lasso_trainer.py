from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np


if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def transform_custom(data, data_2, *args, **kwargs):
    
    if "lasso" not in data_2:
        # return "lasso model isn't being trained."
        return None


    X_train, X_validation, y_train, y_test = data

    lasso = GridSearchCV(Lasso(random_state=0), data_2.get("lasso"), cv=5)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_validation)
    lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))

    return {"model": "lasso", "rmse": lasso_rmse, "trained_model": lasso}