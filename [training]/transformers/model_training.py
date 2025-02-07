from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import mlflow

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, data_2):
    
    rmse_results = {}

    X_train = data_2[0]
    X_validation = data_2[1]
    y_train = data_2[2]
    y_test = data_2[3]

    for model in data :
        if model == "ridge":
            ridge = GridSearchCV(Lasso(random_state=0), data.get(model), cv=5)
            ridge.fit(X_train, y_train)
            ridge_pred = ridge.predict(X_validation)
            ridge_rmse = np.sqrt(mean_squared_error(y_validation,ridge_pred))
            rmse_results["ridge"] = ridge_rmse

        if model == "lasso":
            lasso = GridSearchCV(Ridge(random_state=0), data.get(model), cv=5)
            lasso.fit(X_train, y_train)
            lasso_pred = lasso.predict(X_validation)
            lasso_rmse = np.sqrt(mean_squared_error(y_validation,lasso_pred))
            rmse_results["lasso"] = lasso_rmse

        if model == "knn":
            knn = GridSearchCV(KNeighborsRegressor(), data.get(model), cv=5)
            knn.fit(X_train, y_train)
            knn_pred = knn.predict(X_validation)
            knn_rmse = np.sqrt(mean_squared_error(y_validation,knn_pred))
            rmse_results["knn"] = knn_rmse

        if model == "dt":
            dt = GridSearchCV(DecisionTreeRegressor(random_state=0), data.get(model), cv=5)
            dt.fit(X_train, y_train)
            dt_pred = dt.predict(X_validation)
            dt_rmse = np.sqrt(mean_squared_error(y_validation,dt_pred))
            rmse_results["dt"] = dt_rmse

        if model =="rf":
            rf = GridSearchCV(RandomForestRegressor(random_state=0), data.get(model), cv=5)
            rf.fit(X_train, y_train)
            rf_pred = dt.predict(X_validation)
            rf_rmse = np.sqrt(mean_squared_error(y_validation,rf_pred))
            rmse_results["rf"] = rf_rmse


    return rmse_results


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
