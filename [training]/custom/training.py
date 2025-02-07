from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
import mlflow
from datetime import date



if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test



# current_date = date.today()
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment(f"Mage training_models {current_date} Pipeline")



@custom
def transform_custom(data, data_2, *args, **kwargs):
    rmse_results = {}

    X_train = data_2[0]
    X_validation = data_2[1]
    y_train = data_2[2]
    y_test = data_2[3]

    print("Started training...")
    for model in data :
        if model == "ridge":
            ridge = GridSearchCV(Lasso(random_state=0), data.get(model), cv=5)
            ridge.fit(X_train, y_train)
            ridge_pred = ridge.predict(X_validation)
            ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
            rmse_results["ridge"] = ridge_rmse
            print("Finished training ridge.")

        if model == "lasso":
            lasso = GridSearchCV(Ridge(random_state=0), data.get(model), cv=5)
            lasso.fit(X_train, y_train)
            lasso_pred = lasso.predict(X_validation)
            lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))
            rmse_results["lasso"] = lasso_rmse
            print("Finished training lasso.")

        if model == "knn":
            knn = GridSearchCV(KNeighborsRegressor(), data.get(model), cv=5)
            knn.fit(X_train, y_train)
            knn_pred = knn.predict(X_validation)
            knn_rmse = np.sqrt(mean_squared_error(y_test, knn_pred))
            rmse_results["knn"] = knn_rmse
            print("Finished training KNN.")

        if model == "dt":
            dt = GridSearchCV(DecisionTreeRegressor(random_state=0), data.get(model), cv=5)
            dt.fit(X_train, y_train)
            dt_pred = dt.predict(X_validation)
            dt_rmse = np.sqrt(mean_squared_error(y_test,dt_pred))
            rmse_results["dt"] = dt_rmse
            print("Finished training decision tree.")

        if model =="rf":
            rf = GridSearchCV(RandomForestRegressor(random_state=0), data.get(model), cv=5)
            rf.fit(X_train, y_train)
            rf_pred = dt.predict(X_validation)
            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
            rmse_results["rf"] = rf_rmse
            print("Finished training random forest
            .")


    # with mlflow.start_run(run_name="Saving used models, their hyperparameters and RMSE scores") as run:
    #     input_example = X_train.iloc[:1]
    #     for model in data :
    #         if model == "ridge":
    #             ridge_params = ridge.get_params()
    #             mlflow.log_params(ridge_params)
    #             mlflow.sklearn.log_model(ridge, model, input_example=input_example)
    #         if model == "lasso":
    #             lasso_params = lasso.get_params()
    #             mlflow.log_params(lasso_params)
    #             mlflow.sklearn.log_model(lasso, model, input_example=input_example)
    #         if model == "knn":
    #             knn_params = knn.get_params()
    #             mlflow.log_params(knn_params)
    #             mlflow.sklearn.log_model(knn, model, input_example=input_example)
    #         if model == "dt":
    #             dt_params = dt.best_params_
    #             mlflow.log_params(dt_params)
    #             mlflow.sklearn.log_model(dt, model, input_example=input_example)
    #         if model == "rf":
    #             rfr_params = rf.best_params_
    #             mlflow.log_params(rfr_params)
    #             mlflow.sklearn.log_model(rf, model, input_example=input_example)
        
                
    #     mlflow.log_metrics(rmse_results)

    print("Finished training...")

    best_score = min(rmse_results.values()) 

    for model in rmse_results : 
        if rmse_results[model] == best_score:
            best_model = model
            break

    if best_model == "ridge":
        return ridge, best_score, rmse_results
    if best_model == "lasso":
        return lasso, best_score, rmse_results
    if best_model == "knn":
        return knn, best_score, rmse_results
    if best_model == "dt":
        return dt, best_score, rmse_results
    if best_model == "rf":
        return rf, best_score, rmse_results


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
