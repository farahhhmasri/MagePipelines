@custom
def select_best_model(data, data_2, data_3, data_4, data_5, *args, **kwargs):
    # Get all model results from kwargs.values()
    search_space = [data, data_2, data_3, data_4, data_5]
    models = [m for m in search_space if type(m) is not list]

    # Find the model with the lowest RMSE
    best_model = min(models, key=lambda x: x["rmse"])

    print(f"Best model: {best_model['model']} with RMSE: {best_model['rmse']}")

    # Return the best trained model, print model's RMSE, and all model RMSEs
    print(f"Best model's performance: {best_model['rmse']}")
    print(f"Other model's perfromance: ", {m["model"]: m["rmse"] for m in models})
    return best_model["trained_model"]
