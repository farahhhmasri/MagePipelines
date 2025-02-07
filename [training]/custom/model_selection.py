import ast


if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def transform_custom(*args, **kwargs):
    
    data = {
        "ridge" : {
            "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        },
        "lasso": {
            "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        },
        "knn": {
            "n_neighbors": [3, 5, 7, 9, 11, 13],
            "weights": ["uniform", "distance"],
            "p": [1, 2]  
        },
        "dt": {
            "max_depth": [3, 5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5],
            "max_features": ["sqrt", "log2", None]
        },
        "rf": {
            "n_estimators": [10, 20, 50],
            "max_depth": [3, 5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False]
        }

    }

    selected_models = kwargs.get("selected_models", ["ridge", "lasso", "knn", "dt", "rf"])
    
    # Convert string representation of a list to an actual list
    if isinstance(selected_models, str):
        try:
            selected_models = ast.literal_eval(selected_models)
        except (SyntaxError, ValueError):
            raise ValueError(f"Invalid format for selected_models: {selected_models}. Expected a list.")

    # Ensure it's a list
    if not isinstance(selected_models, list):
        raise TypeError(f"Expected selected_models to be a list, got {type(selected_models)} instead.")


    filtered_models = {model: data[model] for model in selected_models if model in data}

    print(f"Selected models for training: {list(filtered_models.keys())}")

    return filtered_models



@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
