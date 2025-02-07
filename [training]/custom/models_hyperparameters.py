from typing import Optional, List

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

### you can select which model to train by specifying it
### as a global variable 
### options are: all, lasso, ridge, knn, DecisionTree, RandomForest


@custom
def transform_custom(option:Optional[List[str]]=None, *args, **kwargs):

    available_models = {
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
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [3, 5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False]
        }

    }

    if option == []:
        return available_models

    if option != []:
        search_space = {}
        for i in option :
            search_space[i] = available_models.get(i)
        return search_space
    


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
