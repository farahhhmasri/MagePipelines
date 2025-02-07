import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def prediction(data, data_2, *args, **kwargs):

    model = data['final_model'][0].best_estimator_
    
    if not isinstance(data_2, pd.DataFrame):
        return "No data was provided for predictions!"
    
    else: 
        print("Generating predections for the provided data...")
        ids = data_2.pop('id')
        preds = model.predict(data_2)
        predictions = pd.DataFrame({"id":ids, "predictions":preds})
        return predictions


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
