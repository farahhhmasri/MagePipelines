if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(*args, **kwargs) :

    ### select which model to train, if none is selected all available models will be trained
    ## return a list of models names

    option  = ["lasso", "ridge", "dt", "knn", "rf"]

    option = []

    return option

