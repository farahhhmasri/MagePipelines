import io
import pandas as pd
import requests
import os
import opendatasets
import json


if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data_from_api(*args, **kwargs) :

    cwd = os.getcwd()
    data_dir = kwargs['configuration'].get('download_dir')

    if 'diamonds-prices' in os.listdir(f"{cwd}/{data_dir}") :
        print("Data already exists, won't download it.")
        df = pd.read_csv(f"{data_dir}/diamonds-prices/Diamonds Prices2022.csv", index_col=0)
        return df
    
    else:
        ### Adding the kaggle.json for kaggle API authentication
        dir_content  = os.listdir()
        if 'kaggle.json' not in dir_content : 
            with open("kaggle.json", "w") as file:
                data= {"username":kwargs['configuration'].get('username'),\
                "key":kwargs['configuration'].get('key')}
                json.dump(data, file)

        print("Downloading dataset...")
        opendatasets.download_kaggle_dataset(kwargs['configuration'].get('dataset_url'),\
        kwargs['configuration'].get('download_dir'))

        df = pd.read_csv(f"{kwargs['configuration'].get('download_dir')}/diamonds-prices/Diamonds Prices2022.csv", index_col=0)
        return df



@test
def test_output(output, *args) -> None:
    assert output is not None, 'The output is undefined'
