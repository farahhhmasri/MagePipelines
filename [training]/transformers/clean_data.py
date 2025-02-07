import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple
from sklearn.preprocessing import OrdinalEncoder



if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


def removenull(df:pd.DataFrame) -> pd.DataFrame:
    
    ### Removing NaN
    df.dropna(inplace=True)

    ### Removing 0 values
    df = df[(df != 0).all(axis=1)]

    return df

def encodeCats(df:pd.DataFrame) -> pd.DataFrame:
    
    encoder_cut = OrdinalEncoder(categories=[["Fair","Good","Very Good","Premium","Ideal"]], handle_unknown='use_encoded_value', unknown_value = 100).fit(df[["cut"]])
    encoded_cut = encoder_cut.transform(df[["cut"]])

    encoder_color = OrdinalEncoder(categories=[["J","I","H","G","F","E","D"]],  handle_unknown='use_encoded_value', unknown_value = 100).fit(df[["color"]])
    encoded_color = encoder_color.transform(df[["color"]])

    encoder_clarity = OrdinalEncoder(categories=[["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]],  handle_unknown='use_encoded_value', unknown_value = 100).fit(df[["clarity"]])
    encoded_clarity = encoder_clarity.transform(df[["clarity"]])

    df["cut"] = encoded_cut
    df["color"] = encoded_color
    df["clarity"] = encoded_clarity

    df[['cut','color','clarity']] = df[['cut','color','clarity']].replace(100, 0)

    return df

def newFeatures(df:pd.DataFrame) -> pd.DataFrame:

    df["total_depth"] = (df["z"]*2)/(df["x"]+df["y"])
    df["size"] = df["x"] * df["y"] * df["z"]

    df = df.drop(["x","y","z"],axis=1)

    return df

@transformer
def transform(data, *args, **kwargs) -> pd.DataFrame:
    
    cleaned_data = encodeCats(newFeatures(removenull(data)))
    
    return cleaned_data


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
