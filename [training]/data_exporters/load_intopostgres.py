from mage_ai.settings.repo import get_repo_path
from mage_ai.io.config import ConfigFileLoader
from mage_ai.io.postgres import Postgres
import pandas as pd
from os import path

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data_to_postgres(data, **kwargs) -> None:

    if kwargs['export_data'] == "yes" : 

        X_train, X_validation, y_train, y_test = data


        schema_name = 'public'  # Specify the name of the schema to export data to
        table_name = 'mage_traindata'  # Specify the name of the table to export data to
        config_path = path.join(get_repo_path(), 'io_config.yaml')
        config_profile = 'default'

        train_df = pd.concat([X_train, y_train])
        test_df = pd.concat([X_validation, y_test])

        with Postgres.with_config(ConfigFileLoader(config_path, config_profile)) as loader:
            loader.export(
                y_test,
                schema_name,
                table_name,
                index=True,  # Specifies whether to include index in exported table
                if_exists='replace',  # Specify resolution policy if table name already exists
            )

        return f"Saved training dataset to {table_name} table."

    else:
        return f"Dataset wasn't saved since export_data is set to {kwargs['export_data']}."