import mlflow
import pickle
import os
from typing import List, Tuple
from pandas import DataFrame, Series

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)

@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    dv, regr = data

    # mlflow.set_tracking_uri('db_type:///path_to_db')
    mlflow.set_experiment('homework_03')
    mlflow.sklearn.autolog()
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(regr, "model")
        print(f"Model saved in /workspaces/mlops-zoomcamp2024/03-orchestration/mage-mlops/mlruns/315181116513058938/{mlflow.active_run().info.run_uuid}")

        # Save DictVectorizer
        dump_pickle(dv, os.path.join('/home/src/mlops/homework_03', "dv.pkl"))
        mlflow.log_artifact(os.path.join('/home/src/mlops/homework_03', "dv.pkl"), "DictVectorizer")
