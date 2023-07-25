import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed data was saved"
)
def run_train(data_path: str):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment")
    model_name = 'random'
    name_registry = f'model_{model_name}'

    with mlflow.start_run():
        mlflow.set_tag('developer', 'weslley')
        
        mlflow.sklearn.autolog(registered_model_name=name_registry)
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        # signature = infer_signature(X_val, y_val)

        # mlflow.sklearn.log_model(
        #     sk_model=rf,
        #     artifact_path='models_pickle',
        #     registered_model_name=name_registry,
        #     signature=signature
        # )

if __name__ == '__main__':
    run_train()
