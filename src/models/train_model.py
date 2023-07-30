import os
import pickle
import click
import mlflow
import optuna

from optuna.samplers import TPESampler
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from io import BytesIO

import logging
log = logging.getLogger(__name__ + 'train_model')

RF_PARAMS = [
    'iterations', 'learning_rate', 'n_estimators', 'depth', 
    'subsample', 'colsample_bylevel', 'min_data_in_leaf', 'random_state'
]


def get_paths(client, bucket_name, in_data_path, in_cloud, filename):
    if in_cloud:
        file_path = os.path.join(in_data_path, 'processed')
        # response = client.get_object(Bucket=bucket_name, Key=os.path.join(file_path, filename))
        # input_file = BytesIO(response['Body'].read())

        input_file = BytesIO()
        object_key = os.path.join(file_path, filename)
        client.download_fileobj(bucket_name, object_key, input_file)
        input_file = input_file.getvalue()
        return input_file
    
    return input_file if input_file else os.path.join(in_data_path, filename)


def load_pickle(filename):
    return pickle.loads(filename)


def run_optimization(data_path: str, num_trials: int, client, in_data_path, in_cloud, project_name):

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment('hpo_' + project_name)
    mlflow.sklearn.autolog(disable=True)

    train = get_paths(client, 'database', in_data_path, in_cloud, 'train.pkl')
    test = get_paths(client, 'database', in_data_path, in_cloud, 'test.pkl')

    X_train, y_train = load_pickle(train)
    X_val, y_val = load_pickle(test)

    def objective(trial):
        with mlflow.start_run():

            params = {
                'iterations': 500,
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'depth': trial.suggest_int('depth', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.05, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.05, 1.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
                'random_seed':42
            }
                
            rf = CatBoostClassifier(**params)
            mlflow.log_params(params)
            import ipdb; ipdb.set_trace()
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            f1 = f1_score(y_val, y_pred)
            mlflow.log_metric('f1_score', f1)
            return f1

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=num_trials)


def train_and_log_model(data_path, params, project_name):

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment(project_name)

    X_train, y_train = load_pickle(os.path.join(data_path, 'train.pkl'))
    X_test, y_test = load_pickle(os.path.join(data_path, 'test.pkl'))


    with open('models/lin_reg.bin', 'wb') as f_out:
        pickle.dump((dv, lr), f_out)

    with mlflow.start_run():
        for param in RF_PARAMS:
            params[param] = int(params[param])


        clf = CatBoostClassifier(**params)
        clf.fit(X_train, y_train)


        f1 = f1_score(y_test, clf.predict(X_test), squared=False)
        mlflow.log_metric('f1_score', f1)
        import ipdb; ipdb.set_trace()
        # maybe save pipeline with ohe and others pre-process
        # mlflow.log_artifact(local_path='models/lin_reg.bin', artifact_path='models_pickle')

# @click.command()
# @click.option(
#     '--data_path',
#     default='./output',
#     help='Location where the processed NYC taxi trip data was saved'
# )
# @click.option(
#     '--top_n',
#     default=5,
#     type=int,
#     help='Number of top models that need to be evaluated to decide which one to promote'
# )
def run_register_model(data_path: str, top_n: int, project_name):
    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name('hpo_' + project_name)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        filter = "metrics.precision > 0.9 and metrics.recall > 0.8",
        order_by=['metrics.f1 DESC']
    )

    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(project_name)

    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=['metrics.test_rmse ASC']
    )[0]
    
    # Register the best model
    RUN_ID = best_run.info.run_id
    mlflow.register_model(
        f'runs:/{RUN_ID}/model', project_name
    )


if __name__ == '__main__':
    run_register_model()
