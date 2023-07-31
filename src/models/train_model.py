import os
import pickle
import click
import mlflow
import optuna

from optuna.samplers import TPESampler
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, recall_score, precision_score
from io import BytesIO
from monitoring.monitoring import save_reference_monitoring

import logging
log = logging.getLogger(__name__ + 'train_model')

RF_PARAMS = [
    'iterations', 'learning_rate', 'depth', 'subsample',
    'colsample_bylevel', 'min_data_in_leaf', 'random_seed'
]


def get_paths(s3_client, bucket_name, in_data_path, in_cloud, filename):
    if in_cloud:
        if bucket_name == 'pipeline':
            object_key = os.path.join(in_data_path, filename)
        else:
            file_path = os.path.join(in_data_path, 'processed')
            object_key = os.path.join(file_path, filename)

        input_file = BytesIO()
        s3_client.download_fileobj(bucket_name, object_key, input_file)
        input_file = input_file.getvalue()
        return input_file
    
    return input_file if input_file else os.path.join(in_data_path, filename)


def load_pickle(filename):
    return pickle.loads(filename)

def dump_pickle(obj, filename: str):
    with open(filename, 'wb') as f_out:
        return pickle.dump(obj, f_out)


def run_optimization(data_path: str, num_trials: int, client, in_data_path, in_cloud, project_name):

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment('hpo_' + project_name)
    mlflow.autolog(disable=True)

    train = get_paths(client, 'database', in_data_path, in_cloud, 'train.pkl')
    test = get_paths(client, 'database', in_data_path, in_cloud, 'test.pkl')

    X_train, y_train = load_pickle(train)
    X_val, y_val = load_pickle(test)

    def objective(trial):
        with mlflow.start_run():

            params = {
                'iterations': 1000,
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'depth': trial.suggest_int('depth', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.05, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.05, 1.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
                'random_seed':42
            }
                
            clf = CatBoostClassifier(**params)
            mlflow.log_params(params)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            f1 = f1_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            mlflow.log_metric('f1_score', f1)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)
            mlflow.catboost.log_model(clf, artifact_path='model')
            return f1

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=num_trials)


def train_and_log_model(params, project_name, X_train, y_train, X_test, y_test, pipeline_transform):

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment(project_name)
 
    os.makedirs(f'./temp/pipeline/{project_name}', exist_ok=True)
    pipeline_pkl = pickle.dumps(pipeline_transform)
    dump_pickle(pipeline_pkl, f'./temp/pipeline/{project_name}/pipeline.pkl')

    with mlflow.start_run():
        # maybe save pipeline with ohe and others pre-process
        for param in RF_PARAMS:
            if params[param].isdecimal():
                params[param] = int(params[param])
            else:
                params[param] = float(params[param])

        clf = CatBoostClassifier(**params)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        mlflow.log_metric('f1_score', f1)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)

        signature = infer_signature(X_test, y_pred)
        
        mlflow.catboost.log_model(
            cb_model=clf,
            artifact_path='model',
            signature=signature
        )
        mlflow.log_artifact(f'./temp/pipeline/{project_name}/pipeline.pkl', artifact_path='preprocess')


def upgrade_stage(mlf_client, experiment, best_run):
    
    precision = best_run.data.metrics['precision']
    recall = best_run.data.metrics['recall']
    f1_score = best_run.data.metrics['f1_score']

    # get id of the experiment
    id_experiment = experiment.experiment_id

    for mv in mlf_client.search_model_versions(f"name='{experiment.name}'"):
        if mv.current_stage == 'Production':
            # get execution id of the production
            id_run = mv.run_id
            
            # runs get all experiments the specific id
            runs = mlflow.search_runs(experiment_ids=id_experiment)
            df_experiment = runs[runs['run_id'] == id_run]         

            precision_prod = float(df_experiment['metrics.precision'].values)
            recall_prod = float(df_experiment['metrics.recall'].values)
            f1_prod = float(df_experiment['metrics.f1_score'].values)

            if precision > precision_prod and recall > recall_prod and f1_score > f1_prod:
                return 'production'
 
            return None
        
    return 'production'


def model_set_stage(best_run, experiment):

    mlf_client = mlflow.tracking.MlflowClient(tracking_uri=os.environ['MLFLOW_TRACKING_URI'])
    promoved = upgrade_stage(mlf_client, experiment, best_run)

    if promoved:

        version = mlf_client.get_latest_versions(name=experiment.name, stages=['None'])[-1]
        current_version = version.version

        mlf_client.transition_model_version_stage(
            name=experiment.name,
            version=current_version,
            stage="Production",
            archive_existing_versions=True
        )

        return 1

    return 0


def run_register_model(data_path: str, top_n: int, project_name, s3_client, in_cloud):
    ml_client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = ml_client.get_experiment_by_name('hpo_' + project_name)
    runs = ml_client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        filter_string = "metrics.precision > 0.85 and metrics.recall > 0.7",
        order_by=['metrics.f1_score DESC']
    )

    pipeline_transform = get_paths(s3_client, 'pipeline', data_path, in_cloud, 'pipeline.pkl')
    train = get_paths(s3_client, 'database', data_path, in_cloud, 'train.pkl')
    test = get_paths(s3_client, 'database', data_path, in_cloud, 'test.pkl')
    
    pipeline_transform = load_pickle(pipeline_transform)
    X_train, y_train = load_pickle(train)
    X_test, y_test = load_pickle(test)

    for run in runs:
        train_and_log_model(
            params=run.data.params,
            project_name=project_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            pipeline_transform=pipeline_transform
        )

    experiment = ml_client.get_experiment_by_name(project_name)

    # Select the model with the best metrics
    best_run = ml_client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        filter_string = "metrics.precision > 0.85 and metrics.recall > 0.7",
        order_by=['metrics.f1_score DESC']
    )[0]

    # Register the best model
    RUN_ID = best_run.info.run_id
    mlflow.register_model(
        f'runs:/{RUN_ID}/model', project_name
    )

    set_ref_monitoring = model_set_stage(best_run, experiment)

    # set new data referece for monitoring
    if set_ref_monitoring:
        save_reference_monitoring(project_name, s3_client, test, RUN_ID)


if __name__ == '__main__':
    run_register_model()
