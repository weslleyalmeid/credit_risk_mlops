#!/usr/bin/env python
# coding: utf-8


import os
import sys

import uuid
import pickle

import pandas as pd
import mlflow
import boto3

# from prefect.context import get_run_context

from dateutil.relativedelta import relativedelta
from datetime import datetime

from io import BytesIO
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

from dotenv import load_dotenv

# load environments variables
load_dotenv('../.env')

# setting credentials cloud
os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('MLFLOW_S3_ENDPOINT_URL')
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_NAME_PROJECT'] = os.getenv('AWS_NAME_PROJECT')
os.environ['MLFLOW_TRACKING_URI'] = os.getenv('MLFLOW_TRACKING_URI')


mlf_client = mlflow.tracking.MlflowClient(tracking_uri=os.environ['MLFLOW_TRACKING_URI'])
model_production = mlf_client.get_latest_versions(name=os.environ['AWS_NAME_PROJECT'], stages=['Production'])[0]
model = mlflow.catboost.load_model(model_production.source)

RUN_ID = model_production.run_id

def load_pipeline():
    mlf_client.download_artifacts(run_id=RUN_ID, path='preprocessor', dst_path='.')

    with open("preprocessor/pipeline.pkl", "rb") as f_in:
        pipe = pickle.load(f_in)
    return pickle.loads(pipe)

pipeline_transformer = load_pipeline()


# Criação do cliente S3 apontando para o MinIO
s3_client = boto3.client('s3',
                         endpoint_url=os.environ['MLFLOW_S3_ENDPOINT_URL'],
                         aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                         aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])


def get_paths(s3_client, bucket_name, project_name, filename):
    object_key = os.path.join(project_name, filename)

    input_file = BytesIO()
    response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    input_file = BytesIO(response['Body'].read())
    return input_file


def read_dataframe(filename: str):
    return pd.read_csv(filename)


def prepare_features(df: pd.DataFrame):
    
    df['age_group'] = pd.cut(
                            df['person_age'],
                            bins=[20, 26, 36, 46, 56, 66],
                            labels=['20-25', '26-35', '36-45', '46-55', '56-65']
                        )

    df['income_group'] = pd.cut(
                            df['person_income'],
                            bins=[0, 25000, 50000, 75000, 100000, float('inf')],
                            labels=['low', 'low-middle', 'middle', 'high-middle', 'high']
                        )

    df['loan_amount_group'] = pd.cut(
                                    df['loan_amnt'],
                                    bins=[0, 5000, 10000, 15000, float('inf')],
                                    labels=['small', 'medium', 'large', 'very large']
                                )

    # Create loan-to-income ratio
    df['loan_to_income_ratio'] = df['loan_amnt'] / df['person_income']

    # Create loan-to-employment length ratio
    df['loan_to_emp_length_ratio'] =  df['person_emp_length']/ df['loan_amnt']

    # Create interest rate-to-loan amount ratio
    df['int_rate_to_loan_amt_ratio'] = df['loan_int_rate'] / df['loan_amnt']

    cols = pipeline_transformer.named_steps['preprocessor'].get_feature_names_out()
    df = pd.DataFrame(pipeline_transformer.transform(df), columns=cols)
    return df


def apply_model(features):
    preds = model.predict(features)
    return preds


def save_results(df, y_pred, run_id, bucket_name, filename):
    df_result = pd.DataFrame()
    df_result = df
    df_result['predicted_loan_status'] = y_pred
    df_result['model_version'] = run_id

    # parquet_data = df_result.to_parquet(output_file, index=False)
    csv_data = df_result.to_csv(index=False)
    s3_client.put_object(Body=csv_data, Bucket=bucket_name, Key=filename)


def model_prediction():
    input_file = get_paths(s3_client, 'batch', os.environ['AWS_NAME_PROJECT'], 'credit_risk_batch.csv')
    input_file = read_dataframe(input_file)
    features = prepare_features(input_file)
    pred = apply_model(features)
    save_results(features, pred, RUN_ID, 'batch', os.path.join(os.environ['AWS_NAME_PROJECT'], 'result', 'credit_risk_result.csv'))


def run():
    # taxi_type = sys.argv[1] # 'green'
    # year = int(sys.argv[2]) # 2021
    # month = int(sys.argv[3]) # 3

    # run_id = sys.argv[4] # '874e7ae01334406590ac62ddc0422882'
    model_prediction()


if __name__ == '__main__':
    run()