import os
import pickle
import click
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from io import BytesIO


import logging

log = logging.getLogger(__name__ + 'pre_process')


def get_paths(client, in_data_path, in_cloud, dataset):
    
    if in_cloud:

        bucket_name = 'database'
        file_path = in_data_path.split(bucket_name)[1]
        response = client.get_object(Bucket=bucket_name, Key=os.path.join(file_path, dataset))
        input_file = BytesIO(response['Body'].read())
        return input_file
    
    return input_file if input_file else os.path.join(in_data_path, f'{dataset}.csv')


def save_results(client, object, bucket_name, file_name):
    client.put_object(Body=object, Bucket=bucket_name, Key=file_name)


def dump_pickle(obj, filename: str):
    with open(filename, 'wb') as f_out:
        return pickle.dump(obj, f_out)


def read_dataframe(filename: str):
    return pd.read_csv(filename)


def preprocess(df: pd.DataFrame):
    
    df = df.drop(df[df['person_age'] > 80].index, axis=0)

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

    df = df.drop(df[df['person_emp_length'] > 60].index, axis=0)
    df = df.reset_index(drop=True)

    # Create loan-to-income ratio
    df['loan_to_income_ratio'] = df['loan_amnt'] / df['person_income']

    # Create loan-to-employment length ratio
    df['loan_to_emp_length_ratio'] =  df['person_emp_length']/ df['loan_amnt']

    # Create interest rate-to-loan amount ratio
    df['int_rate_to_loan_amt_ratio'] = df['loan_int_rate'] / df['loan_amnt']

    return df



def feature_engineer(df: pd.DataFrame, pipeline_transformer: Pipeline, fit_enconder: bool):
    if fit_enconder:
        # Fit the pipeline to the data
        pipeline_transformer.fit(df)

    cols = pipeline_transformer.named_steps['preprocessor'].get_feature_names_out()
    df = pd.DataFrame(pipeline_transformer.transform(df), columns=cols)
    return df, pipeline_transformer
    

# @click.command()
# @click.option(
#     '--in_data_path',
#     required=False,
#     default=None,
#     help='Location where the raw data was saved'
# )
# @click.option(
#     '--out_data_path',
#     required=False,
#     help='Location where the resulting files will be saved'
# )
# @click.option(
#     '--save_cloud',
#     required=False,
#     help='Flag for save files in the cloud'
# )
# @click.option(
#     '--dataset',
#     required=False,
#     help='Filename of dataset'
# )
def run_data_prep(in_data_path: str, out_data_path: str, out_pipe_path:str, in_cloud: bool, out_cloud: bool, dataset: str, client):

    # Load files
    log.info('Read data')

    if in_cloud:
        input_file = get_paths(client, in_data_path, in_cloud, dataset)
        df = read_dataframe(input_file)

    # Extract the target
    target = 'loan_status'
    X = df.drop([target], axis=1)
    Y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
  
    log.info('Pre-process data')
    X_train = preprocess(X_train)
    X_test = preprocess(X_test)

    ohe_colums = [
        'cb_person_default_on_file', 'loan_grade', 'person_home_ownership',
        'loan_intent', 'income_group', 'age_group', 'loan_amount_group'
    ]

    normal_col = [
        'person_income','person_age','person_emp_length', 'loan_amnt','loan_int_rate',
        'cb_person_cred_hist_length','loan_percent_income', 'loan_to_emp_length_ratio',
        'int_rate_to_loan_amt_ratio'
    ]
    
    log.info('Create transformer and pipeline')
    # Create the column transformer and pipeline
    column_transformer = ColumnTransformer(
        transformers= [
            ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ohe_colums),
            ('scaler', StandardScaler(), normal_col)
        ],
        verbose_feature_names_out=False)

    pipeline_transformer = Pipeline([('preprocessor', column_transformer)])


    log.info('Apply feature engineer')
    X_train, pipeline_transform = feature_engineer(X_train, pipeline_transformer, fit_enconder=True)
    X_test, _ = feature_engineer(X_test, pipeline_transformer, fit_enconder=False)

    log.info('Save artifacts')
    if out_cloud:
        pipeline_pkl = pickle.dumps(pipeline_transform) 
        train_pkl = pickle.dumps((X_train, y_train)) 
        test_pkl = pickle.dumps((X_test, y_test))
        save_results(client, pipeline_pkl, 'pipeline', os.path.join(out_data_path, 'pipeline.pkl'))
        save_results(client, train_pkl, 'database', os.path.join(out_data_path, 'processed', 'train.pkl'))
        save_results(client, test_pkl,'database', os.path.join(out_data_path, 'processed', 'test.pkl'))
        
    else:
        # Create out_data_path folder unless it already exists
        import ipdb; ipdb.set_trace()
        os.makedirs(out_data_path, exist_ok=True)
        os.makedirs(out_pipe_path, exist_ok=True)

        # Save Pipeline transform and datasets
        dump_pickle(pipeline_transform, os.path.join('pipeline', out_pipe_path, 'pipeline_transform.pkl'))
        dump_pickle((X_train, y_train), os.path.join(out_data_path, 'train.pkl'))
        dump_pickle((X_test, y_test), os.path.join(out_data_path, 'test.pkl'))


if __name__ == '__main__':
    run_data_prep()
