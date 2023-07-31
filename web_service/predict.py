import os
import pickle

import mlflow
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import pandas as pd


# load environments variables
load_dotenv('.env')

# setting credentials cloud
os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('MLFLOW_S3_ENDPOINT_URL')
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_NAME_PROJECT'] = os.getenv('AWS_NAME_PROJECT')
os.environ['MLFLOW_TRACKING_URI'] = os.getenv('MLFLOW_TRACKING_URI')

# logged_model = f's3://mlflow-models/1/{RUN_ID}/artifacts/model'
# model = mlflow.pyfunc.load_model(logged_model)

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


def predict(features):
    preds = model.predict(features)
    return int(preds[0])


app = Flask('credit-risk')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data_json = request.get_json()
    df = pd.DataFrame([data_json])
    features = prepare_features(df)
    pred = predict(features)


    result = {
        'loan_status': pred,
        'model_version': RUN_ID
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=9696)
