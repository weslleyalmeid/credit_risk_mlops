import os
import pickle

import mlflow
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Carregar as variáveis de ambiente do arquivo .secrets
load_dotenv('.env')

RUN_ID = os.getenv('RUN_ID')

logged_model = f's3://mlflow-models/1/{RUN_ID}/artifacts/model'
# logged_model = f'runs:/{RUN_ID}/model'

# Definir a configuração do backend store do MLflow
os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('minio_endpoint_url')
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('minio_access_key')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('minio_secret_key')
# os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'
# os.environ['MLFLOW_TRACKING_URI'] = os.getenv('minio_bucket_name')

model = mlflow.pyfunc.load_model(logged_model)


def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features


def predict(features):
    preds = model.predict(features)
    return float(preds[0])


app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred,
        'model_version': RUN_ID
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9697)
