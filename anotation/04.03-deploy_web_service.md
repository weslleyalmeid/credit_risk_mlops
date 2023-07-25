## Getting the model for deployment from MLflow

* Take the code from the previous video
* Train another model, register with MLflow
* Put the model into a scikit-learn pipeline
* Model deployment with tracking server
* Model deployment without the tracking server

Starting the MLflow server with S3:

```bash

export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minio_user
export AWS_SECRET_ACCESS_KEY=minio_password

export MINIO_ACCESS_KEY=minio_user
export MINIO_SECRET_KEY=minio_password

mlflow server \
    --backend-store-uri=sqlite:///mlflow.db \
    --default-artifact-root=s3://mlflow-weslley/
```

```
ln -s ~/PROJETOS/MLEngineer-Studies/mlops-zoomcamp/data ./data
```




Downloading the artifact

```bash
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
export MODEL_RUN_ID="6dd459b11b4e48dc862f4e1019d166f6"

mlflow artifacts download \
    --run-id ${MODEL_RUN_ID} \
    --artifact-path model \
    --dst-path .
```