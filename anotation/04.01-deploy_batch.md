## Batch deployment

* Turn the notebook for training a model into a notebook for applying the model
* Turn the notebook into a script 
* Clean it and parametrize


## Notes

```sh
# convert .ipynb to .py
jupyter nbconvert --to script score.ipynb


# check file bucket
aws s3 ls s3://test-minio/ --endpoint-url http://localhost:9000 --profile minio_user


# set deploymment
python score_deploy.py


# work-pools > select_work > work_queues > name_queue > work_pool_queue_id
# prefect agent start work_queue_id
prefect agent start 25d04579-8dbb-428c-bc36-50f9d8b6d873


# enviar todos os arquivos do diretório para o bucket
aws s3 cp ./ s3://test-minio/ --endpoint-url http://localhost:9000 --recursive --profile minio_user

# pipenv
pipenv install scikit-learn==1.0.2 flask --python=3.9


docker build -t ride-duration-prediction:v1 .
docker run -it --rm ride-duration-prediction:v1

docker build --build-arg SECRETS_FILE=.secrets -t ride-duration-prediction:v1 .
docker run -it --rm -p 9000:9000 -p 9001:9001 ride-duration-prediction:v1
```