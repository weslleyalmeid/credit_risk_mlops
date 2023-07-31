## Problem description

Credit analysis prediction problem, project based on the one from kaggle.

database: https://www.kaggle.com/datasets/laotse/credit-risk-dataset
model: https://www.kaggle.com/code/anshtanwar/credit-risk-prediction-precision-100#Modeling


## Implementation plan:

- [x] cleanup data
- [x] exploratory data analysis
- [x] train model
- [x] ml pipeline for hyperparameter tuning
- [x] model registry
- [x] Web server
- [x] Batch
- [x] docker and docker-compose
- [ ] Monitoring
- [ ] tests (partial)
- [ ] linters
- [ ] Makefile and CI/CD
- [ ] deploy to cloud
- [ ] logging and monitoring



## Initialization

1 - Start docker compose

```sh
docker compose --env-file .env up -d
```

2 - Run init-project
```sh
python init_project.py --name_project 'credit_risk'
```

3 - Preprocess, train optimization, registry model and data reference monitoring
```sh
python main.py --preprocess True --hpo True --in_cloud False --out_cloud True
```

4 - Run model web-service
```sh
cd web-service

docker build -t credit-risk-predict:v1 .

docker run -it --rm\
        --env-file .env \
        -p 9696:9696 \
        --network credit_risk_mlops_internal \
        -v $(pwd)/model:/app/model \
        credit-risk-predict:v1
```

request
```sh
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "person_age": 22,
    "person_income": 59000,
    "person_home_ownership": "RENT",
    "person_emp_length": 123,
    "loan_intent": "PERSONAL",
    "loan_grade": "D",
    "loan_amnt": 35000,
    "loan_int_rate": 16.02,
    "loan_percent_income": 0.59,
    "cb_person_default_on_file": "Y",
    "cb_person_cred_hist_length": 3
  }' \
  http://0.0.0.0:9696/predict
```

5 - Run model batch
```sh
cd batch

python batch.py
```