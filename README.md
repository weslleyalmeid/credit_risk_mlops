## Course Project

The goal of this project is to apply everything we learned
in this course and build an end-to-end machine learning project.

Remember that to pass the project, you must evaluate 3 peers. If you don't do that, your project can't be considered compelete.  

* [2022 Projects](../cohorts/2022/07-project)
* 2023 Projects (TBD)


## Problem statement

For the project, we will ask you to build an end-to-end ML project. 

For that, you will need:

* Select a dataset that you're interested in (see [datasets.md](https://github.com/DataTalksClub/data-engineering-zoomcamp/blob/main/week_7_project/datasets.md))
* Train a model on that dataset tracking your experiments
* Create a model training pipeline
* Deploy the model in batch, web service or streaming
* Monitor the performance of your model
* Follow the best practices 


## Technologies 

You don't have to limit yourself to technologies covered in the course. You can use alternatives as well:

* Cloud: AWS, GCP, Azure or others
* Experiment tracking tools: MLFlow, Weights & Biases, ... 
* Workflow orchestration: Prefect, Airflow, Flyte, Kubeflow, Argo, ...
* Monitoring: Evidently, WhyLabs/whylogs, ...
* CI/CD: Github actions, Gitlab CI/CD, ...
* Infrastructure as code (IaC): Terraform, Pulumi, Cloud Formation, ...

If you use something that wasn't covered in the course, 
be sure to explain what the tool does.

If you're not certain about some tools, ask in Slack.


## Peer review criteria

(This is still a draft. Feedbask is welcome)

* Problem description
    * 0 points: Problem is not described
    * 1 point: Problem is described but shortly or not clearly 
    * 2 points: Problem is well described and it's clear what the problem the project solves
* Cloud
    * 0 points: Cloud is not used, things run only locally
    * 2 points: The project is developed on the cloud OR the project is deployed to Kubernetes or similar container management platforms
    * 4 points: The project is developed on the cloud and IaC tools are used for provisioning the infrastructure
* Experiment tracking and model registry
    * 0 points: No experiment tracking or model registry
    * 2 points: Experiments are tracked or models are registred in the registry
    * 4 points: Both experiment tracking and model registry are used
* Workflow orchestration
    * 0 points: No workflow orchestration
    * 2 points: Basic workflow orchestration
    * 4 points: Fully deployed workflow 
* Model deployment
    * 0 points: Model is not deployed
    * 2 points: Model is deployed but only locally
    * 4 points: The model deployment code is containerized and could be deployed to cloud or special tools for model deployment are used
* Model monitoring
    * 0 points: No model monitoring
    * 2 points: Basic model monitoring that calculates and reports metrics
    * 4 points: Comprehensive model monitoring that send alerts or runs a conditional workflow (e.g. retraining, generating debugging dashboard, switching to a different model) if the defined metrics threshold is violated
* Reproducibility
    * 0 points: No instructions how to run code at all
    * 2 points: Some instructions are there, but they are not complete
    * 4 points: Instructions are clear, it's easy to run the code, and the code works. The version for all the dependencies are specified.
* Best practices
    * [ ] There are unit tests (1 point)
    * [ ] There is an integration test (1 point)
    * [ ] Linter and/or code formatter are used (1 point)
    * [ ] There's a Makefile (1 point)
    * [ ] There are pre-commit hooks (1 point)
    * [ ] There's a CI/CD pipeline (2 points)




## Problem description


## Implementation plan:

- [x] cleanup data
- [x] exploratory data analysis
- [x] train model
- [x] ml pipeline for hyperparameter tuning
- [x] model registry
- [] ML-serve API server
- [] ML-serve Stream server (optional)
- [] tests (partial)
- [] linters
- [] Makefile and CI/CD
- [] deploy to cloud
- [] logging and monitoring
- [] batch reporting
- [] docker and docker-compose everything
- [] reporting server


## Initialization

### Storage with MinIO

run docker compose for get up minIO, access in *localhost:9001*
```sh
docker-compose --env-file .secrets  up -d
docker-compose --env-file .secrets  down
```

create bucket
```sh
aws s3 mb \
    s3://mlflow-models \
    --endpoint-url=http://localhost:9000 \
    --profile minio_user
```

### Tracking and Registry with MLflow

[Scenario 5](https://mlflow.org/docs/latest/tracking.html#scenario-5-mlflow-tracking-server-enabled-with-proxied-artifact-storage-access) MLflow Tracking Server enabled with proxied artifact storage access: 
```sh
# <dialect>+<driver>://<username>:<password>@<host>:<port>/<database>
mlflow server \
    --backend-store-uri=postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5432/${POSTGRES_DATABASE} \
    --default-artifact-root=s3://${AWS_BUCKET_NAME}/ \
    --artifacts-destination=s3://${AWS_BUCKET_NAME}/ \
    --host=0.0.0.0
```