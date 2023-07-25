FROM continuumio/miniconda3:latest

RUN pip install mlflow boto3 psycopg2-binary

ADD . /app
WORKDIR /app
