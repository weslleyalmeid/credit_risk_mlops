import sys
import logging
from logging import basicConfig                     # configuracao do logging
from logging import DEBUG, INFO                     # levels
from logging import FileHandler, StreamHandler      # Mostrar log no terminal e pode salver em N arquivos

from data_processing.preprocess import run_data_prep
from models.train_model import run_optimization
from dotenv import load_dotenv
import os
import boto3
import click

# load environments variables
load_dotenv('../.env')

# setting credentials cloud
os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('MLFLOW_S3_ENDPOINT_URL')
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_NAME_PROJECT'] = os.getenv('AWS_NAME_PROJECT')
os.environ['MLFLOW_TRACKING_URI'] = os.getenv('MLFLOW_TRACKING_URI')

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)

os.environ['SRC_DIR'] = SRC_DIR
os.environ['ROOT_DIR'] = ROOT_DIR

# create client
s3_client = boto3.client('s3',
                         endpoint_url=os.environ['MLFLOW_S3_ENDPOINT_URL'],
                         aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                         aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])


basicConfig(
    level=INFO,
    format='%(levelname)s:%(asctime)s:%(message)s',
    handlers=[
        StreamHandler()
    ]
)

log = logging.getLogger(__name__)


@click.command()
@click.option(
    '--in_cloud',
    default=False,
    required=True,
    help='Input location where the raw data was saved'
)
@click.option(
    '--out_cloud',
    default=False,
    required=True,
    help='Output location where the artifacts and data will saved'
)
def main(in_cloud, out_cloud):
    log.info('Program started')

    log.info('Set directories and paths')
    
    if in_cloud:
        RAW_PATH = os.path.join('s3://', 'database', os.environ['AWS_NAME_PROJECT'], 'raw')
    else:
        DATA_DIR = os.path.join(ROOT_DIR, 'data', '{folder}')
        RAW_PATH = DATA_DIR.format(folder='raw')
        
    if out_cloud:
        OUT_DATA_PATH = OUT_PIPE_PATH = os.environ['AWS_NAME_PROJECT']
    else:
        OUT_DATA_PATH = os.path.join(ROOT_DIR, 'output', 'database', os.environ['AWS_NAME_PROJECT'])
        OUT_PIPE_PATH = os.path.join(ROOT_DIR, 'output', 'pipeline', os.environ['AWS_NAME_PROJECT'])

    log.info('Run data pre-process')
    if False:
        run_data_prep(
            in_data_path=RAW_PATH,
            out_data_path=OUT_DATA_PATH,
            out_pipe_path=OUT_PIPE_PATH,
            in_cloud=in_cloud,
            out_cloud=out_cloud,
            dataset='credit_risk.csv',
            client=s3_client
        )

    log.info('Run optimization model')

    run_optimization(OUT_DATA_PATH, 5, s3_client, OUT_DATA_PATH, in_cloud, os.environ['AWS_NAME_PROJECT'])

    log.info('Program finished')


if __name__ == '__main__':
    main()
    