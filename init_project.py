from dotenv import load_dotenv
import os
import boto3
import click

# load environments variables
load_dotenv('.env')

# setting credentials cloud
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')
os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('MLFLOW_S3_ENDPOINT_URL')
# create client
s3_client = boto3.client(
                    's3',
                    endpoint_url=os.environ['MLFLOW_S3_ENDPOINT_URL'],
                    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
                )


def create_bucket_if_not_exists(bucket_name, key=None):
    
    try:
        if key:
            response = list(s3_client.list_objects(Bucket=bucket_name, Prefix=f'{key}/', MaxKeys=1))
            if 'Contents' in response:
                print(f'{bucket_name}: Nested folder {key} already exists and is owned by you.') 
            else:
                s3_client.put_object(Bucket=bucket_name, Key=f'{key}/')
                print(f'{bucket_name}: Nested folder {key} created successfully.')
        else:
            s3_client.create_bucket(Bucket=bucket_name)
            print(f'Bucket {bucket_name} created successfully.')
        
    except s3_client.exceptions.BucketAlreadyOwnedByYou or s3_client.exceptions.ClientError:
        print(f'Bucket {bucket_name} already exists and is owned by you.')


@click.command()
@click.option(
    '--name_project',
    required=True,
    help='Location where the raw data was saved'
)
def start_project(name_project):
    
    create_bucket_if_not_exists('mlflow-artifacts')

    create_bucket_if_not_exists('pipeline')
    create_bucket_if_not_exists('pipeline', name_project)
    
    create_bucket_if_not_exists('database')
    create_bucket_if_not_exists('database', os.path.join(name_project, 'raw'))
    create_bucket_if_not_exists('database', os.path.join(name_project, 'interim'))
    create_bucket_if_not_exists('database', os.path.join(name_project, 'processed'))


    create_bucket_if_not_exists('monitoring')
    create_bucket_if_not_exists('monitoring', name_project)


    create_bucket_if_not_exists('batch')
    create_bucket_if_not_exists('batch', name_project)


if __name__ == '__main__':
    start_project()