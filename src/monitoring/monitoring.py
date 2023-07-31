import os
import pickle
import logging

log = logging.getLogger(__name__ + 'pre_process')


def save_results(s3_client, object, bucket_name, file_name):
    s3_client.put_object(Body=object, Bucket=bucket_name, Key=file_name)


def save_reference_monitoring(project_name, s3_client, df, run_id):
    data_reference = pickle.dumps(df)
    bucket_name = 'monitoring'
    file_name= os.path.join(project_name, run_id, 'data_reference.pkl')
    save_results(s3_client, data_reference, bucket_name, file_name)
