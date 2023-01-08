from __future__ import annotations

import sys

import prefect
from prefect import flow
from prefect import get_run_logger
from prefect import task
from prefect.task_runners import SequentialTaskRunner
from prefect_gcp.cloud_storage import GcsBucket

# import json
# from kaggle.api.kaggle_api_extended import KaggleApi

# from prefect_gcp import GcpCredentials, GcsBucket
# from prefect_gcp.credentials import GcpCredentials

fn = "arxiv-metadata-oai-snapshot.json"
fn_processed = "arxiv-metadata-oai-snapshot-processed.jsonl"


@task
def create_mock_file():
    with open(fn_processed, "w") as f:
        f.write("Ciaoneeee!")


@task
def copy_processed_file_to_bucket():
    # gcp_credentials_block = GcpCredentials.load("gcp-credentials")
    gcp_cloud_storage_bucket_block = GcsBucket.load("block-gcs-bucket")
    path = gcp_cloud_storage_bucket_block.upload_from_path(
        fn_processed,
        f"arxiv-test/{fn_processed}",
    )
    return path


@task
def log_task(name):
    logger = get_run_logger()
    logger.info("Hello %s!", name)
    logger.info("Prefect Version = %s 🚀", prefect.__version__)


@flow(task_runner=SequentialTaskRunner())
def log_flow(name: str):
    log_task(name)


#    copy_processed_file_to_bucket()


if __name__ == "__main__":
    name = sys.argv[1]
    log_flow(name)
