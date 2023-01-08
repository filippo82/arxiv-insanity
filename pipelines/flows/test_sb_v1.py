import json
import sys
import prefect
from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner

# from prefect_gcp import GcpCredentials, GcsBucket
# from prefect_gcp.credentials import GcpCredentials
from prefect_gcp.cloud_storage import GcsBucket

from kaggle.api.kaggle_api_extended import KaggleApi

fn = "arxiv-metadata-oai-snapshot.json"
fn_processed = "arxiv-metadata-oai-snapshot-processed.jsonl"


@task
def download_dataset():
    logger = get_run_logger()
    api = KaggleApi()
    api.authenticate()

    logger.info("Start - Download dataset")
    api.dataset_download_files("Cornell-University/arxiv", quiet=False, unzip=True)
    logger.info("End - Download dataset")


@task
def prepare_json_for_bigquery():
    logger = get_run_logger()

    # BigQuery field names cannot contain `-`.
    # Hence, we replace `-` with `_`
    invalid_field_names = {
        "journal-ref": "journal_ref",
        "report-no": "report_no",
    }

    # We need to convert `authors_parsed`, which is a `List[List[str]]`,
    # to a `List[Dict[str, str]]`.
    # This corresponds to an ARRAY of RECORDS in BigQuery.

    log_step = 100000

    with open(fn, "r") as f, open(fn_processed, "w") as fp:
        cnt = 0
        for line in f:
            article = json.loads(line)
            processed = {}
            for k, v in article.items():
                if k in invalid_field_names.keys():
                    processed[invalid_field_names[k]] = v
                elif k == "authors_parsed":
                    authors_processed = []
                    for author in article["authors_parsed"]:
                        authors_processed.append(
                            {
                                "last_name": author[0],
                                "first_name": author[1],
                                "other_name": author[2],
                            }
                        )
                    processed["authors_parsed"] = authors_processed
                else:
                    processed[k] = v

            # Validate processed entry

            json.dump(processed, fp)
            fp.write("\n")

            cnt += 1
            if cnt % log_step == 0:
                logger.info(f"Processed {cnt:7d} articles")

    logger.info(f"Processed a total of {cnt:7d} articles")


@task
def copy_processed_file_to_bucket():
    # gcp_credentials_block = GcpCredentials.load("gcp-credentials")
    gcp_cloud_storage_bucket_block = GcsBucket.load("block-gcs-bucket")
    path = gcp_cloud_storage_bucket_block.upload_from_path(
        fn_processed, f"root/arxiv-test/{fn_processed}"
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
    download_dataset()
    prepare_json_for_bigquery()
#    copy_processed_file_to_bucket()


if __name__ == "__main__":
    name = sys.argv[1]
    log_flow(name)
