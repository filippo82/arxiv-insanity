from __future__ import annotations

import json
import sys

import prefect
from kaggle.api.kaggle_api_extended import KaggleApi
from prefect import flow
from prefect import get_run_logger
from prefect import task
from prefect.task_runners import SequentialTaskRunner
from prefect_gcp.cloud_storage import GcsBucket

FN = "arxiv-metadata-oai-snapshot.json"
FN_PROCESSED = "arxiv-metadata-oai-snapshot-processed.jsonl"

PREFECT_STORAGE_BLOCK_GCS_BUCKET = "block-bucket-arxiv-data"


@task
def download_dataset():
    """Download the metadata of all arXiv preprints from Kaggle.
    The [arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)
    is updated on a monthly basis.
    """
    logger = get_run_logger()

    # Create Kaggle client
    api = KaggleApi()
    api.authenticate()

    logger.info("Start downloading dataset")
    api.dataset_download_files("Cornell-University/arxiv", quiet=False, unzip=True)
    logger.info("Dataset has been downloaded")


@task
def prepare_jsonl_for_bigquery():
    """Process the downloaded metadata file so that it can be directly loaded
    into BigQuery. This [comment](https://www.kaggle.com/datasets/Cornell-University/arxiv/discussion/376149)
    explains why this additional processing is required.
    """
    logger = get_run_logger()

    # BigQuery field names cannot contain `-`.
    # Hence, we replace `-` with `_`.
    invalid_field_names = {
        "journal-ref": "journal_ref",
        "report-no": "report_no",
    }

    # We need to convert `authors_parsed`, which is a `List[List[str]]`,
    # to a `List[Dict[str, str]]`.
    # This corresponds to an ARRAY of RECORDS in BigQuery.

    log_step = 100000

    logger.info("Start processing JSONL file")
    with open(FN) as f, open(FN_PROCESSED, "w") as fp:
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
                            },
                        )
                    processed["authors_parsed"] = authors_processed
                else:
                    processed[k] = v

            # Validate processed entry
            # TODO

            json.dump(processed, fp)
            fp.write("\n")

            cnt += 1
            if cnt % log_step == 0:
                logger.info(f"Processed {cnt:7d} articles")

    logger.info(f"Processed a total of {cnt:7d} articles")


@task
def copy_processed_file_to_bucket() -> str:
    """Copy the processed file to a GCS bucket.

    Returns
    -------
    str
        URI of the processed file in the GCS bucket.
    """
    gcp_cloud_storage_bucket_block = GcsBucket.load(PREFECT_STORAGE_BLOCK_GCS_BUCKET)
    path = gcp_cloud_storage_bucket_block.upload_from_path(FN_PROCESSED, f"kaggle/{FN_PROCESSED}")
    return path


@task
def log_prefect_version(name: str):
    """Simple logger which prints the Prefect version.

    Parameters
    ----------
    name : str
        The name of a dude or dudette.
    """
    logger = get_run_logger()
    logger.info("Hello %s!", name)
    logger.info("Prefect Version = %s 🚀", prefect.__version__)


@flow(task_runner=SequentialTaskRunner())
def flow_get_arxiv_kaggle_dataset(name: str = "Filippo"):
    """Download from Kaggle, process, and copy the JSONL file to a GCS bucket.

    Parameters
    ----------
    name : str, optional
        The name of a dude or dudette, by default "Filippo".
    """
    logger = get_run_logger()
    log_prefect_version(name)
    download_dataset()
    prepare_jsonl_for_bigquery()
    path = copy_processed_file_to_bucket()
    logger.info(f"The processed file has been stored at {path}")


if __name__ == "__main__":
    name = sys.argv[1]
    flow_get_arxiv_kaggle_dataset(name)
