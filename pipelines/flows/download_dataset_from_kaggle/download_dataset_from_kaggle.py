from __future__ import annotations

import json

import prefect
from dateutil.parser import parse
from kaggle.api.kaggle_api_extended import KaggleApi
from prefect import flow
from prefect import get_run_logger
from prefect import task
from prefect.blocks.system import DateTime
from prefect.task_runners import SequentialTaskRunner
from prefect_gcp.cloud_storage import GcsBucket

DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
DATE_FORMAT = "%Y-%m-%d"

KAGGLE_DATASET_NAME = "Cornell-University/arxiv"

FN = "../arxiv-metadata-oai-snapshot.json"
FN_PROCESSED = "arxiv-metadata-oai-snapshot-processed.jsonl"

PREFECT_STORAGE_BLOCK_GCS_BUCKET = "block-bucket-arxiv-data"
PREFECT_STORAGE_BLOCK_DATETIME = "block-datetime-arxiv-data-last-updated"


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
    api.dataset_download_files(KAGGLE_DATASET_NAME, quiet=False, unzip=True)
    logger.info("Dataset has been downloaded")


@task
def get_dataset_last_updated():
    """Retrieve the date of the last update for the dataset and
    store it into a Prefect Date Time storage block.
    """
    logger = get_run_logger()

    # Create Kaggle client
    api = KaggleApi()
    api.authenticate()

    datasets = api.datasets_list(search=KAGGLE_DATASET_NAME)

    last_updated = None

    for dataset in datasets:
        if dataset.get("ref") == KAGGLE_DATASET_NAME:
            last_updated = dataset.get("lastUpdated")
            break

    if last_updated:
        last_updated_parsed = parse(last_updated)
        last_updated_formatted = last_updated_parsed.date().strftime(DATETIME_FORMAT)
    else:
        raise ValueError(f"The {KAGGLE_DATASET_NAME} dataset has no `lastUpdated` field")

    logger.info(f"The {KAGGLE_DATASET_NAME} dataset has been last updated on {last_updated_formatted}")

    block_last_updated = DateTime(value=last_updated_formatted)
    # With `overwrite=True` we overwrite the existing block
    uuid = block_last_updated.save(name=PREFECT_STORAGE_BLOCK_DATETIME, overwrite=True)

    logger.info(
        f"The {PREFECT_STORAGE_BLOCK_DATETIME} Date Time storage block has been created with UUID {uuid}",
    )


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
                elif k == "categories":
                    processed["categories"] = article["categories"].split(" ")
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
                    # Overwrite the original authors field
                    processed["authors"] = [
                        f'{author["first_name"]} {author["last_name"]} {author["other_name"]}'.strip()
                        for author in processed["authors_parsed"]
                    ]
                else:
                    processed[k] = v

            # Add origin
            processed["origin"] = "kaggle"
            # Add link
            latest_version = sorted(processed["versions"], key=lambda x: x["version"])[-1]["version"]
            processed["link"] = f'http://arxiv.org/abs/{processed["id"]}{latest_version}'

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
    get_dataset_last_updated()
    prepare_jsonl_for_bigquery()
    path = copy_processed_file_to_bucket()
    logger.info(f"The processed file has been stored at {path}")


if __name__ == "__main__":
    name = "Filippo is in da house"
    flow_get_arxiv_kaggle_dataset(name)
