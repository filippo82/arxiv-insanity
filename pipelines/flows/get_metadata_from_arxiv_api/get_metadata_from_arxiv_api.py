from __future__ import annotations

import time
from datetime import datetime
from datetime import timedelta
from datetime import timezone

import feedparser
import prefect
from dateutil.parser import parse
from prefect import flow
from prefect import get_run_logger
from prefect import task
from prefect.blocks.system import DateTime
from prefect.task_runners import SequentialTaskRunner
from prefect_gcp.cloud_storage import GcsBucket
from requests import Session

DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
DATE_FORMAT = "%Y-%m-%d"

MAX_RESULTS = 100

ARXIV_API_BASE_URL = "http://export.arxiv.org/api/query?"

KAGGLE_DATASET_NAME = "Cornell-University/arxiv"

FN = "arxiv-metadata-oai-snapshot.json"
FN_PROCESSED = "arxiv-metadata-oai-snapshot-processed.jsonl"

PREFECT_STORAGE_BLOCK_GCS_BUCKET = "block-bucket-arxiv-data"
PREFECT_STORAGE_BLOCK_DATETIME = "block-datetime-arxiv-data-last-updated"


@task
def get_dataset_last_updated_from_block() -> datetime:
    """Retrieve the date of the last update for the dataset
    from a Prefect Date Time storage block.

    Returns
    -------
    datetime
        Date of last update of the dataset
    """
    logger = get_run_logger()

    block_last_updated = DateTime.load(name=PREFECT_STORAGE_BLOCK_DATETIME)

    last_updated_formatted = block_last_updated.value

    # Add time zone
    last_updated_formatted_with_tz = last_updated_formatted.replace(tzinfo=timezone.utc)

    logger.info(
        f"The {KAGGLE_DATASET_NAME} dataset has been last updated on {last_updated_formatted_with_tz}",
    )

    last_updated_formatted_with_tz_minus_one_week = last_updated_formatted_with_tz - timedelta(days=7)

    logger.info(
        "There is most likely a time delta between the creation date "
        "of the Kaggle dataset and the most recent updated date. "
        "Hence, we return the updated date of the dataset minues 7 days: "
        f"{last_updated_formatted_with_tz_minus_one_week}",
    )

    return last_updated_formatted_with_tz_minus_one_week


@task
def make_get_request(last_updated_from_kaggle_dataset: datetime) -> list[dict]:

    logger = get_run_logger()

    query = "cat:cs.*"

    session = Session()

    articles = []
    pagination_start = 0
    while True:
        logger.info(f"Page: {pagination_start}")
        query_parameter = f"search_query={query}&sortBy=lastUpdatedDate&start={pagination_start * MAX_RESULTS}&max_results={MAX_RESULTS}"  # noqa
        res = session.get(f"{ARXIV_API_BASE_URL}{query_parameter}")

        res.raise_for_status()

        time.sleep(0.5)

        res_parsed = feedparser.parse(res.text)
        entries = res_parsed.get("entries", [])
        if len(entries) == MAX_RESULTS:
            is_last_updated_set = False
            # Batch of MAX_RESULTS entries
            for entry in entries:

                arxiv_id = entry.get("id").split("/")[-1]
                version = arxiv_id[-2:]

                article = {
                    "id": arxiv_id,
                    "submitter": None,
                    "authors": [author["name"] for author in entry.get("authors", [])],
                    "title": entry.get("title", None),
                    "comments": entry.get("arxiv_comment", None),
                    "journal_ref": entry.get("journal_ref", None),
                    "doi": None,
                    "report_no": None,
                    "categories": [category["term"] for category in entry.get("tags", [])],
                    "license": None,
                    "abstract": entry.get("summary", None),
                    "versions": [
                        {
                            "version": version,
                            "created": entry.get("published", None),
                        },
                    ],
                    "update_date": entry.get("updated", None),
                }
                articles.append(article)

                # We save the first date of the batch because `sortBy=lastUpdatedDate`.
                # The first item is the most recent
                if not is_last_updated_set:
                    last_updated = parse(article["update_date"])
                    logger.info(f"The most recent article of the batch was updated on {last_updated}")

            pagination_start += 1

            logger.info(
                f"I will keep going till {last_updated} is older than {last_updated_from_kaggle_dataset}.",
            )
            if last_updated < last_updated_from_kaggle_dataset:
                break
        else:
            logger.info(f"The request did not return {MAX_RESULTS} entries. We need to wait and try again.")
            time.sleep(2.0)

        if pagination_start > 100:
            logger.info(f"We stop anyway after making 100 requests of {MAX_RESULTS} entries each.")
            break

    return articles


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
def flow_get_metadata_from_arxiv_api(name: str = "Filippo"):
    """Download from Kaggle, process, and copy the JSONL file to a GCS bucket.

    Parameters
    ----------
    name : str, optional
        The name of a dude or dudette, by default "Filippo".
    """
    logger = get_run_logger()
    log_prefect_version(name)
    last_updated_date = get_dataset_last_updated_from_block()
    articles = make_get_request(last_updated_date)
    # prepare_jsonl_for_bigquery()
    # path = copy_processed_file_to_bucket()
    logger.info(f"There are {len(articles)} articles ...")


if __name__ == "__main__":
    name = "Filippo is in da house"
    flow_get_metadata_from_arxiv_api(name)
