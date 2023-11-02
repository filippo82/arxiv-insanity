from __future__ import annotations

import json
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
from prefect.blocks.system import JSON
from prefect.task_runners import SequentialTaskRunner
from prefect_gcp.cloud_storage import GcsBucket
from requests import Session
from requests.exceptions import HTTPError

VERBOSE = False

CONSTANTS = JSON.load("arxiv-block-json-constants").value

MAX_RESULTS = 1000

ARXIV_API_BASE_URL = "http://export.arxiv.org/api/query?"

TODAY = datetime.today().strftime(CONSTANTS["DATE_FORMAT"])

FN_PROCESSED = CONSTANTS["FN_PROCESSED_API"].format(TODAY)

CATEGORIES_ACTIVE_CS_SUBSET = CONSTANTS["CATEGORIES_ACTIVE_CS_SUBSET"]


@task
def load_dataset_last_updated_from_block() -> datetime:
    """Retrieve the date of the most recently updated article
    from a Prefect Date Time storage block.

    Returns
    -------
    datetime
        Date of last update of the dataset
    """
    logger = get_run_logger()

    block_last_updated = DateTime.load(name=CONSTANTS["PREFECT_STORAGE_BLOCK_DATETIME"])

    last_updated = block_last_updated.value

    logger.info(
        f"The most recent article from arXiv has been last updated on {last_updated}",
    )

    last_updated_minus_time_delta = last_updated - timedelta(hours=3)

    logger.info(
        "Just to be sure, we return the updated date of the most recent article minus 3 hours: "
        f"{last_updated_minus_time_delta}",
    )

    return last_updated_minus_time_delta


@task
def make_get_request(last_updated_date_from_block: datetime, query: str) -> int:
    """_summary_

    From https://info.arxiv.org/help/api/user-manual.html#3311-title-id-link-and-updated:

    > Because the arXiv submission process works on a 24 hour submission cycle,
    new articles are only available to the API on the midnight after the articles were processed.
    The <updated> tag thus reflects the midnight of the day that you are calling the API.
    This is very important - search results do not change until new articles are added.
    Therefore there is no need to call the API more than once in a day for the same query.
    Please cache your results. This primarily applies to production systems,
    and of course you are free to play around with the API while you are developing your program!

    Parameters
    ----------
    last_updated_date : datetime
        _description_

    query : str
        Query

    Returns
    -------
    int
        Total number of articles retrieved from the arXiv API.
    """
    logger = get_run_logger()

    session = Session()

    with open(FN_PROCESSED, "w") as fp:
        pass

    cnt = 0
    pagination_start = 0
    while True:
        logger.info(f"Page: {pagination_start}")
        query_parameter = f"search_query={query}&sortBy=lastUpdatedDate&start={pagination_start * MAX_RESULTS}&max_results={MAX_RESULTS}"  # noqa
        res = session.get(f"{ARXIV_API_BASE_URL}{query_parameter}")

        try:
            res.raise_for_status()
        except HTTPError as e:
            logger.error(f"ADD ERROR MESSAGE! {e}")

        time.sleep(1.0)

        articles = []
        res_parsed = feedparser.parse(res.text)
        entries = res_parsed.get("entries", [])
        if len(entries) == MAX_RESULTS:
            is_last_updated_set = False
            # Batch of MAX_RESULTS entries
            for entry in entries:
                categories = [category["term"] for category in entry.get("tags", [])]
                if set(categories).isdisjoint(CATEGORIES_ACTIVE_CS_SUBSET):
                    if VERBOSE:
                        logger.info(f"Skipping article with categories: {' '.join(categories)}")
                    continue
                arxiv_id = entry.get("id").split("/")[-1]
                version = arxiv_id[-2:]

                article = {
                    "id": arxiv_id,
                    "link": entry.get("id"),
                    "submitter": None,
                    "authors": [author["name"] for author in entry.get("authors", [])],
                    "title": entry.get("title", None),
                    "comments": entry.get("arxiv_comment", None),
                    "journal_ref": entry.get("journal_ref", None),
                    "doi": None,
                    "report_no": None,
                    "categories": categories,
                    "license": None,
                    "abstract": entry.get("summary", None),
                    # NOTE: the way I build `versions` is most likely not accurate.
                    # `published` indicates when the article was first published as version 1.
                    "versions": [
                        {
                            "version": version,
                            "created": entry.get("published", None),
                        },
                    ],
                    "update_date": parse(entry.get("updated", TODAY)).strftime(CONSTANTS["DATETIME_FORMAT"]),
                    "update_date_ts": int(parse(entry.get("updated", TODAY)).timestamp()),
                    # Add origin
                    "origin": "arxiv_api",
                }
                articles.append(article)

                # We save the first date of the batch because `sortBy=lastUpdatedDate`.
                # The first item is the most recent
                if not is_last_updated_set:
                    is_last_updated_set = True
                    # Add UTC time zone
                    last_updated = parse(f'{article["update_date"]}Z')
                    logger.info(f"The most recent article of the batch was updated on {last_updated}")

            with open(FN_PROCESSED, "a") as fp:
                for article in articles:
                    json.dump(article, fp)
                    fp.write("\n")

            cnt += MAX_RESULTS
            pagination_start += 1

            logger.info(
                f"I will keep going till {last_updated} is older than {last_updated_date_from_block}.",
            )
            if last_updated < last_updated_date_from_block:
                break
        else:
            logger.info(f"The request did not return {MAX_RESULTS} entries. We need to wait and try again.")
            time.sleep(10.0)

        if pagination_start > 100:
            logger.info(f"We stop anyway after making 100 requests of {MAX_RESULTS} entries each.")
            break

    return cnt


@task
def copy_processed_file_to_bucket() -> str:
    """Copy the processed file to a GCS bucket.

    Returns
    -------
    str
        URI of the processed file in the GCS bucket.
    """
    gcp_cloud_storage_bucket_block = GcsBucket.load(CONSTANTS["PREFECT_STORAGE_BLOCK_GCS_BUCKET"])
    path = gcp_cloud_storage_bucket_block.upload_from_path(FN_PROCESSED, f"arxiv-api/{FN_PROCESSED}")
    return path


@task
def save_dataset_last_updated_to_block() -> datetime:
    """Retrieve the date of the last update for the dataset and
    store it into a Prefect Date Time storage block.
    """
    with open(FN_PROCESSED) as fp:
        for line in fp:
            most_recent_article = json.loads(line)
            break

    last_updated_formatted = parse(most_recent_article["update_date"])
    last_updated_formatted_with_tz = last_updated_formatted.replace(tzinfo=timezone.utc)

    block_last_updated = DateTime(value=last_updated_formatted_with_tz)
    # With `overwrite=True` we overwrite the existing block
    _ = block_last_updated.save(name=CONSTANTS["PREFECT_STORAGE_BLOCK_DATETIME"], overwrite=True)

    return last_updated_formatted_with_tz


@task
def log_prefect_version(name: str) -> None:
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
def flow_get_metadata_from_arxiv_api(name: str = "Filippo", query: str = "cat:cs.*"):
    """Get articles metadata from arXiv API, create a JSONL file, and copy it to a GCS bucket.

    Parameters
    ----------
    name : str, optional
        The name of a dude or dudette, by default "Filippo".
    """
    logger = get_run_logger()
    log_prefect_version(name)
    last_updated_date = load_dataset_last_updated_from_block()
    total_number_of_articles = make_get_request(last_updated_date, query)
    path = copy_processed_file_to_bucket()
    last_updated_date = save_dataset_last_updated_to_block()
    logger.info(
        f"The metadata for {total_number_of_articles} articles have been retrieved from the arXiv API.",
    )
    logger.info(
        f"The most recent article from the arXiv API has been last updated on {last_updated_date}",
    )
    logger.info(f"The processed file has been stored at {path}")


if __name__ == "__main__":
    name = "Filippo is in da house"
    flow_get_metadata_from_arxiv_api(name)
