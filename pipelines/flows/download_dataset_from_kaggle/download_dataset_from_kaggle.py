from __future__ import annotations

import json
from datetime import timedelta
from datetime import timezone

import prefect
from dateutil.parser import parse
from dateutil.tz import UTC
from kaggle.api.kaggle_api_extended import KaggleApi
from prefect import flow
from prefect import get_run_logger
from prefect import task
from prefect import variables
from prefect.blocks.system import DateTime
from prefect.blocks.system import JSON
from prefect.task_runners import SequentialTaskRunner
from prefect_gcp.cloud_storage import GcsBucket

DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
DATE_FORMAT = "%Y-%m-%d"

env = variables.get("arxiv_env", "DEV")

if env == "PROD":
    KAGGLE_DATASET_NAME = variables.get("kaggle_dataset_name")

    FN = "arxiv-metadata-oai-snapshot.json"
    FN_PROCESSED = "arxiv-metadata-oai-snapshot-processed.jsonl"
else:
    KAGGLE_DATASET_NAME = variables.get("kaggle_dataset_name_dev")

    FN = "arxiv-metadata-oai-snapshot-1000.json"
    FN_PROCESSED = "arxiv-metadata-oai-snapshot-processed-1000.jsonl"

PREFECT_STORAGE_BLOCK_GCS_BUCKET = "arxiv-block-bucket-data"
PREFECT_STORAGE_BLOCK_DATETIME = "arxiv-block-datetime-data-last-updated"
PREFECT_STORAGE_BLOCK_JSON = "arxiv-block-json-dataset-version-number"


@task
def download_dataset() -> None:
    """Download the metadata of all arXiv preprints from Kaggle.
    The [arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)
    is updated on a monthly basis.

    This task assumes that the `~/.kaggle/kaggle.json` exists and
    contains the required credentials.
    """
    logger = get_run_logger()

    # Create Kaggle client
    api = KaggleApi()
    api.authenticate()

    logger.info("Start downloading dataset")
    api.dataset_download_files(KAGGLE_DATASET_NAME, quiet=False, unzip=True)
    logger.info("Dataset has been downloaded")


@task
def save_dataset_last_updated_to_block() -> None:
    """Retrieve the date of the last update for the dataset and
    store it into a Prefect Date Time storage block.

    This task assumes that the `~/.kaggle/kaggle.json` file exists and
    contains the required credentials.
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
        last_updated_parsed = parse(last_updated).astimezone(UTC)
        # last_updated_formatted = last_updated_parsed.date().strftime(DATETIME_FORMAT)
    else:
        raise ValueError(f"The {KAGGLE_DATASET_NAME} dataset has no `lastUpdated` field")

    logger.info(f"The {KAGGLE_DATASET_NAME} dataset has been last updated on {last_updated_parsed}")

    # Replace `tzinfo=tzutc()` with `tzinfo=datetime.timezone.utc`
    # This is required because `prefect` cannot handle `tzutc()`
    last_updated_parsed_with_tz = last_updated_parsed.replace(tzinfo=timezone.utc)

    last_updated_parsed_with_tz_minus_time_delta = last_updated_parsed_with_tz - timedelta(days=3)

    logger.info(
        "There is most likely a time delta between the creation date "
        "of the Kaggle dataset and the most recent updated date. "
        "Hence, we return the updated date of the dataset minus 3 days: "
        f"{last_updated_parsed_with_tz_minus_time_delta}",
    )

    block_last_updated = DateTime(value=last_updated_parsed_with_tz_minus_time_delta)
    # With `overwrite=True` we overwrite the existing block
    uuid = block_last_updated.save(name=PREFECT_STORAGE_BLOCK_DATETIME, overwrite=True)

    logger.info(
        f"The {PREFECT_STORAGE_BLOCK_DATETIME} Date Time storage block has been created with UUID {uuid}",
    )


@task
def save_dataset_version_number_to_block() -> None:
    """Retrieve the version number of the dataset and
    store it into a Prefect JSON storage block.

    This task assumes that the `~/.kaggle/kaggle.json` file exists and
    contains the required credentials.
    """
    logger = get_run_logger()

    # Create Kaggle client
    api = KaggleApi()
    api.authenticate()

    datasets = api.datasets_list(search=KAGGLE_DATASET_NAME)

    current_version_number = None

    for dataset in datasets:
        if dataset.get("ref") == KAGGLE_DATASET_NAME:
            current_version_number = dataset.get("currentVersionNumber")
            break

    if current_version_number:
        current_version_number = int(current_version_number)
    else:
        raise ValueError(f"The {KAGGLE_DATASET_NAME} dataset has no `current_version_number` field")

    logger.info(f"The {KAGGLE_DATASET_NAME} dataset's version number is {current_version_number}")

    block_version_number = JSON(value={"currentVersionNumber": current_version_number})
    # With `overwrite=True` we overwrite the existing block
    uuid = block_version_number.save(name=PREFECT_STORAGE_BLOCK_JSON, overwrite=True)

    logger.info(f"The {PREFECT_STORAGE_BLOCK_JSON} JSON storage block has been created with UUID {uuid}")


@task
def prepare_jsonl_for_bigquery() -> None:
    """Process the downloaded metadata file so that it can be directly loaded
    into BigQuery. This [comment](https://www.kaggle.com/datasets/Cornell-University/arxiv/discussion/376149)
    explains why this additional processing is required.

    The following is an example of a preprint metadata downloaded from Kaggle:

    ```json
    {
        "id": "0704.0001",
        "submitter": "Pavel Nadolsky",
        "authors": "C. Bal\\'azs, E. L. Berger, P. M. Nadolsky, C.-P. Yuan",
        "title": "Calculation of prompt diphoton production cross sections at Tevatron and\n  LHC energies",
        "comments": "37 pages, 15 figures; published version",
        "journal-ref": "Phys.Rev.D76:013009,2007",
        "doi": "10.1103/PhysRevD.76.013009",
        "report-no": "ANL-HEP-PR-07-12",
        "categories": "hep-ph",
        "license": null,
        "abstract": "  A fully differential calculation in perturbative quantum chromodynamics is\npresented for the production of massive photon pairs at hadron colliders. All\nnext-to-leading order perturbative contributions from quark-antiquark,\ngluon-(anti)quark, and gluon-gluon subprocesses are included, as well as\nall-orders resummation of initial-state gluon radiation valid at\nnext-to-next-to-leading logarithmic accuracy. The region of phase space is\nspecified in which the calculation is most reliable. Good agreement is\ndemonstrated with data from the Fermilab Tevatron, and predictions are made for\nmore detailed tests with CDF and DO data. Predictions are shown for\ndistributions of diphoton pairs produced at the energy of the Large Hadron\nCollider (LHC). Distributions of the diphoton pairs from the decay of a Higgs\nboson are contrasted with those produced from QCD processes at the LHC, showing\nthat enhanced sensitivity to the signal can be obtained with judicious\nselection of events.\n",
        "versions": [
            {
                "version": "v1",
                "created": "Mon, 2 Apr 2007 19:18:42 GMT"
            },
            {
                "version": "v2",
                "created": "Tue, 24 Jul 2007 20:10:27 GMT"
            }
        ],
        "update_date": "2008-11-26",
        "authors_parsed": [
            [
                "Balázs",
                "C.",
                ""
            ],
            [
                "Berger",
                "E. L.",
                ""
            ],
            [
                "Nadolsky",
                "P. M.",
                ""
            ],
            [
                "Yuan",
                "C. -P.",
                ""
            ]
        ]
    }
    ```

    """  # noqa
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
                elif k == "update_date":
                    processed["update_date"] = parse(v).strftime(DATETIME_FORMAT)
                    processed["update_date_ts"] = int(parse(v).timestamp())
                else:
                    processed[k] = v

            # Add origin
            processed["origin"] = "kaggle"
            # Add link
            latest_version = sorted(processed["versions"], key=lambda x: x["version"])[-1]["version"]
            processed["link"] = f'http://arxiv.org/abs/{processed["id"]}{latest_version}'
            processed["id"] = f'{processed["id"]}{latest_version}'

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
def flow_get_arxiv_kaggle_dataset(name: str = "Filippo") -> None:
    """Download from Kaggle, process, and copy the JSONL file to a GCS bucket.

    Parameters
    ----------
    name : str, optional
        The name of a dude or dudette, by default "Filippo".
    """
    logger = get_run_logger()
    log_prefect_version(name)
    download_dataset()
    save_dataset_last_updated_to_block()
    save_dataset_version_number_to_block()
    prepare_jsonl_for_bigquery()
    path = copy_processed_file_to_bucket()
    logger.info(f"The processed file has been stored at {path}")


if __name__ == "__main__":
    name = "Filippo is in da house"
    flow_get_arxiv_kaggle_dataset(name)
