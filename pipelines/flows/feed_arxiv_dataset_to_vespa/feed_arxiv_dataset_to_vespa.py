from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import gcsfs
import prefect
from prefect import flow, get_run_logger, task
from prefect.blocks.system import JSON
from prefect.task_runners import SequentialTaskRunner
from prefect_gcp.cloud_storage import GcsBucket
from prefect_gcp.secret_manager import GcpSecret
from vespa.application import Vespa
from vespa.io import VespaResponse
from zarr import Array as ZarrArray
from zarr import open_consolidated

VERBOSE = False

CONSTANTS = JSON.load("arxiv-block-json-constants").value

PATH_TEMP = Path(CONSTANTS["PATH_TEMP"])
PATH_TEMP.mkdir(exist_ok=True)

TODAY = datetime.today().strftime(CONSTANTS["DATE_FORMAT"])
FN_PROCESSED = CONSTANTS["FN_PROCESSED_API"].format(TODAY)
FN_EMBEDDINGS = CONSTANTS["FN_EMBEDDINGS_API"].format(TODAY)  # This is actually a directory

ARXIV_CATEGORIES_SUBSET = {category.lower() for category in CONSTANTS["ARXIV_CATEGORIES_ACTIVE_CS_SUBSET"]}

WORD_EMBEDDING_DIMENSION = 384
ZARR_CHUNK_SIZE = WORD_EMBEDDING_DIMENSION * 2


@task
def get_vespa_endpoint() -> str:
    """Retrieve Vespa endpoint from Google Cloud Secret Manager.

    Returns
    -------
    str
        Vespa end point (http://address:port).
    """
    gcpsecret_block = GcpSecret.load(CONSTANTS["ARXIV_BLOCK_GCP_SECRET_VESPA_ENDPOINT"])

    gcpsecret_block_value = gcpsecret_block.read_secret().decode("utf-8")

    return gcpsecret_block_value


@task
def download_processed_file_from_bucket() -> str:
    """Download the processed file from a GCS bucket.

    Returns
    -------
    str
        URI of the processed file in the GCS bucket.
    """
    gcp_cloud_storage_bucket_block = GcsBucket.load(CONSTANTS["PREFECT_STORAGE_BLOCK_GCS_BUCKET"])
    fn_processed = gcp_cloud_storage_bucket_block.download_object_to_path(
        f"arxiv-api/{FN_PROCESSED}",
        PATH_TEMP / FN_PROCESSED,
    )
    return fn_processed


@task
def get_embeddings_as_zarr_array_from_bucket() -> ZarrArray:
    """Download `zarr` array with embeddings from Google Cloud Storage.

    Returns
    -------
    ZarrArray
        Input `zarr` array with embeddings.
    """

    logger = get_run_logger()

    gcp_cloud_storage_bucket_block = GcsBucket.load(CONSTANTS["PREFECT_STORAGE_BLOCK_GCS_BUCKET"])
    bucket = gcp_cloud_storage_bucket_block.bucket
    bucket_folder = gcp_cloud_storage_bucket_block.bucket_folder.replace("/", "")

    gcs = gcsfs.GCSFileSystem()
    logger.info(f"gs://{bucket}/{bucket_folder}/arxiv-api/{FN_EMBEDDINGS}")
    store = gcs.get_mapper(
        root=f"gs://{bucket}/{bucket_folder}/arxiv-api/{FN_EMBEDDINGS}",
    )
    z = open_consolidated(store=store, mode="r")
    logger.info(f"The Zarr array of the embeddings has shape {z.shape}")

    return z


@task
def feed_data_to_vespa(vespa_url: str, fn_processed: str, z: ZarrArray) -> int:
    """Feed data to Vespa

    Parameters
    ----------
    vespa_url : str
        Vespa end point (address:port).
    fn_processed : str
        Input file with metadata.
    z : ZarrArray
        Input `zarr` array with embeddings.

    Returns
    -------
    int
        _description_
    """
    logger = get_run_logger()

    def callback(response: VespaResponse, id: str):
        if not response.is_successful():
            print(f"Error when feeding document {id}: {response.get_json()}")

    vespa = Vespa(url=vespa_url)

    # This value comes from the `batch_size` used to compute embeddings with a V100 GPU
    batch_size_zarr_to_read = ZARR_CHUNK_SIZE * 50

    batch_size_vespa_feed = 500

    log_step = batch_size_vespa_feed * 10

    restart_from = -1
    # restart_from = 2_000_000

    logger.info("Start feeding documents to Vespa")
    with open(fn_processed) as f:
        cnt = 0
        cnt_to_ingest = 0
        cnt_skipped = 0
        cnt_embedding = 0
        batch_documents = []
        batch_embeddings = z[:batch_size_zarr_to_read]
        for line in f:
            if cnt >= restart_from:
                article = json.loads(line)
                # logger.info(f"cnt:{cnt}")
                if ARXIV_CATEGORIES_SUBSET.isdisjoint(
                    [category.lower() for category in article["categories"]],
                ):
                    # logger.info(f"Skipping article {article['id']} with categories {article['categories']}")
                    cnt_skipped += 1
                else:
                    # del article["authors_parsed"]
                    # https://blog.marvik.ai/2022/11/17/how-to-quickly-implement-a-text-search-system-using-pyvespa/
                    article["abstract_embedding"] = {
                        "values": batch_embeddings[cnt_embedding].tolist(),
                    }
                    batch_documents.append(
                        {
                            "id": article["id"],
                            "fields": article,
                        },
                    )
                    # cnt += 1
                    cnt_to_ingest += 1
                    cnt_embedding += 1

                    # del article["authors_parsed"]
                    # # https://blog.marvik.ai/2022/11/17/how-to-quickly-implement-a-text-search-system-using-pyvespa/
                    # article["abstract_embedding"] = {"values": batch_embeddings[cnt_embedding].tolist()}
                    # batch_documents.append(
                    #     {
                    #         "id": article["id"],
                    #         "fields": article,
                    #     }
                    # )
                    # cnt_ingested += 1
                    # cnt_embedding += 1

                    if cnt_embedding % batch_size_zarr_to_read == 0:
                        # logger.info(
                        #     f"Read embeddings from {cnt_to_ingest:7d}:{cnt_to_ingest + batch_size_zarr_to_read:7d}"
                        # )
                        batch_embeddings = z[cnt_to_ingest : cnt_to_ingest + batch_size_zarr_to_read]
                        cnt_embedding = 0

                    if cnt_to_ingest % (batch_size_vespa_feed * 10) == 0:
                        # logger.info(f"Processed {cnt:7d} articles")
                        logger.info(f"Batch size: {len(batch_documents):7d}")

                        _ = vespa.feed_iterable(
                            batch_documents,
                            schema="article",
                            max_queue_size=batch_size_vespa_feed,
                            callback=callback,
                        )
                        batch_documents = []

                    if cnt % log_step == 0:
                        logger.info(f"Processed {cnt:7d} articles")
                        logger.info(f"Ingested {cnt_to_ingest:7d} articles")
                        logger.info(f"Skipped {cnt_skipped:7d} articles")

            # else:
            cnt += 1

            if cnt > 2_500_000:
                break

    if batch_documents:
        # logger.info(batch_documents[0])
        _ = vespa.feed_iterable(
            batch_documents,
            schema="article",
            callback=callback,
        )

    logger.info(f"Processed a total of {cnt:7d} articles")
    logger.info(f"Ingested a total of {cnt_to_ingest:7d} articles")
    logger.info(f"Skipped a total of {cnt_skipped:7d} articles")

    return cnt_to_ingest


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
def feed_data_and_embeddings_to_vespa(name: str = "Filippo"):
    """Feed data to Vespa.

    Parameters
    ----------
    name : str, optional
        The name of a dude or dudette, by default "Filippo".
    """
    logger = get_run_logger()
    log_prefect_version(name)
    vespa_url = get_vespa_endpoint()
    fn_processed = download_processed_file_from_bucket()
    z = get_embeddings_as_zarr_array_from_bucket()
    cnt_to_ingest = feed_data_to_vespa(vespa_url, fn_processed, z)
    logger.info(f"Ingested a total of {cnt_to_ingest:7d} articles")


if __name__ == "__main__":
    name = "Filippo is in da house"
    feed_data_and_embeddings_to_vespa(name)
