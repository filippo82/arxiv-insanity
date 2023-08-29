from __future__ import annotations

import json

import prefect
import torch
import zarr
from prefect import flow
from prefect import get_run_logger
from prefect import task
from prefect import variables
from prefect.task_runners import SequentialTaskRunner
from prefect_gcp.cloud_storage import GcsBucket
from sentence_transformers import SentenceTransformer

DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
DATE_FORMAT = "%Y-%m-%d"

env = variables.get("arxiv_env", "DEV")

if env == "PROD":
    FN_PROCESSED = "arxiv-metadata-oai-snapshot-processed.jsonl"
    FN_EMBEDDINGS = "kaggle-embeddings.zarr"  # This is actually a directory
else:
    FN_PROCESSED = "arxiv-metadata-oai-snapshot-processed-1000.jsonl"
    FN_EMBEDDINGS = "kaggle-embeddings-1000.zarr"  # This is actually a directory

PREFECT_STORAGE_BLOCK_GCS_BUCKET = "arxiv-block-bucket-data"

# https://huggingface.co/sentence-transformers/allenai-specter
# TRANSFORMERS_CHECKPOINT = "sentence-transformers/allenai-specter"
# WORD_EMBEDDING_DIMENSION = 768
# https://huggingface.co/intfloat/e5-small-v2
TRANSFORMERS_CHECKPOINT = "intfloat/e5-small-v2"
WORD_EMBEDDING_DIMENSION = 384

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@task
def download_processed_file_from_bucket() -> None:
    """Download the processed file from a GCS bucket.

    Returns
    -------
    str
        URI of the processed file in the GCS bucket.
    """
    gcp_cloud_storage_bucket_block = GcsBucket.load(PREFECT_STORAGE_BLOCK_GCS_BUCKET)
    _ = gcp_cloud_storage_bucket_block.download_object_to_path(f"kaggle/{FN_PROCESSED}", FN_PROCESSED)


@task
def compute_embeddings(transformers_batch_size: int) -> None:
    """Compute embeddings for the abstract.

    Parameters
    ----------
    transformers_batch_size : int
        The batch size used for the computation of the embeddings.
    """
    logger = get_run_logger()

    logger.info(f"The device is: {DEFAULT_DEVICE}")

    model = SentenceTransformer(TRANSFORMERS_CHECKPOINT, device=DEFAULT_DEVICE)

    with open(FN_PROCESSED) as f:
        line_offset = []
        offset = 0
        for line in f:
            line_offset.append(offset)
            offset += len(line)

    num_articles = len(line_offset)
    logger.info(f"The JSONL file contains a total of {num_articles:7d} articles")

    # Empirical values for batch/chunk sizes
    zarr_chunk_size = 384
    number_of_batches_to_process = zarr_chunk_size * 1

    store = zarr.DirectoryStore(FN_EMBEDDINGS)

    z = zarr.open(
        store=store,
        mode="w",
        shape=(num_articles, WORD_EMBEDDING_DIMENSION),
        chunks=(zarr_chunk_size, None),
        dtype="float32",
    )

    log_step = 1  # Number of batches

    # line_start = 0

    logger.info("Start processing JSONL file")
    with open(FN_PROCESSED) as f:
        cnt = 1
        cnt_batch = 1
        batch = []
        for line in f:
            abstract = json.loads(line).get("abstract", "")
            if TRANSFORMERS_CHECKPOINT == "intfloat/e5-small-v2":
                abstract = f"passage: {abstract}"
            batch.append(abstract)
            if cnt % (transformers_batch_size * number_of_batches_to_process) == 0:
                logger.info(f"Number of articles processed so far: {cnt}")
                logger.info(f"Number of batches processed so far: {cnt_batch}")
                logger.info(f"Number of articles in the current batch: {len(batch)}")
                normalize_embeddings = True if TRANSFORMERS_CHECKPOINT == "intfloat/e5-small-v2" else False

                embeddings = model.encode(
                    batch,
                    batch_size=transformers_batch_size,
                    show_progress_bar=True,
                    normalize_embeddings=normalize_embeddings,  # Required by `intfloat/e5-small-v2`
                )

                logger.info(f"The shape of the emebddings is: {embeddings.shape}")

                # embeddings = np.zeros([32, 768])
                batch = []
                start = (cnt_batch - 1) * (transformers_batch_size * number_of_batches_to_process)
                end = cnt_batch * (transformers_batch_size * number_of_batches_to_process)
                logger.info(f"{start}:{end}")
                z[start:end] = embeddings

                cnt_batch += 1

            if cnt % (log_step * transformers_batch_size * number_of_batches_to_process) == 0:
                logger.info(
                    f"Processed {int(cnt / transformers_batch_size):5d} "
                    f"batches of {transformers_batch_size} articles each "
                    f"for a total of {cnt:7d} articles.",
                )

            cnt += 1

        if batch:
            embeddings = model.encode(batch, batch_size=transformers_batch_size, show_progress_bar=True)
            start = (cnt_batch - 1) * (transformers_batch_size * number_of_batches_to_process)
            z[start:] = embeddings

            cnt_batch += 1

        logger.info(f"Processed a total of {cnt:7d} articles")

        zarr.consolidate_metadata(store)

        logger.info("Consolidated zarr store")


@task
def copy_zarr_directory_to_bucket() -> str:
    """Copy the zarr directory to a GCS bucket.

    Returns
    -------
    str
        URI of the zrr file in the GCS bucket.
    """
    gcp_cloud_storage_bucket_block = GcsBucket.load(PREFECT_STORAGE_BLOCK_GCS_BUCKET)
    path = gcp_cloud_storage_bucket_block.upload_from_folder(
        from_folder=FN_EMBEDDINGS,
        to_folder=f"kaggle/{FN_EMBEDDINGS}",
    )
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
def flow_compute_embeddings_for_kaggle_dataset(name: str = "Filippo", transformers_batch_size: int = 384):
    """Download from Kaggle, process, and copy the JSONL file to a GCS bucket.

    Parameters
    ----------
    name : str, optional
        The name of a dude or dudette, by default "Filippo".
    """
    logger = get_run_logger()
    log_prefect_version(name)
    download_processed_file_from_bucket()
    compute_embeddings(transformers_batch_size)
    path = copy_zarr_directory_to_bucket()
    logger.info(f"The embeddings have been computed and stored at {path}")


if __name__ == "__main__":
    name = "Filippo is in da house"
    flow_compute_embeddings_for_kaggle_dataset(name)
