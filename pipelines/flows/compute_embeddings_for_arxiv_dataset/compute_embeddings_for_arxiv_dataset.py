from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import prefect
import torch
import torch.nn.functional as F
import zarr
from optimum.onnxruntime import ORTModelForFeatureExtraction
from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime.configuration import OptimizationConfig
from prefect import flow
from prefect import get_run_logger
from prefect import task
from prefect.blocks.system import JSON
from prefect.task_runners import SequentialTaskRunner
from prefect_gcp.cloud_storage import GcsBucket
from torch import device
from torch import Tensor
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerBase

VERBOSE = False

CONSTANTS = JSON.load("arxiv-block-json-constants").value

PATH_TEMP = Path(CONSTANTS["PATH_TEMP"])

TODAY = datetime.today().strftime(CONSTANTS["DATE_FORMAT"])
FN_PROCESSED = CONSTANTS["FN_PROCESSED_API"].format(TODAY)
FN_EMBEDDINGS = CONSTANTS["FN_EMBEDDINGS_API"].format(TODAY)  # This is actually a directory

ARXIV_CATEGORIES_SUBSET = CONSTANTS["ARXIV_CATEGORIES_ACTIVE_CS_SUBSET"]

# https://huggingface.co/intfloat/e5-small-v2
# MODEL_CHECKPOINT = "intfloat/e5-small-v2"
# WORD_EMBEDDING_DIMENSION = 384
# NORMALIZE_EMBEDDINGS = True
# https://huggingface.co/BAAI/bge-small-en-v1.5
MODEL_CHECKPOINT = "BAAI/bge-small-en-v1.5"
MODEL_NAME = MODEL_CHECKPOINT.split("/")[-1]
WORD_EMBEDDING_DIMENSION = 384
NORMALIZE_EMBEDDINGS = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SentenceTransformersWithONNX:
    def __init__(self, model, tokenizer, device: device = "cpu") -> None:
        self.model = model
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.device: str = device

        self.model = self.model.to(self.device)

    # Copied from the model card: https://huggingface.co/intfloat/e5-small-v2
    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """_summary_

        Parameters
        ----------
        last_hidden_states : Tensor
            _description_
        attention_mask : Tensor
            _description_

        Returns
        -------
        Tensor
            _description_
        """
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def encode(
        self,
        batch_of_sentences: str | list[str],
        batch_size: int = 1,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = False,
    ):
        """_summary_

        Parameters
        ----------
        batch_of_sentences : Union[str, List[str]]
            _description_
        batch_size : int, optional
            _description_, by default 1
        show_progress_bar : bool, optional
            _description_, by default False
        normalize_embeddings : bool, optional
            _description_, by default False
        """

        # Tokenize the input texts and move the output to the right `DEVICE`
        batch_dict = self.tokenizer(
            batch_of_sentences,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            # Perform inference
            outputs = self.model(**batch_dict)

            # Perform pooling
            embeddings = self.average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])

            # Normalize embeddings
            if normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)

        # Move output to `cpu` and convert to `numpy` before assigning to the `zarr` array
        return embeddings.cpu().numpy()


@task
def create_onnx_model() -> None:
    """Create ONNX model."""
    logger = get_run_logger()
    onnx_path = PATH_TEMP / f"{MODEL_NAME}-onnx"

    model_onnx = ORTModelForFeatureExtraction.from_pretrained(MODEL_CHECKPOINT, export=True)
    model_onnx.save_pretrained(onnx_path)

    # Create ORTOptimizer and define optimization configuration
    optimize_for_gpu = True if DEVICE == "cuda" else False
    optimizer = ORTOptimizer.from_pretrained(model_onnx)
    optimization_config = OptimizationConfig(
        optimization_level=99,  # enable all optimizations
        optimize_for_gpu=optimize_for_gpu,
        # fp16=True,
    )

    # Apply the optimization configuration to the model
    optimizer.optimize(
        save_dir=onnx_path,
        optimization_config=optimization_config,
    )

    model_optimized = ORTModelForFeatureExtraction.from_pretrained(
        onnx_path,
        file_name="model_optimized.onnx",
    )

    if DEVICE != "cuda":
        # Create ORTQuantizer and define quantization configuration
        quantizer = ORTQuantizer.from_pretrained(model_optimized)
        quantization_config = AutoQuantizationConfig.avx2(is_static=False, per_channel=True)
        # quantization_config = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
        quantizer.quantize(save_dir=onnx_path, quantization_config=quantization_config)
        # model_quantized = ORTModelForFeatureExtraction.from_pretrained(
        #     onnx_path,
        #     file_name="model_optimized_quantized.onnx",
        # )

    logger.info("List created files:")
    for file in PATH_TEMP.rglob("*"):
        logger.info(file)


@task
def download_processed_file_from_bucket() -> None:
    """Download the processed file from a GCS bucket.

    Returns
    -------
    str
        URI of the processed file in the GCS bucket.
    """
    gcp_cloud_storage_bucket_block = GcsBucket.load(CONSTANTS["PREFECT_STORAGE_BLOCK_GCS_BUCKET"])
    _ = gcp_cloud_storage_bucket_block.download_object_to_path(f"arxiv-api/{FN_PROCESSED}", PATH_TEMP / FN_PROCESSED)


@task
def compute_embeddings(transformers_batch_size: int) -> None:
    """Compute embeddings for the abstract.

    Parameters
    ----------
    transformers_batch_size : int
        The batch size used for the computation of the embeddings.
    """
    logger = get_run_logger()

    logger.info(f"The device is: {DEVICE}")

    onnx_path = PATH_TEMP / f"{MODEL_NAME}-onnx"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    if DEVICE == "cpu":
        # Optimized model
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(
            onnx_path,
            file_name="model_optimized_quantized.onnx",
            use_io_binding=True,
        )
        model = SentenceTransformersWithONNX(model=onnx_model, tokenizer=tokenizer, device=DEVICE)
        # Normal model
        # model = SentenceTransformer(MODEL_CHECKPOINT, device=DEVICE)
    elif DEVICE == "cuda":
        model = ORTModelForFeatureExtraction.from_pretrained(
            onnx_path,
            file_name="model_optimized.onnx",
            use_io_binding=True,
        ).to(DEVICE)

    with open(PATH_TEMP / FN_PROCESSED) as f:
        line_offset = []
        offset = 0
        for line in f:
            line_offset.append(offset)
            offset += len(line)

    num_articles = len(line_offset)
    logger.info(f"The JSONL file contains a total of {num_articles:7d} articles")

    # Empirical values for batch/chunk sizes
    zarr_chunk_size = WORD_EMBEDDING_DIMENSION * 2
    # number_of_batches_to_process = zarr_chunk_size * 1
    number_of_batches_to_process = 1

    store = zarr.DirectoryStore(PATH_TEMP / FN_EMBEDDINGS)

    # The final shape of the `zarr` array will be:
    # (num_articles, WORD_EMBEDDING_DIMENSION)
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
    with open(PATH_TEMP / FN_PROCESSED) as f:
        cnt = 1
        cnt_batch = 1
        batch = []
        start_time = perf_counter()
        for line in f:
            categories = json.loads(line).get("categories", [])
            if set(categories).isdisjoint(ARXIV_CATEGORIES_SUBSET):
                if VERBOSE:
                    logger.info(f"Skipping article with categories: {' '.join(categories)}")
                continue
            abstract = json.loads(line).get("abstract", "")
            if MODEL_CHECKPOINT == "intfloat/e5-small-v2":
                abstract = f"passage: {abstract}"
            batch.append(abstract)
            if cnt % (transformers_batch_size * number_of_batches_to_process) == 0:
                logger.info(f"Number of articles processed so far: {cnt}")
                logger.info(f"Number of batches processed so far: {cnt_batch}")
                logger.info(f"Number of articles in the current batch: {len(batch)}")

                embeddings = model.encode(
                    batch,
                    batch_size=transformers_batch_size,
                    show_progress_bar=True,
                    normalize_embeddings=NORMALIZE_EMBEDDINGS,
                )

                # The shape of `embeddings` is:
                # (num_articles_in_the_batch, WORD_EMBEDDING_DIMENSION)
                logger.info(f"The shape of the emebddings is: {embeddings.shape}")

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
                latency = perf_counter() - start_time
                logger.info(f"latency: {latency:.2f}")
                start_time = perf_counter()

            cnt += 1

        if batch:
            embeddings = model.encode(
                batch,
                batch_size=transformers_batch_size,
                show_progress_bar=True,
                normalize_embeddings=NORMALIZE_EMBEDDINGS,
            )
            logger.info(f"The shape of the emebddings is: {embeddings.shape}")

            start = (cnt_batch - 1) * (transformers_batch_size * number_of_batches_to_process)
            end = start + len(batch)
            logger.info(f"{start}:{end}")
            z[start:end] = embeddings

            cnt_batch += 1

        logger.info(f"Processed a total of {end:7d} articles")

        z.resize(end, WORD_EMBEDDING_DIMENSION)
        zarr.consolidate_metadata(store)

        logger.info("Consolidated zarr store")


# @task
# def copy_zarr_directory_to_bucket() -> str:
#     """Copy the zarr directory to a GCS bucket.

#     Returns
#     -------
#     str
#         URI of the zrr file in the GCS bucket.
#     """
#     gcp_cloud_storage_bucket_block = GcsBucket.load(PREFECT_STORAGE_BLOCK_GCS_BUCKET)
#     path = gcp_cloud_storage_bucket_block.upload_from_folder(
#         from_folder=FN_EMBEDDINGS, to_folder=f"arxiv-api/{FN_EMBEDDINGS}"
#     )
#     return path


@task
def copy_zarr_directory_to_bucket() -> str:
    """Copy the zarr directory to a GCS bucket.

    Returns
    -------
    str
        URI of the zarr file in the GCS bucket.
    """
    logger = get_run_logger()
    gcp_cloud_storage_bucket_block = GcsBucket.load(CONSTANTS["PREFECT_STORAGE_BLOCK_GCS_BUCKET"])

    cnt = 0
    for fn in (PATH_TEMP / FN_EMBEDDINGS).glob("*"):
        fn2 = fn.relative_to(PATH_TEMP)
        logger.info(f"Copying {fn} to {fn2}")

        path = gcp_cloud_storage_bucket_block.upload_from_path(from_path=fn, to_path=f"arxiv-api/{fn2}")
        cnt += 1
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
def flow_compute_embeddings_for_arxiv_dataset(name: str = "Filippo", transformers_batch_size: int = 48):
    """Download from Kaggle, process, and copy the JSONL file to a GCS bucket.

    Parameters
    ----------
    name : str, optional
        The name of a dude or dudette, by default "Filippo".
    """
    logger = get_run_logger()
    log_prefect_version(name)
    create_onnx_model()
    download_processed_file_from_bucket()
    compute_embeddings(transformers_batch_size)
    path = copy_zarr_directory_to_bucket()
    logger.info(f"The embeddings have been computed and stored at {path}")


if __name__ == "__main__":
    name = "Filippo is in da house"
    transformers_batch_size: int = 48
    flow_compute_embeddings_for_arxiv_dataset(name=name, transformers_batch_size=transformers_batch_size)
