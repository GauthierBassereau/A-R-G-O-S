# Dataset Laion400m metadata shards (parquet files) from HuggingFace
# ----------------------------------
# BASE = "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta"
# UUID = "5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36"
# N_SHARDS = 32
# remote_parquets = [f"{BASE}/part-{i:05d}-{UUID}-c000.snappy.parquet" for i in range(N_SHARDS)]
# ----------------------------------
import logging
import random
import socket
import time
from io import BytesIO
from typing import Iterable, Iterator, List, Optional, Sequence

import requests
import torch
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from torch.utils.data import IterableDataset
from datasets import load_dataset
from torchvision import transforms

from source.utils.image_transforms import make_transform
from source.configs import HTTPConfig, StreamConfig

logger = logging.getLogger(__name__)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# ---------- Internals ----------

def _make_session(http: HTTPConfig, pool_size_override: Optional[int] = None) -> requests.Session:
    pool = pool_size_override or http.pool_size
    sess = requests.Session()
    retry = Retry(
        total=http.total_retries,
        backoff_factor=0.1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(pool_connections=pool, pool_maxsize=pool, max_retries=retry)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.headers.update({"User-Agent": http.user_agent})
    return sess


def _fetch_and_process(url: str, session: requests.Session,
                       http: HTTPConfig, transform: transforms.Compose) -> Optional[torch.Tensor]:
    try:
        r = session.get(url, timeout=(http.connect_timeout, http.read_timeout))
        if r.status_code != 200:
            return None
        img = Image.open(BytesIO(r.content)).convert("RGB")
        x = transform(img)  # (3,H,W) -> tensor
        return x
    except (requests.RequestException, OSError, socket.timeout):
        return None
    except Exception:
        return None
    
# ---------- Dataset ----------

class DatasetStream(IterableDataset):
    """
    Stream images from remote parquet metadata (URL column) with concurrent HTTP.
    Produces individual tensors; use `batch_iterator` to batch them.
    """

    def __init__(
        self,
        parquet_urls: Sequence[str],
        stream_cfg: StreamConfig = StreamConfig(),
        http_cfg: HTTPConfig = HTTPConfig(),
    ) -> None:
        super().__init__()
        self.parquet_urls = list(parquet_urls)
        self.stream_cfg = stream_cfg
        self.http_cfg = http_cfg
        self.transform = make_transform(resize=stream_cfg.transform_size)

    def _hf_stream(self):
        return load_dataset(
            "parquet",
            data_files={"train": self.parquet_urls},
            streaming=True,
        )["train"]

    def __iter__(self) -> Iterator[torch.Tensor]:
        # Local RNG for shuffling so multiprocessing workers are independent
        rng = random.Random(self.stream_cfg.seed)

        session = _make_session(self.http_cfg, pool_size_override=self.stream_cfg.max_workers)
        executor = ThreadPoolExecutor(max_workers=self.stream_cfg.max_workers)
        ds = self._hf_stream()
        ds_iter = iter(ds)

        in_flight = set()
        shuffle_buf: List[torch.Tensor] = []
        ok = fail = 0

        def submit(url: str):
            return executor.submit(_fetch_and_process, url, session, self.http_cfg, self.transform)

        try:
            logger.info("Priming pipeline to %d requests…", self.stream_cfg.max_in_flight)
            while len(in_flight) < self.stream_cfg.max_in_flight:
                try:
                    row = next(ds_iter)
                except StopIteration:
                    break
                url = row.get(self.stream_cfg.url_col_name)
                if not url:
                    logger.debug("Skipping row with no URL column '%s'.", self.stream_cfg.url_col_name)
                    continue
                in_flight.add(submit(url))
            logger.info("Primed with %d requests.", len(in_flight))

            last_log = time.time()

            while in_flight:
                done, _ = wait(in_flight, timeout=0.25, return_when=FIRST_COMPLETED)

                for fut in done:
                    in_flight.discard(fut)
                    try:
                        x = fut.result()
                    except Exception:
                        x = None
                    if x is not None:
                        ok += 1
                        shuffle_buf.append(x)
                        if len(shuffle_buf) >= self.stream_cfg.shuffle_buffer:
                            idx = rng.randrange(len(shuffle_buf))
                            out = shuffle_buf.pop(idx)
                            yield out
                    else:
                        fail += 1

                # Top up
                try:
                    while len(in_flight) < self.stream_cfg.max_in_flight:
                        row = next(ds_iter)
                        url = row.get(self.stream_cfg.url_col_name)
                        if not url:
                            continue
                        in_flight.add(submit(url))
                except StopIteration:
                    pass

                # Periodic stats
                if time.time() - last_log > self.stream_cfg.log_interval_s:
                    logger.info("ok=%d fail=%d in_flight=%d buf=%d",
                                ok, fail, len(in_flight), len(shuffle_buf))
                    last_log = time.time()

            # Flush buffer
            rng.shuffle(shuffle_buf)
            for x in shuffle_buf:
                yield x

        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            session.close()
            logger.info("Shutdown completed. Total ok=%d fail=%d", ok, fail)


def batch_iterator(tensor_iter: Iterable[torch.Tensor], batch_size: int, drop_last: bool = True) -> Iterator[torch.Tensor]:
    """Group a stream of tensors into batches."""
    buf: List[torch.Tensor] = []
    for x in tensor_iter:
        buf.append(x)
        if len(buf) == batch_size:
            yield torch.stack(buf, dim=0)
            buf.clear()
    if not drop_last and buf:
        yield torch.stack(buf, dim=0)
