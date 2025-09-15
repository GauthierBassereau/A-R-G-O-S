import asyncio
import io
from typing import AsyncGenerator, Dict, Any, Optional, Tuple, List, Iterator
import aiohttp
from datasets import load_dataset
from PIL import Image
import torch
from torchvision import transforms
from source.configs import HFStreamConfig
import logging

log = logging.getLogger(__name__)

class HFAsyncImageDataLoader:
    """
    Streams rows from a HF dataset (URL/TEXT), downloads images concurrently,
    applies a transform, and yields dict batches:
        {
          "images": FloatTensor [B, C, H, W],
          "texts":  List[str],
          "urls":   List[str],
        }
    """
    def __init__(
        self,
        hf_dataset: str = "laion/laion-coco",
        split: str = "train",
        streaming: bool = True,
        batch_size: int = 64,
        yield_partial_final: bool = False,
        max_concurrency: int = 32,
        per_host_limit: int = 8,
        total_timeout_sec: float = 2.0,
        connect_timeout_sec: float = 2.0,
        read_timeout_sec: float = 2.0,
        retries: int = 1,
        user_agent: str = "hf-image-loader/1.0",
        ssl: bool = False,
        ttl_dns_cache: int = 300,
        url_key: str = "URL",
        text_key: str = "TEXT",
        transform=None,
    ):
        # store config values
        self.hf_dataset = hf_dataset
        self.split = split
        self.streaming = streaming
        self.batch_size = batch_size
        self.yield_partial_final = yield_partial_final
        self.max_concurrency = max_concurrency
        self.per_host_limit = per_host_limit
        self.total_timeout_sec = total_timeout_sec
        self.connect_timeout_sec = connect_timeout_sec
        self.read_timeout_sec = read_timeout_sec
        self.retries = retries
        self.user_agent = user_agent
        self.ssl = ssl
        self.ttl_dns_cache = ttl_dns_cache
        self.url_key = url_key
        self.text_key = text_key

        self.transform = transform

        ds = load_dataset(
            self.hf_dataset,
            split=self.split,
            streaming=self.streaming,
        )
        self.ds_iter = iter(ds)

    @classmethod
    def from_config(cls, cfg: HFStreamConfig, transform=None, ds_iter=None):
        """Construct from a HFStreamConfig instance, mirroring ImageDecoderTranspose style."""
        return cls(
            hf_dataset=cfg.hf_dataset,
            split=cfg.split,
            streaming=cfg.streaming,
            batch_size=cfg.batch_size,
            yield_partial_final=cfg.yield_partial_final,
            max_concurrency=cfg.max_concurrency,
            per_host_limit=cfg.per_host_limit,
            total_timeout_sec=cfg.total_timeout_sec,
            connect_timeout_sec=cfg.connect_timeout_sec,
            read_timeout_sec=cfg.read_timeout_sec,
            retries=cfg.retries,
            user_agent=cfg.user_agent,
            ssl=cfg.ssl,
            ttl_dns_cache=cfg.ttl_dns_cache,
            url_key=cfg.url_key,
            text_key=cfg.text_key,
            transform=transform,
        )

    # ---- internal async helpers ----

    async def _fetch_image(
        self,
        session: aiohttp.ClientSession,
        url: str,
        semaphore: asyncio.Semaphore,
    ) -> Optional[Tuple[torch.Tensor, str]]:
        """
        Returns (tensor, url) on success, None on failure after retries.
        """
        backoff = 0.5
        attempt = 0
        while attempt <= self.retries:
            try:
                async with semaphore:
                    async with session.get(url) as resp:
                        resp.raise_for_status()
                        data = await resp.read()

                img = Image.open(io.BytesIO(data)).convert("RGB")
                tensor = self.transform(img) if self.transform is not None else transforms.ToTensor()(img)
                return tensor, url

            except Exception:
                if attempt == self.retries:
                    return None
                await asyncio.sleep(backoff)
                backoff *= 2
                attempt += 1

    async def _iter_image_batches(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Async generator that keeps up to max_concurrency downloads in flight,
        assembling batches as results complete.
        """
        timeout = aiohttp.ClientTimeout(
            total=self.total_timeout_sec,
            connect=self.connect_timeout_sec,
            sock_read=self.read_timeout_sec,
        )
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrency,
            limit_per_host=self.per_host_limit,
            ttl_dns_cache=self.ttl_dns_cache,
            ssl=self.ssl,
        )
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={"User-Agent": self.user_agent},
            raise_for_status=False,
        ) as session:

            pending: set[asyncio.Task] = set()
            batch_imgs: List[torch.Tensor] = []
            batch_texts: List[str] = []
            batch_urls: List[str] = []

            def schedule_next() -> bool:
                try:
                    row = next(self.ds_iter)
                except StopIteration:
                    return False
                url = row[self.url_key]
                text = row.get(self.text_key, "")
                task = asyncio.create_task(self._fetch_image(session, url, semaphore))
                # Attach associated metadata for when the task completes
                task._meta = {"text": text}
                pending.add(task)
                return True

            # Prime the pipeline
            log.info("Priming pipeline with up to %d tasks...", self.max_concurrency)
            for _ in range(self.max_concurrency):
                if not schedule_next():
                    break

            while pending:
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Consume finished tasks and refill the pipeline
                for task in done:
                    result = task.result()
                    meta_text = getattr(task, "_meta", {}).get("text", "")

                    if result is not None:
                        tensor, url = result
                        batch_imgs.append(tensor)
                        batch_texts.append(meta_text)
                        batch_urls.append(url)

                        if len(batch_imgs) == self.batch_size:
                            yield {
                                "images": torch.stack(batch_imgs, dim=0),
                                "texts": batch_texts,
                                "urls": batch_urls,
                            }
                            batch_imgs, batch_texts, batch_urls = [], [], []

                    # Refill pipeline
                    schedule_next()

            # Final partial batch
            if batch_imgs and self.yield_partial_final:
                yield {
                    "images": torch.stack(batch_imgs, dim=0),
                    "texts": batch_texts,
                    "urls": batch_urls,
                }
        log.info("Finished streaming all images.")

    # ---- public sync iterator ----

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Synchronous iterator that bridges the async generator,
        so you can use this directly in your training loop:
            for batch in loader:
                ...
        """
        loop = asyncio.new_event_loop()
        try:
            agen = self._iter_image_batches().__aiter__()
            while True:
                yield loop.run_until_complete(agen.__anext__())
        except StopAsyncIteration:
            return
        finally:
            loop.close()


if __name__ == "__main__":
    from source.utils.image_transforms import make_transform
    from source.utils.utils import debug_show_img
    logging.basicConfig(level=logging.INFO)
    cfg = HFStreamConfig()
    loader = HFAsyncImageDataLoader.from_config(cfg, transform=make_transform(512))
    for batch in loader:
        imgs = batch["images"]  # [B, C, H, W]
        texts = batch["texts"]  # list[str]
        urls = batch["urls"]    # list[str]
        print(imgs.shape, texts[0] if texts else "")
        debug_show_img(imgs, i=0)
        debug_show_img(imgs, i=1)