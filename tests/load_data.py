from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler


ds = LeRobotDataset(
    repo_id="lerobot/svla_so100_pickplace",
    download_videos=True,          # skip mp4s at first if you only test the code path
    force_cache_sync=True,          # ensure fresh meta from the Hub
)
print(ds)

dataset = make_dataset(cfg)

dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )