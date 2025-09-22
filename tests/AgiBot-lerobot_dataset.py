from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

repo_id = "agibot-world/AgiBotWorld-Beta"
dataset = StreamingLeRobotDataset(repo_id)

for batch in dataset:
    print((list(batch.keys())))