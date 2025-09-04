import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
image = Image.open("source/tests/figure_image.jpg").convert("RGB")

processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
model = AutoModel.from_pretrained(
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    dtype=torch.float16,
    device_map="mps"
)

print(model)
print("Model parameters:", sum(p.numel() for p in model.parameters()))

inputs = processor(images=image, return_tensors="pt").to(model.device)
# print size of inputs
print("Input pixel values shape:", inputs['pixel_values'].shape)
with torch.inference_mode():
    outputs = model(**inputs)

print("Last hidden state shape:", outputs.last_hidden_state.shape)
print("Pooled output shape:", outputs.pooler_output.shape)