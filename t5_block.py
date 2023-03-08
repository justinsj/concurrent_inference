import torch
from transformers import T5Model
# Load T5 model
model_name = "t5-small"
model = T5Model.from_pretrained(model_name, torchscript=True)
model.eval()

print(model)