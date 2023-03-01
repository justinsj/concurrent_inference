import time
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms

def transform(pil_image, image_size):
    pil_image = pil_image.convert("RGB")
    
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])
    return preprocess(pil_image)


class CNNProvider(object):
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.ext = "JPEG"

    def load_inputs(self, inputs_path):
        inputs_list = list(Path(inputs_path).rglob(f"*.{self.ext}"))[0:100]
        return inputs_list

    def prepare_input(self, image_path):
        image = Image.open(image_path)
        image = transform(image, self.image_size)
        image.to(self.device)
        return image

    def format_input(self, input_argument):
        input_data = self.prepare_input(input_argument)
        return (input_data, input_argument, time.time())

    def inference_from_queue_message(self, model, queue_message):
        image, input_argument, start_time = queue_message
        image = torch.unsqueeze(image, 0)
        image = image.to(self.device)
        output = model(image)
        return output, input_argument, start_time

    def get_model(self):
        model = torch.jit.load(self.model_name + ".pt")
        model.eval().to(self.device)
        return model

class ResNet50Provider(CNNProvider):
    def __init__(self, model_name, device):
        super().__init__(model_name, device)
        self.image_size = 224

class CodeBertProvider:
    def prepare_input(self, string):
        code_tokens=tokenizer.tokenize(string)
        tokens=[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        tokens_ids=tokenizer.convert_tokens_to_ids(tokens)

        example = torch.tensor(tokens_ids)[None,:]
        return example

    def inference(self, model, inputs):
        model(inputs)

    def load_inputs(self, inputs_path, ext="JPEG"):
        inputs_list = list(Path(inputs_path).rglob(f"*.{ext}"))[0:100]
        return inputs_list


def get_provider(model_name, device):
    if (model_name == "codebert-base"):
        return CodeBertProvider(model_name, device)
    if (model_name == "resnet50"):
        return ResNet50Provider(model_name, device)