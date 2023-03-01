import os
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
    def __init__(self, model_folder, model_name, device, max_count):
        self.model_folder = model_folder
        self.model_name = model_name
        self.device = device
        self.ext = "JPEG"
        self.max_count = max_count
    def load_inputs(self, inputs_path):
        inputs_list = list(Path(inputs_path).rglob(f"*.{self.ext}"))[0:self.max_count]
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
        model = torch.jit.load(os.path.join(self.model_folder, self.model_name + ".pt"))
        model.eval().to(self.device)
        return model

class ResNet50Provider(CNNProvider):
    def __init__(self, model_folder, model_name, device, max_count):
        super().__init__(model_folder, model_name, device, max_count)
        self.image_size = 224

class CodeBertProvider:
    def prepare_input(self, string):
        return string

    def inference_from_queue_message(self, model, queue_message):
        input_string, input_argument, start_time = queue_message
        code_tokens=tokenizer.tokenize(input_string)
        tokens=[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        tokens_ids=tokenizer.convert_tokens_to_ids(tokens)

        example = torch.tensor(tokens_ids)[None,:]
        example = example.to(device)
        output = model(example)
        return output, input_argument, start_time

    def load_inputs(self, inputs_path, ext="JPEG"):
        inputs_list = pd.read_csv(inputs_path)['input'].tolist()
        return inputs_list


def get_provider(model_folder, model_name, device, max_count):
    if (model_name == "codebert-base"):
        return CodeBertProvider(model_folder, model_name, device, max_count)
    if (model_name == "resnet50"):
        return ResNet50Provider(model_folder, model_name, device, max_count)