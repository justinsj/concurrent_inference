import os
import time
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, AutoConfig

from split_model import get_split_model

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




class Provider(object):
    def __init__(self, model_folder, model_name, device, max_count):
        self.model_folder = model_folder
        self.model_name = model_name
        self.device = device
        self.ext = "JPEG"
        self.max_count = max_count
        self.split_model = self.get_split_model()

    def get_split_model(self):
        model_name = self.model_name
        model = self.get_model()
        device = self.device
        split_model = get_split_model(model_name, model, device)
        return split_model
    
    def load_inputs(self, inputs_path):
        raise NotImplementedError(f"load_inputs() not implemented for {self.__class__.__name__}")

    def prepare_input(self, image_path):
        raise NotImplementedError(f"prepare_input() not implemented for {self.__class__.__name__}")

    def format_input(self, input_argument):
        input_data = self.prepare_input(input_argument)
        return (input_data, input_argument, time.time())

    def inference_from_queue_message(self, model, queue_message):
        raise NotImplementedError(f"inference_from_queue_message() not implemented for {self.__class__.__name__}")

    def get_model(self):
        model_name = self.model_name.replace('-', '_')
        model_path = os.path.join(self.model_folder, model_name + ".pt")
        print(f"Loading model from {model_path}...")
        model = torch.jit.load(os.path.join(model_path))
        model.eval()
        model = model.to(self.device)
        return model

    def get_next_input(self, module_idx, example):
        return example
    
    def start_load_modules(self):
        self.load_list_of_modules(self.model, 3)

    def get_inputs_folder(self):
        raise NotImplementedError(f"get_inputs_folder() not implemented for {self.__class__.__name__}")

class CNNProvider(Provider):
    def get_inputs_folder(self):
        return './input_folder'
    
    def load_inputs(self, inputs_path):
        inputs_list = list(Path(inputs_path).rglob(f"*.{self.ext}"))[0:self.max_count]
        return inputs_list

    def prepare_input(self, image_path):
        image = Image.open(image_path)
        image = transform(image, self.image_size)
        image = image.to(self.device)
        return image

    def inference_from_queue_message(self, model, queue_message):
        image, input_argument, start_time = queue_message
        image = torch.unsqueeze(image, 0)
        image = image.to(self.device)
        output = model(image)
        return output, input_argument, start_time

class ResNet50Provider(CNNProvider):
    def __init__(self, model_folder, model_name, device, max_count):
        super().__init__(model_folder, model_name, device, max_count)
        self.image_size = 224

    def load_inputs(self, input_folder):
        inputs_list = list(Path(input_folder).rglob(f"*.{self.ext}"))[0:self.max_count]
        return inputs_list
class ViTProvider(CNNProvider):
    def __init__(self, model_folder, model_name, device, max_count):
        super().__init__(model_folder, model_name, device, max_count)
        self.image_size = 518
    def get_model(self):
        model_name = self.model_name.replace('-', '_')
        model_path = os.path.join(self.model_folder, model_name + ".pt")
        # if not os.path.exists(model_path):
        model = torch.hub.load("facebookresearch/swag", model=model_name)
        # else:
        #     model = torch.jit.load(model_path)
        model.eval()
        model = model.to(self.device)
        return model
    
class LMProvider(Provider):
    def get_inputs_folder(self):
        return '../measure-lms/codebert/clean_inputs.csv'
    
    def prepare_input(self, string):
        return string
    def load_inputs(self, inputs_path, ext="JPEG"):
        inputs_list = pd.read_csv(inputs_path)['input'].tolist()[:self.max_count]
        return inputs_list
class CodeBertProvider(LMProvider):
    number_inferences = 0

    def inference_from_queue_message(self, model_tokenizer, queue_message):
        # CodeBertProvider.number_inferences += 1
        # print(f"Number of inferences: {CodeBertProvider.number_inferences}")
        input_string, input_argument, start_time = queue_message
        model, tokenizer = model_tokenizer
        code_tokens=tokenizer.tokenize(input_string)
        tokens=[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        tokens_ids=tokenizer.convert_tokens_to_ids(tokens)

        example = torch.tensor(tokens_ids)[None,:]
        example = example.to(self.device)
        output = model(example)
        
        return output, input_argument, start_time

    def get_model(self):
        model_name = self.model_name.replace('-', '_')
        model_path = os.path.join(self.model_folder, model_name + ".pt")
        print(f"Loading model from {model_path}...")
        model = torch.jit.load(os.path.join(model_path))
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        model = model.to(self.device)

        return (model, tokenizer)

class AlbertProvider(LMProvider):
    number_inferences = 0

    def inference_from_queue_message(self, model_tokenizer, queue_message):
        model, tokenizer = model_tokenizer
        input_string, input_argument, start_time = queue_message

        # example = tokenizer(input_string, return_tensors="pt")
        example = torch.tensor(tokenizer.encode(input_string, add_special_tokens=True)).unsqueeze(0)
        example = example.to(self.device)
        output = model(example)
        
        return output, input_argument, start_time

    def get_model(self):
        model_name = self.model_name.replace('-', '_')
        model_path = os.path.join(self.model_folder, model_name + ".pt")
        print(f"Loading model from {model_path}...")
        model = torch.jit.load(os.path.join(model_path))
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name.replace('_','-'))

        
        model = model.to(self.device)

        return (model, tokenizer)


class T5Provider(LMProvider):
    number_inferences = 0

    def inference_from_queue_message(self, model_tokenizer, queue_message):
        model, tokenizer = model_tokenizer
        input_string, input_argument, start_time = queue_message

        input_ids = tokenizer(input_string, return_tensors='pt').input_ids
        attention_mask = input_ids.ne(model.config.pad_token_id).long()
        decoder_input_ids = tokenizer(input_string, return_tensors='pt').input_ids

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        decoder_input_ids = decoder_input_ids.to(self.device)
        
        output = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
        return output, input_argument, start_time

    def get_model(self):
        model_name = self.model_name.replace('-', '_')
        model_path = os.path.join(self.model_folder, model_name + ".pt")
        print(f"Loading model from {model_path}...")
        model = torch.jit.load(os.path.join(model_path))
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name.replace('_','-'))
        config = AutoConfig.from_pretrained(model_name.replace('_','-'))
        model.config = config
        model = model.to(self.device)

        return (model, tokenizer)

def get_provider(model_folder, model_name, device, max_count):
    if (model_name in ["codebert_base",'DialoGPT_large','bart_large','gpt2_xl']):
        return CodeBertProvider(model_folder, model_name, device, max_count)
    if (model_name in ['albert_xxlarge_v2']):
        return AlbertProvider(model_folder, model_name, device, max_count)
    if (model_name in ['t5_3B']):
        return T5Provider(model_folder, model_name, device, max_count)
    if (model_name == "resnet50"):
        return ResNet50Provider(model_folder, model_name, device, max_count)
    if (model_name in ['vit_h14_in1k']):
        return ViTProvider(model_folder, model_name, device, max_count)