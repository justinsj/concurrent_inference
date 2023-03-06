import sys
from transformers  import RobertaTokenizerFast
from torch.nn import ModuleList
import os
import torch
import torchvision

from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
import time
from flask import Flask, request, jsonify
import json
import requests
import base64
import io
import asyncio
from PIL import Image
import numpy as np
import aiohttp
import argparse
import math
import pandas as pd

DEFAULT_MAX_DEPTH = 3
        
class SplitModel(torch.nn.Module):
    def __init__(self, model_name, model, device):
        super().__init__()
        self.model_name = model_name
        self.model = model
        self.device = device

        # self.traced_model = torch.jit.trace(model, example)

        # self.modules = torch.nn.Sequential(*list(model.modules()))
        # self.modules = list(model.children())
        # self.list

        self.list_of_modules = self.get_list_of_modules()
        self.total_modules = len(self.list_of_modules)
    def get_input_data(self, module_idx, example):
        raise NotImplementedError(f"get_input_data not implemented for {self.__class__.__name__}")

    def get_list_of_modules(self):
        return self.load_list_of_modules(self.model, DEFAULT_MAX_DEPTH)
    def get_children(self, module):
        return list(module.children())
    def load_list_of_modules(self, module, remaining_depth = 0, acc = []):
        if remaining_depth <= 0:
            print(f"Adding {module.__class__.__name__} to list of modules")
            acc.append(module)
            return acc
        children = self.get_children(module)
        if (len(children) == 0):
            print(f"Adding {module.__class__.__name__} to list of modules")
            acc.append(module)
            return acc
        else:
            for child in children:
                # Add self if it does not have any children
                self.load_list_of_modules(child, remaining_depth - 1, acc)
            return acc
    
    def get_temp_name(self, module_idx):
        return f"{self.model_name}_{module_idx}.pt"
        
    def trace(self, module_idx, example, strict=True):
        module_part = self.list_of_modules[module_idx]
        print(f" Example: {example}")
        traced_model = torch.jit.trace(module_part, example, strict=strict)
        return traced_model
    
    def inference(self, module_idx, example):
        module_part = self.list_of_modules[module_idx]
        if (module_part.__class__.__name__ == "AlbertLayerGroup"):
            head_mask = [None] * self.model.config.num_hidden_layers
            return module_part(example, head_mask = head_mask)
        return module_part(example)

    def format_input(self, module_idx, example):
        return example
    def get_next_input(self, module_idx, example):
        example = self.inference(module_idx, example)
        return example
    
    def reset(self):
        pass

class CNNSplitModel(SplitModel):
    def format_input(self, module_idx, example):
        if (module_idx == 0):
            example = torch.unsqueeze(example, 0)
            example = example.to(self.device)
        return example
    def get_input_data(self, module_idx, example):
        input_shape = list(example.shape)
        input_bytes = example.nelement() * example.element_size()
        return input_shape, input_bytes
class ResNetSplitModel(CNNSplitModel):
    def get_next_input(self, module_idx, example):
        example = self.list_of_modules[module_idx](example)
        
        if module_idx == self.total_modules - 2:
            return example.view(example.size(0), -1)

        return example
    def get_list_of_modules(self):
        return self.load_list_of_modules(self.model, 1)

class ViTSplitModel(CNNSplitModel):
    def get_next_input(self, module_idx, example):
        example = self.list_of_modules[module_idx](example)

        if module_idx == 1:
            example = example.permute(0, 2, 3, 1)
            example = example.reshape(1369, 1, 1280)
        
        print(f"Module {module_idx} shape: {example.shape}")
        if module_idx == self.total_modules - 2:
            example = example[0]

        if module_idx == self.total_modules - 1:
            example = example.reshape(1, 1000)

        return example
class LMSplitModel(SplitModel):
    def __init__(self, model_name, model_tokenizer, device):
        model, tokenizer = model_tokenizer
        super().__init__(model_name, model, device)
        self.tokenizer = tokenizer

    def get_input_data(self, module_idx, example):
        input_shape = [len(example)]
        input_bytes = sys.getsizeof(example)
        return input_shape, input_bytes
    
class CodeBertSplitModel(LMSplitModel):
    def get_children(self, module):
        children = []
        if type(module) == tuple:
            children = module
        elif (type(module) == RobertaTokenizerFast):
            return []
        elif (module.__class__.__name__ == 'RobertaEmbeddings'):
            return []
        else:
            children = module.children()
        return list(children)
    
    def format_input(self, module_idx, example):
        tokenizer = self.tokenizer
        if (module_idx == 0):
            code_tokens=tokenizer.tokenize(example)
            tokens=[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
            tokens_ids=tokenizer.convert_tokens_to_ids(tokens)

            example = torch.tensor(tokens_ids)[None,:]
            example = example.to(self.device)
        
        if (module_idx in [2,3,4,5,6,7,8,9,10,11,12,13]):
            example = example[0]
        return example

class AlbertSplitModel(LMSplitModel):
    def __init__(self, model_name, model_tokenizer, device):
        super().__init__(model_name, model_tokenizer, device)
        self.dtype = torch.float32
        self.original_example = None
    def get_list_of_modules(self):
        print(f"Model: {self.model}")
        return self.load_list_of_modules(self.model, 3)
    def get_children(self, module):
        children = []
        print(f"Module name: {module.__class__.__name__}, type: {type(module)}")
        if type(module) == tuple:
            children = module
        elif (type(module) == RobertaTokenizerFast):
            return []
        elif ('Embedding' in module.__class__.__name__):
            return []
        else:
            children = module.children()
        return list(children)
    
    def get_input_data(self, module_idx, example):
        input_shape = [len(example)]
        input_bytes = sys.getsizeof(example)
        return input_shape, input_bytes

    def format_input(self, module_idx, example):
        tokenizer = self.tokenizer
        if (module_idx == 0):
            # example = torch.tensor(tokenizer.encode(example, add_special_tokens=True)).unsqueeze(0)
            # example = example.to(self.device)
            example = tokenizer.encode(example, return_tensors="pt")
            example = example.to(self.device)
        
        # if (module_idx in [2,3,4,5,6,7,8,9,10,11,12,13]):
        #     example = example[0]

        if (module_idx == 3):
            return example[0]
        return example
    
    def trace(self, module_idx, example, strict=True):
        module_part = self.list_of_modules[module_idx]
        print(f" Example: {example}")
        if (module_part.__class__.__name__ == "AlbertLayerGroup"):
            head_mask = [0] * self.model.config.num_hidden_layers
            # Convert head_mask to tensor
            head_mask = torch.tensor(head_mask)
            head_mask = head_mask.to(self.device)
            input_shape = example.size()[:-1]
            print(f"Input shape: {input_shape}")
            attention_mask = torch.ones(input_shape, device=self.device)
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min
            traced_model = torch.jit.trace(module_part, (example, extended_attention_mask, head_mask), strict=strict)
            return traced_model
        traced_model = torch.jit.trace(module_part, example, strict=strict)
        return traced_model
    
class T5SplitModel(LMSplitModel):
    def __init__(self, model_name, model_tokenizer, device):
        super().__init__(model_name, model_tokenizer, device)
        self.original_example = None
        
    def get_list_of_modules(self):
        print(f"Model: {self.model}")
        return self.load_list_of_modules(self.model, 3)
    def get_children(self, module):
        children = []
        print(f"Module name: {module.__class__.__name__}, type: {type(module)}")
        if type(module) == tuple:
            children = module
        elif (type(module) == RobertaTokenizerFast):
            return []
        elif ('Embedding' in (module.__class__.__name__)):
            return []
        else:
            if (module.__class__.__name__ == 'T5ForConditionalGeneration'):
                print(f" Found T5ForConditionalGeneration")
                children = list(module.children())[1:] # Skip the first shared module
            else:
                children = module.children()
        return list(children)
    
    def get_input_data(self, module_idx, example):
        input_shape = [len(example)]
        input_bytes = sys.getsizeof(example)
        return input_shape, input_bytes

    def format_input(self, module_idx, example):
        tokenizer = self.tokenizer
        module_part = self.list_of_modules[module_idx]
        if (module_part.__class__.__name__ == 'Embedding'):
            if (self.original_example == None):
                self.original_example = example
            else:
                example = self.original_example
            example = tokenizer.encode(example, return_tensors="pt")
            example = example.to(self.device)

        print(f" Example in format ({example.__class__.__name__}): {example}")
        cur_name = module_part.__class__.__name__

        if ((cur_name in ['T5Block','T5LayerNorm']) and self.list_of_modules[module_idx-1].__class__.__name__ == 'T5Block'):
            return example[0]
        return example
    
    def reset(self):
        self.original_example = None

    
def get_split_model(model_name, model, device):
    if model_name == "resnet50":
        return ResNetSplitModel(model_name, model, device)
    if model_name == "vit_h14_in1k":
        return ViTSplitModel(model_name, model, device)
    if model_name in ['codebert_base','DialoGPT_large','bart_large','gpt2_xl']:
        return CodeBertSplitModel(model_name, model, device)
    if model_name == "albert_xxlarge_v2":
        return AlbertSplitModel(model_name, model, device)
    if model_name in ['t5_3B','t5_small']:
        return T5SplitModel(model_name, model, device)
    raise Exception(f"Model {model_name} not found in get_split_model")