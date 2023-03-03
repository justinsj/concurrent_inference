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
        
class SplitModel(torch.nn.Module):
    def __init__(self, model_name, model):
        super().__init__()
        self.model_name = model_name
        self.model = model

        # self.traced_model = torch.jit.trace(model, example)

        # self.modules = torch.nn.Sequential(*list(model.modules()))
        # self.modules = list(model.children())
        # self.list

        self.list_of_modules = []
        self.start_load_modules()
        self.total_modules = len(self.list_of_modules)

    def start_load_modules(self):
        self.load_list_of_modules(self.model, 3)

    def load_list_of_modules(self, module, remaining_depth = 0):
        if remaining_depth == 0:
            return
        for child in module.children():
            # Add self if it does not have any children
            if len(list(child.children())) == 0 or remaining_depth == 1:
                self.list_of_modules.append(child)
            self.load_list_of_modules(child, remaining_depth - 1)
    def get_temp_name(self, module_idx):
        return f"{self.model_name}_{module_idx}.pt"
        
class ResNetSplitModel(SplitModel):
    def get_next_input(self, module_idx, example):
        example = self.list_of_modules[module_idx](example)
        
        if module_idx == self.total_modules - 2:
            return example.view(example.size(0), -1)

        return example
    def start_load_modules(self):
        self.load_list_of_modules(self.model, 1)

class ViTSplitModel(SplitModel):
    def get_next_input(self, module_idx, example):
        example = self.list_of_modules[module_idx](example)

        if module_idx == 1:
            example = example.permute(0, 2, 3, 1)
            example = example.reshape(1369, 1, 1280)
        
        if module_idx == self.total_modules - 2:
            example = example[0]

        if module_idx == self.total_modules - 1:
            example = example.reshape(1, 1000)

        return example



def get_split_model(model_name, model):
    if model_name == "resnet50":
        return ResNetSplitModel(model_name, model)
    if model_name == "vit":
        return ViTSplitModel(model_name, model)
    return SplitModel(model_name, model)