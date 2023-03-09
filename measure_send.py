import time
import os
import torch
import gc

import pandas as pd

model_folder = '../measure-lms/codebert'
model_names = [
    # 'resnet50', # 102735697
    # 'vit_h14_in1k', # 2534512143

    # 'codebert_base', # 498965601
    # 'albert-xxlarge-v2', # 890450058
    # 'DialoGPT-large', # 3134799287
    # 'bart-large', # 1625830197
    # 'gpt2-xl', # 6282033981
    't5-3B', # 11408097021
    
    # 't5-small',
    ]

default_device = torch.device('cpu')

df = pd.DataFrame()
NUM_TESTS = 10
output_path = f'logger/logs/cuda_1/send/measurements.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
first_time = True
for idx, model_name in enumerate(model_names):
    i = 0
    while i < NUM_TESTS:
        # Load the pt file
        path = os.path.join(model_folder,model_name.replace('-','_') + '.pt')
        model = torch.jit.load(path, map_location=default_device)
        model = model.to(default_device)

        torch.cuda.empty_cache()
        gc.collect()

        start = time.time()
        model = model.to(torch.device('cuda:0'))
        end = time.time()
        send_time = end - start

        if (first_time):
            first_time = False
            continue
        

        df = df.append({
            'model_name': model_name,
            'i': i,
            'send_time': send_time
        }, ignore_index=True)

        df.to_csv(output_path, index=False)
        i+=1

        





