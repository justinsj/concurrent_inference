import argparse
import gc

import time
import torch
import os
import io
from provider import get_provider
import pandas as pd
# Measure the execution time of the distributed models

# Goal is to have measured dataframe with the following headers:
# action,module_idx,num_modules,cold_start_time,exec_time,input_shape,input_bytes,file_size

model_folder = './models'
# model_names = [
#     # 'resnet50', # 102735697
#     # 'vit_h14_in1k', # 2534512143

#     'codebert_base', # 498965601
#     'albert-xxlarge-v2', # 890450058
#     'DialoGPT-large', # 3134799287
#     'bart-large', # 1625830197
#     'gpt2-xl', # 6282033981
#     't5-3B', # 11408097021

#     # 't5-small',
#     ]

GPU_INDEX = 0

device_name = f'cuda:{GPU_INDEX}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)

# inputs_folder = '../measure-lms/codebert/clean_inputs.csv'
MAX_COUNT = 50

NUM_TESTS = 10


class ColdStartData:
    def __init__(self, cold_start_time, file_size, send_time):
        self.cold_start_time = cold_start_time
        self.file_size = file_size
        self.send_time = send_time

def measure_cold_start(module_idx, split_model, example, device):
    temp_filename = split_model.get_temp_name(module_idx) 

    module_part = split_model.list_of_modules[module_idx]

    module_trace = split_model.trace(module_idx, example, strict=False)
    module_trace.save(temp_filename)
    
    # Get the size of the file
    file_size = os.path.getsize(temp_filename)

    # Load the data from the file as bytes
    data = open(temp_filename, 'rb').read()

    default_device = torch.device('cpu')
    start = time.time()
    module_sample = torch.jit.load(io.BytesIO(data))
    end = time.time()
    
    module_sample = module_sample.to(default_device)

    cold_start_time = end - start

    start = time.time()
    module_sample = module_sample.to(device)
    end = time.time()
    send_time = end - start
    # Delete the file
    os.remove(temp_filename)
    return ColdStartData(cold_start_time, file_size, send_time)

def measure_inference(module_idx, split_model, example, device):
    start = time.time()
    split_model.inference(module_idx, example)
    end = time.time()
    exec_time = end - start
    return exec_time

BASE_FOLDER = f"logger/logs/{device_name.replace(':','_')}/times"

parser = argparse.ArgumentParser()
parser.add_argument('model_name', type=str, help='Model name (e.g. codebert_base)')
# For each DL model
# for model_name in model_names:

args = parser.parse_args()
model_name = args.model_name
model_name = model_name.replace('-', '_')
output_path = os.path.join(BASE_FOLDER, f'measurements_{model_name}.csv')
os.makedirs(os.path.dirname(output_path), exist_ok=True)

provider = get_provider(model_folder, model_name, device, MAX_COUNT)
print(provider)
inputs_folder = provider.get_inputs_folder()
inputs_list = provider.load_inputs(inputs_folder)
# print(inputs_list[0:5])

main_df = pd.DataFrame(columns=['action','module_idx','num_modules','cold_start_time','exec_time','input_shape','input_bytes','file_size'])
# For each input example
for input_argument in inputs_list:
    # Run the step-by-step inference n times
    for i in range(NUM_TESTS):
        # For each model part 
        split_model = provider.split_model
        example = provider.prepare_input(input_argument)
        for module_idx, module_part in enumerate(split_model.list_of_modules):
            
            example = split_model.format_input(module_idx, example)
            module_part = split_model.list_of_modules[module_idx]
            print(f"Running {model_name} {module_idx}/{split_model.total_modules} {i}/{NUM_TESTS} {module_part.__class__.__name__}")
            input_shape, input_bytes = split_model.get_input_data(module_idx, example)

            # Measure the cold start time
            cold_start_data = measure_cold_start(module_idx, split_model, example, device)
            # cold_start_time, file_size, send_time

            # Measure the inference time
            exec_time = measure_inference(module_idx, split_model, example, device)
            # exec_time
            
            # Keep the output and perform the conversion to the next input
            example = split_model.get_next_input(module_idx, example)

            main_df = main_df.append({
                'action': 'cold_start', 
                'module_idx': module_idx, 
                'num_modules': split_model.total_modules, 
                'cold_start_time': cold_start_data.cold_start_time,
                'send_time': cold_start_data.send_time, 
                'exec_time': exec_time,
                'input_shape': input_shape,
                'input_bytes': input_bytes,
                'file_size': cold_start_data.file_size,
            }, ignore_index=True)

            main_df.to_csv(output_path, index=False)
        
        provider.split_model.reset()
        torch.cuda.empty_cache()
        gc.collect()




