import time
import torch
import os
import io
from provider import get_provider
import pandas as pd
# Measure the execution time of the distributed models

# Goal is to have measured dataframe with the following headers:
# action,module_idx,num_modules,cold_start_time,exec_time,input_shape,input_bytes,file_size

model_folder = '../measure-lms/codebert'
model_names = [
    # 'resnet50',
    'vit_h14_in1k'
    ]

GPU_INDEX = 1
device = torch.device(f'cuda:{GPU_INDEX}' if torch.cuda.is_available() else 'cpu')

# inputs_folder = '../measure-lms/codebert/clean_inputs.csv'
MAX_COUNT = 3#98

NUM_TESTS = 1


class ColdStartData:
    def __init__(self, cold_start_time, file_size):
        self.cold_start_time = cold_start_time
        self.file_size = file_size

def measure_cold_start(module_idx, split_model, example):
    temp_filename = split_model.get_temp_name(module_idx) 

    module_part = split_model.list_of_modules[module_idx]

    module_trace = split_model.trace(module_idx, example)
    module_trace.save(temp_filename)
    
    # Get the size of the file
    file_size = os.path.getsize(temp_filename)

    # Load the data from the file as bytes
    data = open(temp_filename, 'rb').read()

    start = time.time()
    module_sample = torch.jit.load(io.BytesIO(data))
    end = time.time()

    # Delete the file
    os.remove(temp_filename)

    cold_start_time = end - start
    return ColdStartData(cold_start_time, file_size)

def measure_inference(module_idx, split_model, example):
    start = time.time()
    split_model.inference(module_idx, example)
    end = time.time()
    exec_time = end - start
    return exec_time

# For each DL model
for model_name in model_names:
    output_path = f'logger/logs/times/measurements_{model_name}.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    provider = get_provider(model_folder, model_name, device, MAX_COUNT)
    print(provider)
    inputs_folder = provider.get_inputs_folder()
    inputs_list = provider.load_inputs(inputs_folder)
    print(inputs_list[0:5])
    
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
                print(f"Running {model_name} {module_idx}/{split_model.total_modules} {i}/{NUM_TESTS}")
                input_shape = list(example.shape)
                input_bytes = example.nelement() * example.element_size()

                # Measure the cold start time
                cold_start_data = measure_cold_start(module_idx, split_model, example)
                # cold_start_time, file_size

                # Measure the inference time
                exec_time = measure_inference(module_idx, split_model, example)
                # exec_time
                
                # Keep the output and perform the conversion to the next input
                example = split_model.get_next_input(module_idx, example)

                main_df.append({
                    'action': 'cold_start', 
                    'module_idx': module_idx, 
                    'num_modules': split_model.total_modules, 
                    'cold_start_time': cold_start_data.cold_start_time, 
                    'exec_time': exec_time,
                    'input_shape': input_shape,
                    'input_bytes': example.nelement() * example.element_size(),
                    'file_size': cold_start_data.file_size,
                }, ignore_index=True)

                main_df.to_csv(output_path, index=False)




