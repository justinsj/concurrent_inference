import time
import torch
from provider import CodeBertProvider

model_name = 'codebert_base'
model_folder = '../measure-lms/codebert'
device = torch.device("cuda:" + str(1) if torch.cuda.is_available() else "cpu")
max_count = 200
provider = CodeBertProvider(model_folder, model_name, device, max_count)

input_path = '~/measure-lms/codebert/clean_inputs.csv'

inputs_list = provider.load_inputs(input_path)

model_tokenizer = provider.get_model()

for idx, input_argument in enumerate(inputs_list):
    print(f"item: {idx+1}")
    input_data = provider.prepare_input(input_argument)
    start_time = time.time
    queue_message = (input_data, input_argument, start_time)

    output = provider.inference_from_queue_message(model_tokenizer, queue_message)

