import os
import argparse
from transformers import AutoTokenizer, AutoModel, T5Model
import torch
# https://huggingface.co/transformers/v3.3.1/pretrained_models.html

# model_name = "albert-xxlarge-v2"
# model_name = "t5-11B"
# model_name = "t5-3B"
# model_name = "DialoGPT-large"
# model_name = "facebook/bart-large"
# model_name = "gpt2-xl"

model_names = [
    # 'microsoft/codebert-base',
    # 'albert-xxlarge-v2', 
    # 't5-11B', 
    't5-3B',
    # 't5-small', 
    # 'microsoft/DialoGPT-large', 
    # 'facebook/bart-large', 
    # 'gpt2-xl',
    # 'llama'
    ]


def prepare_input(tokenizer, string):
    code_tokens=tokenizer.tokenize(string)
    tokens=[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    tokens_ids=tokenizer.convert_tokens_to_ids(tokens)

    example = torch.tensor(tokens_ids)[None,:]
    # example = example.to(device)
    return example

def download_model(model_name, filename, device):
    if model_name in ['t5-3B']:
        model = T5Model.from_pretrained(model_name, torchscript=True)
    else:
        model = AutoModel.from_pretrained(model_name, torchscript=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # if (model_name == 't5-small'):
    #     tokenizer = model.transform()
    model.eval()
    model.to(device)

    print(f"Model name: {model_name}")

    input_string = "Hello, my dog is cute"
    # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # inputs = inputs.to(device)
    # outputs = model(**inputs)

    # last_hidden_states = outputs.last_hidden_state
    # example = prepare_input(tokenizer, "Hello, my dog is cute")
    # if (model_name == 't5-small'):
    #     example = transform("Hello, my dog is cute")

    # example = example.to(device)
    # example = torch.tensor([[1,2,3]])
    # example = example.to(device)

    # example = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # example = example.to(device)

    # code_tokens=tokenizer.tokenize("def foo():\n    print('hello')")
    # tokens=[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    # tokens_ids=tokenizer.convert_tokens_to_ids(tokens)

    # example = torch.tensor(tokens_ids)[None,:]
    if model_name in ['t5-3B','albert-xxlarge-v2']:
        input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
        attention_mask = input_ids.ne(model.config.pad_token_id).long()
        decoder_input_ids = tokenizer('<pad> <extra_id_0> cute dog <extra_id_1> the <extra_id_2>', return_tensors='pt').input_ids

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        decoder_input_ids = decoder_input_ids.to(device)

        traced_model = torch.jit.trace(model, (input_ids, attention_mask, decoder_input_ids))
    else:
        example = torch.tensor([[1,2,3]])#example = prepare_input(tokenizer, input_string)
        example = example.to(device)
        traced_model = torch.jit.trace(model, example)

    traced_model.save(filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gi','--gpu_index', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu")

    print(device)
    for model_name in model_names:
        print(model_name)
        # Check if file already exists
        filename = model_name.split('/')[-1].replace('-','_')+'.pt'
        print(f"Filename: {filename}")
        if os.path.exists(filename):
            print('File already exists')
            continue
        
        # if model_name == 'codebert':
        # else:
        download_model(model_name, filename, device)
