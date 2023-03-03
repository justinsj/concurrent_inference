import os
import time
import torch
import torch.multiprocessing as mp
import statistics
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import torchvision
from read_and_detect import read_images_into_q, detect_objects
from torchvision.models.detection import fasterrcnn_resnet50_fpn

import io

from PIL import Image
from queue import Empty
from pathlib import Path
from output_handler import handle_output
from provider import get_provider

def print_qsize(event, precv_pipe, queue):
    try:
        pbar = tqdm(bar_format="{desc}")
        while not (event.is_set() and queue.empty()):
            if not precv_pipe.poll(): continue
            remaining, name = precv_pipe.recv()
            pbar.desc = f"rem : {remaining:4}, " + \
                f"qsize : {queue.qsize():2}, " + \
                f"current : {name}"
            pbar.update()
            time.sleep(0.05)
        pbar.close()
    except NotImplementedError as err:
        print("JoinableQueue.qsize has not been implemented;"+
            "remainging can't be shown")

def caller(device, 
    inputs_path, output_path, 
    detector_count=2, qsize=8, rate=15, 
    model_name="resnet50",
    model_folder='.',
    max_count=100):

    start = time.time()
    # Initialize sync structures
    queue = mp.JoinableQueue(qsize)
    event = mp.Event()
    data_map = mp.Manager().dict()
    precv_pipe, psend_pipe = mp.Pipe(duplex=False)
    closables = [queue, precv_pipe, psend_pipe]
    lock = mp.Lock()

    provider = get_provider(model_folder, model_name, device, max_count)
    print(f"Provider: {provider}")
    # Initialize processes
    reader_process = mp.Process(
        target=read_images_into_q,
        args=(provider, inputs_path, queue, event, psend_pipe, rate)
    )
    
    shared_list = mp.Manager().list()
    detector_processes = [\
            mp.Process(\
                target=detect_objects,\
                args=(provider, queue, event, lock, output_path, shared_list, data_map, i))\
            for i in range(detector_count)]

    # Starting processes
    reader_process.start()
    [dp.start() for dp in detector_processes]

    # print_qsize(event, precv_pipe, queue)

    # Waiting for processes to complete
    [dp.join() for dp in detector_processes]
    reader_process.join()

    # Closing everything
    [c.close() for c in closables]
    time_taken = time.time() - start
    print(f"time taken : {time_taken} s.")
    
    # Print the rate of inference
    num_items = provider.max_count
    throughput = num_items / time_taken
    print(f"rate : {throughput} images/s")
    
    # Get list of latencies
    latencies = [end - start for (path, output_string, start, end) in shared_list]
    
    # Calculate the average latency of requests
    avg_latency = statistics.mean(latencies)
    
    # Calculate the p99 latency of requests
    latencies.sort()
    
    p99_latency = np.percentile(latencies, 99)
    
    average_rate = sum([data_map[i]["rate"] for i in range(detector_count)]) / detector_count
    print(data_map)
    
    output_path = f"logger/logs/rates/{model_name}_{detector_count}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame({
        'msg_rate': rate,
        'detector_count': detector_count,
        'q_size': qsize,
        'throughput': throughput,
        'avg_latency': avg_latency,
        'p99_latency': p99_latency,
        'model_name': model_name,
        'max_count': max_count,
        'average_rate': average_rate,
    }, index=[0])

    df.to_csv(output_path, mode='a', index=False, header=not os.path.exists(output_path))
    # # If file is empty, add the headers
    # if (not Path(filename).is_file()) or (Path(filename).stat().st_size == 0):
    #     with open(filename, "w+") as f:
    #         f.write("msg_rate,detector_count,q_size,throughput,avg_latency,p99_latency,model_name,max_count,average_rate\n")
    # # Store the rate in a file
    # with open(filename, "a+") as f:
    #     f.write(f"{rate},{detector_count},{qsize},{throughput},{avg_latency},{p99_latency},{model_name},{max_count},{average_rate}\n")
        
        
    # Store the average rate among detectors in a file
    # Each entry in data_map is a dict of rate, count, and duration
    
    # filename = "average_rates.csv"

    # if (not Path(filename).is_file()) or (Path(filename).stat().st_size == 0):
    #     with open(filename, "w+") as f:
    #         f.write("model_name,rate,detector_count\n")
    # with open(filename, "a+") as f:
    #     f.write(f"{model_name},{average_rate},{detector_count}\n")

