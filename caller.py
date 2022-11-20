import time
import torch
import torch.multiprocessing as mp
import statistics
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import torchvision
from read_and_detect import read_images_into_q, detect_objects
from torchvision.models.detection import fasterrcnn_resnet50_fpn

import io
from pymemcache.client import base as memcached

from PIL import Image
from queue import Empty
from pathlib import Path
from output_handler import handle_output


   

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
def transform(pil_image):
    # Transforms to apply on the input PIL image
    return torchvision.transforms.functional.to_tensor(pil_image)
def caller(device, images_path, output_path, detector_count=2, qsize=8, rate=15, mcaddress=None, model_name="resnet50"):

    start = time.time()
    # Initialize sync structures
    queue = mp.JoinableQueue(qsize)
    event = mp.Event()
    data_map = mp.Manager().dict()
    precv_pipe, psend_pipe = mp.Pipe(duplex=False)
    closables = [queue, precv_pipe, psend_pipe]
    lock = mp.Lock()
    mc_client = memcached.Client((mcaddress.split(":")[0], int(mcaddress.split(":")[1])))

    
    # Initialize the memcached database with all the images
    for image_path in Path(images_path).rglob("*.JPEG"):
        mc_client.set(image_path.name, image_path.read_bytes())

    sleep_time = 5
    time.sleep(sleep_time)
    
    
    # Initialize processes
    reader_process = mp.Process(
        target=read_images_into_q,
        args=(images_path, queue, event, psend_pipe, rate, mcaddress)
    )
    
    shared_list = mp.Manager().list()
    detector_processes = [\
            mp.Process(\
                target=detect_objects,\
                args=(queue, event, model_name,\
                    device, lock, output_path, shared_list, mcaddress, data_map, i))\
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
    num_images = len(list(Path(images_path).rglob("*.JPEG")))
    throughput = num_images / time_taken
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
    
    filename = "rates.csv"
    # If file is empty, add the headers
    if (not Path(filename).is_file()) or (Path(filename).stat().st_size == 0):
        with open(filename, "w+") as f:
            f.write("rate,detector_count,q_size,throughput,avg_latency,p99_latency,model_name,average_rate\n")
    # Store the rate in a file
    with open(filename, "a+") as f:
        f.write(f"{rate},{detector_count},{qsize},{throughput},{avg_latency},{p99_latency},{model_name},{average_rate}\n")
        
        
    # Store the average rate among detectors in a file
    # Each entry in data_map is a dict of rate, count, and duration
    
    filename = "average_rates.csv"
    if (not Path(filename).is_file()) or (Path(filename).stat().st_size == 0):
        with open(filename, "w+") as f:
            f.write("model_name,rate,detector_count\n")
    with open(filename, "a+") as f:
        f.write(f"{model_name},{average_rate},{detector_count}\n")

