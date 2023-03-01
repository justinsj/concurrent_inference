import pandas as pd
import io
import time
import torch
import torchvision
from PIL import Image
from queue import Empty
from output_handler import handle_output
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms

COMPLETE = "READING_COMPLETE"

def read_images_into_q(provider, inputs_path, queue, event, psend_pipe, rate=15, wait_time=0.05):
    """
    Reader process, if queue is not full it will read an `ext` image from
    `images_path` and put it onto the `queue` after applying the `transform`, 
    else it will wait for `wait_time` for the  queue to free up.
    
    It uses `send_pipe` to signal downstream processes when all images have 
    been entered into the queue.

    It uses `psend_pipe` for indication.
    """
    assert(rate != 0)
    
    inputs_list = provider.load_inputs(inputs_path)
    print(f"processing {len(inputs_list)} images... ")
    
    # Calculate when the send the next image
    next_timestamp = time.time() + 1 / rate
    while len(inputs_list) > 0:
        if not queue.full():
            input_argument = inputs_list.pop()
            queue.put(provider.format_input(input_argument))
            
            psend_pipe.send((len(inputs_list), input_argument))
        wait_duration = next_timestamp - time.time()
        time.sleep(wait_duration if wait_duration > 0 else 0)
        next_timestamp = time.time() + 1 / rate
        
    event.set()
    queue.join()
    
def detect_objects(provider, queue, event, model_name, device, lock, output_path, shared_list, data_map, idx):
    """
    Detector process, Reads a transformed image from the `queue`
    passes it to the detector from `get_detector` and processes the 
    output using `lock`  and `output_path` file for handling the output.
    Uses `pipe` to know if all the images have been written to
    the `queue`.
    """
    file = open(output_path.as_posix(), "a")
    
    model = provider.get_model()

    start_time = time.time()
    count = 0
    while not (event.is_set() and queue.empty()):
        try:
            queue_message = queue.get(block=True, timeout=0.1)
        except Empty:
            continue

        with torch.no_grad():
            output, input_argument, start_time = provider.inference_from_queue_message(model, queue_message)
            
            count += 1
        queue.task_done()
        end_time = time.time()
        handle_output(input_argument, output, lock, file, shared_list, start_time, end_time)
    end_time = time.time()
    duration = end_time - start_time
    rate = count / duration
    data_map[idx] = {"rate": rate, "duration": duration, "count": count}
    file.close()
