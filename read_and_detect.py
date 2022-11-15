import io
import time
import torch
import torchvision
from PIL import Image
from queue import Empty
from pathlib import Path
from output_handler import handle_output
from pymemcache.client import base as memcached

COMPLETE = "READING_COMPLETE"
def transform(pil_image):
    # Transforms to apply on the input PIL image
    return torchvision.transforms.functional.to_tensor(pil_image)

def read_images_into_q(images_path, queue, event, psend_pipe, rate=15, mcaddress=None, ext="JPEG",\
        wait_time=0.05, transform=transform, ):
    """
    Reader process, if queue is not full it will read an `ext` image from
    `images_path` and put it onto the `queue` after applying the `transform`, 
    else it will wait for `wait_time` for the  queue to free up.
    
    It uses `send_pipe` to signal downstream processes when all images have 
    been entered into the queue.

    It uses `psend_pipe` for indication.
    """
    assert(rate != 0)
    
    mc_client = memcached.Client((mcaddress.split(":")[0], int(mcaddress.split(":")[1])))
    image_list = list(Path(images_path).rglob(f"*.{ext}"))
    print(f"processing {len(image_list)} images... ")
    
    # Calculate when the send the next image
    next_timestamp = time.time() + 1 / rate
    while len(image_list) > 0:
        if not queue.full():
            image_path = image_list.pop()
            # image = Image.open(image_path)
            # print(image_path.name)
            image = Image.open(io.BytesIO(mc_client.get(image_path.name)))
            image = transform(image)
            
            queue.put((image_path, time.time()))
            psend_pipe.send((len(image_list), image_path.name))
        wait_duration = next_timestamp - time.time()
        time.sleep(wait_duration if wait_duration > 0 else 0)
        next_timestamp = time.time() + 1 / rate
        
    event.set()
    queue.join()

def detect_objects(queue, event, detector, device, lock, output_path, shared_list, mcaddress):
    """
    Detector process, Reads a transformed image from the `queue`
    passes it to the detector from `get_detector` and processes the 
    output using `lock`  and `output_path` file for handling the output.
    Uses `pipe` to know if all the images have been written to
    the `queue`.
    """
    file = open(output_path.as_posix(), "a")
    detector.eval().to(device)
    
    mc_client = memcached.Client((mcaddress.split(":")[0], int(mcaddress.split(":")[1])))
    while not (event.is_set() and queue.empty()):
        try:
            image_path, start_time = queue.get(block=True, timeout=0.1)
            image = Image.open(io.BytesIO(mc_client.get(image_path.name)))
            image = transform(image)
        except Empty:
            continue

        with torch.no_grad():
            image = [image.to(device)]
            output = detector(image)[0]
        queue.task_done()
        end_time = time.time()
        handle_output(image_path, output, lock, file, shared_list, start_time, end_time)
    file.close()
