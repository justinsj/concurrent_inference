import pandas as pd
import io
import time
import torch
import torchvision
from PIL import Image
from queue import Empty
from pathlib import Path
from output_handler import handle_output
from pymemcache.client import base as memcached
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms

COMPLETE = "READING_COMPLETE"
def transform(pil_image):
#     # Transforms to apply on the input PIL image

#     do_transform = transforms.Compose([            #[1]
#     transforms.Resize(256),                    #[2]
#     transforms.CenterCrop(224),                #[3]
#     transforms.ToTensor(),                     #[4]
#     # transforms.Normalize(                      #[5]
#     # mean=[0.485, 0.456, 0.406],                #[6]
#     # std=[0.229, 0.224, 0.225]                  #[7]
    
#     # )
#     ])

    # If image is grayscale, convert to RGB
    pil_image = pil_image.convert("RGB")
    
    preprocess = transforms.Compose([
        transforms.Resize(518),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])
#     # return torchvision.transforms.functional.to_tensor(pil_image)
    return preprocess(pil_image)

def read_images_into_q(images_path, queue, event, psend_pipe, rate=15, mcaddress=None, ext="JPEG",\
        wait_time=0.05 ):
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
            # image = Image.open(io.BytesIO(mc_client.get(image_path.name)))
            # image = transform(image)
            image = None
            queue.put((image, image_path, time.time()))
            psend_pipe.send((len(image_list), image_path.name))
        wait_duration = next_timestamp - time.time()
        time.sleep(wait_duration if wait_duration > 0 else 0)
        next_timestamp = time.time() + 1 / rate
        
    event.set()
    queue.join()
def get_detector(model_name):
    # if (model_name == "resnet50"):
    #     return torchvision.models.resnet50(pretrained=True)
    # elif (model_name == "inception_v3"):
    #     return torchvision.models.inception_v3(pretrained=True)
    # elif (model_name == "vit_b16_in1k"):
    #     return torch.hub.load("facebookresearch/swag", model="vit_b16_in1k")
    # elif (model_name == "vgg19_bn"):
    #     return torchvision.models.vgg19_bn(pretrained=True)
    # elif (model_name == "wide_resnet101_2"):
    #     return torchvision.models.wide_resnet101_2(pretrained=True)
    # elif (model_name == "vit_h14_in1k"):
    #     return torch.hub.load("facebookresearch/swag", model="vit_h14_in1k")

    return torch.jit.load(model_name + ".pt")
def detect_objects(queue, event, model_name, device, lock, output_path, shared_list, mcaddress, data_map, idx):
    """
    Detector process, Reads a transformed image from the `queue`
    passes it to the detector from `get_detector` and processes the 
    output using `lock`  and `output_path` file for handling the output.
    Uses `pipe` to know if all the images have been written to
    the `queue`.
    """
    file = open(output_path.as_posix(), "a")
    detector = get_detector(model_name)
    detector.eval().to(device)
    
    # labels = pd.read_csv("labels.txt", sep="\n", header=None).values.tolist()
    
    start_time = time.time()
    mc_client = memcached.Client((mcaddress.split(":")[0], int(mcaddress.split(":")[1])))
    count = 0
    while not (event.is_set() and queue.empty()):
        try:
            image, image_path, start_time = queue.get(block=True, timeout=0.1)
            image = Image.open(io.BytesIO(mc_client.get(image_path.name)))
            image = transform(image)
        except Empty:
            continue

        with torch.no_grad():
            
            # img_t = transform(image)
            # batch_t = torch.unsqueeze(img_t.to(device), 0)
            image = torch.unsqueeze(image, 0)
            image = image.to(device)
            output = detector(image)
            count += 1
        queue.task_done()
        end_time = time.time()
        handle_output(image_path, output, lock, file, shared_list, start_time, end_time)
    end_time = time.time()
    duration = end_time - start_time
    rate = count / duration
    data_map[idx] = {"rate": rate, "duration": duration, "count": count}
    file.close()
