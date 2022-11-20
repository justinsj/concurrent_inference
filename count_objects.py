import torch
import argparse
import torch.multiprocessing as mp
from caller import caller
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Get the count of detected objects.")
    parser.add_argument("-f", "--folder", 
            help="folder having the images", type=Path,
            required=True)
    parser.add_argument("-o", "--output",
            help="path of the output file", type=Path,
            required=True, dest="output")
    parser.add_argument("-d", "--detectors", 
            help="number of detector processes", type=int, default=2)
    parser.add_argument("-q", "--qsize", 
            help="size of the image queue", type=int, default=8)
    parser.add_argument('-r', '--rate', 
            help="rate of incoming requests added to the queue", type=str, default="1000")
    parser.add_argument('-n','--model_name', type=str, default="resnet50",
                        help="model name")
    parser.add_argument('-i', '--gpu_index', type=int, default=0, help="gpu index")
    parser.add_argument('-s', '--image_size', type=int, default=224, help="size of input image for inference")
    args = parser.parse_args()

    mp.set_start_method("spawn")
#     device = torch.device("cuda:" + str(args.gpu_index) if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # Create the memcached connection
    caller(device, args.folder, args.output, args.detectors, args.qsize, float(args.rate), args.model_name, args.image_size)
