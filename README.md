# concurrent_inference

An example of how to use the `multiprocessing` package along with PyTorch.

Code pertaining to this [Medium post](https://18alan.medium.com/concurrent-inference-e2f438469214).

---

## What does it do?
![Data flow diagram](media/usecase.png)
- A processes [R] reads images from a folder [Fo] and multiple detection processes [D#] are used to obtain the class wise count of objects in the images and write it to a file [Fi].
- It uses `torch.multiprocessing` for multiprocessing, `PIL.Image` to read the images and `tqdm` to keep track of the queue.  
![processing](media/processing.gif)

## Usage
- Basic usage : `$ python count_objects.py -f input_folder -o output_file.log -d 4`
- For other options : `$ python count_objects.py -h`

Start the memcached server
```
memcached -p 11211 -m 30000 -v -I 1024m -l 192.168.168.173
```
```
python logger/logger.py logger/logs/gpu-saturation-8-rn50-1.csv -i 1

```
```
python count_objects.py -f input_folder -o output_file.log -q 1000 -r 1000 -n resnet50 -s 224 -d 1
```


https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries