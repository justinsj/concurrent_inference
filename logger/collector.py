import subprocess
import psutil
import time
from functools import reduce
import pandas as pd

BYTE_TO_BIT = 8
MEGA = 1000000

def get_gpu_data(last_data, args):
    '''
    Returns a dictionary of keys of metrics and corresponding values.
    The values are strings that are the corresponding gpu compute 
    and memory utilization based on last_data and args.

    get_gpu_data: (dictof Str (anyof Str Float)) (dictof Str Str) -> (dictof Str (anyof Str Float))
    '''
    
    # Get the target GPU index from args
    gpu_index = args['gpu-index']
    
    # Use nvidia-smi to get the GPU data
    cmd = 'nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.used --format=csv,noheader,nounits -i {}'.format(gpu_index)
    
    # Execute the command
    gpu_data = subprocess.check_output(cmd, shell=True).decode('utf-8').split(',')
    
    # Create the dictionary for the data
    data = {}
    data['gpu_util'] = float(gpu_data[0])
    data['gpu_memory_util'] = float(gpu_data[1])
    data['gpu_memory_total'] = float(gpu_data[2])
    data['gpu_memory_used'] = float(gpu_data[3])
    
    return data

def load_util_data(last_data={}, args={'net-iface':'eth0'}):
    '''
    Returns a dictionary of keys of metrics and corresponding values.
    The values are strings that are the corresponding cpu / memory / 
    disk / network IO measurement based on last_data.

    load_data: (dictof Str (anyof Str Float)) -> (dictof Str (anyof Str Float))
    '''
    data = {}

    # Get CPU freq
    cpu_freq = psutil.cpu_freq()

    # Get the memory data
    virtual_memory = psutil.virtual_memory()

    # Get disk data
    disk_usage = psutil.disk_usage('/')

    # Get the network data
    net_if_stats = psutil.net_if_stats()

    # Get the network IO data
    net_io_counters = psutil.net_io_counters()

    # Get the current unix timestamp
    unix = time.time()

    # Get the CPU data
    data['unix'] = unix
    data['cpu_util'] = psutil.cpu_percent()
    data['cpu_freq_current'] = cpu_freq.current * 1000
    data['cpu_freq_min'] = cpu_freq.min
    data['cpu_freq_max'] = cpu_freq.max

    data['memory_available'] = virtual_memory.available
    data['memory_total'] = virtual_memory.total
    
    data['disk_free'] = disk_usage.free
    data['disk_total'] = disk_usage.total


    data['net_speed'] = reduce( lambda t, n: t + n[1].speed, filter(lambda item: item[0].startswith(args['net-iface']),net_if_stats.items()), 0) * MEGA

    delta_net_byte_recv = (
        net_io_counters.bytes_recv - last_data['net_byte_recv'] if 
        ('net_byte_recv' in last_data) else 0
    )
    delta_net_byte_sent = (
        net_io_counters.bytes_sent - last_data['net_byte_sent'] if 
        ('net_byte_sent' in last_data) else 0
    )
    delta_unix = unix - last_data['unix'] if ('unix' in last_data) else 1 # This is in seconds

    net_speed_used = (delta_net_byte_recv + delta_net_byte_sent)*8 / delta_unix
    data['net_speed_used'] = net_speed_used
    data['net_byte_recv'] = net_io_counters.bytes_recv
    data['net_byte_sent'] = net_io_counters.bytes_sent
    
    data['net_util'] = net_speed_used / data['net_speed'] * 100
    
    gpu_data = get_gpu_data(last_data, args)
    data['gpu_util'] = gpu_data['gpu_util']
    data['gpu_memory_util'] = gpu_data['gpu_memory_util']
    data['gpu_memory_total'] = gpu_data['gpu_memory_total']
    data['gpu_memory_used'] = gpu_data['gpu_memory_used']

    return data
