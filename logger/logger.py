'''
Created on April, 2022
@author: Justin San Juan
@Project: CS798-002
'''
import time
import psutil
import pandas as pd
import json
import argparse

from collector import load_util_data
from analytics import get_efficiencies

def main(loop_duration_s, outpath, net_iface, gpu_index):
    args = {'net-iface': net_iface, 'gpu-index': gpu_index}
    last_data = load_util_data(args=args)
    print(last_data)
    df = pd.read_json(json.dumps([last_data]), orient='records')
    df.to_csv(outpath, index=False, header=True)
    efficiencies = get_efficiencies(last_data)
    print(efficiencies)
    time.sleep(loop_duration_s)

    while True:
        last_time = time.time()
        # Load the data
        data = load_util_data(last_data,args=args)
        
        df = pd.read_json(json.dumps([data]), orient='records')
        df.to_csv(outpath, mode='a', index=False, header=False)

        efficiencies = get_efficiencies(data)
        # Convert all values to strings with 2 decimal places
        efficiencies = {k: "{:.2f}".format(v) for k, v in efficiencies.items()}
        print(efficiencies)
        last_data = data

        # Wait for time to start the next iteration almost exactly
        # PERIOD_DURATION_S seconds after the last iteration started.
        # This accounts for the execution time of publishing to each topic. 
        new_time = time.time()
        diff = max(0, loop_duration_s + last_time - new_time)
        time.sleep(diff)

if __name__ == '__main__':       
    parser = argparse.ArgumentParser(description='Run a resource utilization logging process.')
    parser.add_argument('outpath', type=str, help='Save logs to destination csv path')
    parser.add_argument('-l','--loop_duration_s', type=int, default=1, 
        help='Loop duration in seconds')
    parser.add_argument('-n','--net_iface', type=str, default='enp24s0',
        help='Network interface to use for network speed measurement')
    parser.add_argument('-i','--gpu_index', type=int, default=0,
        help='GPU index to use for GPU utilization measurement')
    

    args = parser.parse_args()
    main(args.loop_duration_s, args.outpath, args.net_iface, args.gpu_index)

    
        
        
