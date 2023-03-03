import os
import signal
import subprocess
import time
import pandas as pd
# Measure the host and gpu compute and memory footprints of the model
# See concurrent inference for related code


MAX_COUNT = 98
model_names = [
    'codebert-base',
    'albert-xxlarge-v2', 
    't5-3B',
    'DialoGPT-large', 
    'bart-large', 
    'gpt2-xl',
]

for model_name in model_names:
    model_name = model_name.replace('-', '_')
    print(f"Processing model: {model_name}")
    d = 1
    while True:
        logger_csv_path = f"logger/logs/{model_name}_{d}.csv"
        logger_command = f"python logger/logger.py {logger_csv_path} -i 1"
        process_command = f"python count_objects.py -if ../measure-lms/codebert/clean_inputs.csv -o output_file.log -q 1000 -r 1000 -m {model_name} -mf ../measure-lms/codebert -d {d} -i 1 -mc {MAX_COUNT}"

        # Process
        ## Start the logger in background but keep the handle
        logger_pid = subprocess.Popen(logger_command.split(' '), close_fds=True).pid

        print(f"Logger pid: {logger_pid}")
        ## Wait for a few seconds to measure idle state
        time.sleep(5)
        # os.kill(logger_pid, signal.SIGKILL)
        

        print(f"Starting process")
        ## Start the process
        process_handle = subprocess.Popen(process_command, shell=True, close_fds=True)
        print(f"Waiting for process to finish")
        ## Wait for the process to finish successfully
        process_handle.wait()

        print(f"Process finished: {process_handle.returncode}")

        ## Check that the process finished successfully
        assert(process_handle.returncode == 0)

        ## Wait for a few seconds
        time.sleep(5)

        print(f"Stopping logger")

        ## Stop the logger
        os.kill(logger_pid, signal.SIGKILL)

        time.sleep(1)

        # Run the single inference at a time to get average latency
        process_command = f"python count_objects.py -if ../measure-lms/codebert/clean_inputs.csv -o output_file.log -q 1 -r 1000 -m {model_name} -mf ../measure-lms/codebert -d {d} -i 1 -mc {MAX_COUNT}"

        process_handle = subprocess.Popen(process_command, shell=True, close_fds=True)
        print(f"Waiting for process to finish")
        ## Wait for the process to finish successfully
        process_handle.wait()
        assert(process_handle.returncode == 0)

        ## Load and format the results
        df = pd.read_csv(logger_csv_path)
        print(df)
        # If at least 1 entry has gpu_util greater than 95, break
        has_active_row = df[df['gpu_util'] > 95].shape[0] > 0
        if has_active_row:
            break
        else:
            d *= 2