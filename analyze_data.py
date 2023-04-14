import os
import pandas as pd

MiB_TO_B = 1024 * 1024
# Read the logged data from logger/logs

def calculate_gpu_compute_usage(average_execution_time, rate):
    '''
    Returns a float from 0 to 1 of the compute usage of the model on a GPU.
    
    calculate_gpu_compute_usage: Float Float -> Float
    
    Examples:
    if average_execution_time is 1s, and rate is 2/s
    then the gpu compute usage is 0.5
    
    if average_execution_time is 1s, and rate is 1/s
    then the gpu compute usage is 1
    
    if average_execution_time is 4s, and rate is 0.25/s
    then the gpu compute usage is 1

    if average_execution_time is 4s, and rate is 1/s
    then the gpu compute usage is 0.25

    if average_execution_time is 4s, and rate is 0.33/s
    then the gpu compute usage is 0.75
    '''
    print(f"Avg exec: {average_execution_time}")
    gcu = 1 / (average_execution_time * rate)
    assert gcu <= 1.0, f"Expected {gcu} to be <= 1.0"
    return gcu
    
def get_rate(model_name, d):
    df = pd.read_csv(f'logger/logs/rates/{model_name}_{d}.csv')
    df = df[df['model_name'] == model_name][df['detector_count'] == d][df['q_size'] == 1000]
    df['rate'] = df['average_rate'] * df['detector_count']
    return df['rate'].mean()

def get_average_execution_time(model_name, d):
    df = pd.read_csv(f'logger/logs/rates/{model_name}_{d}.csv')
    df = df[df['model_name'] == model_name][df['detector_count'] == d][df['q_size'] == 1]
    return df['avg_latency'].mean()

def process_file(path, model_name, d, model_folder):
    # Read the data from the file
    df = pd.read_csv(path)

    print(df)

    # Get the rows where the process is idle
    # GPU memory used is < 10MB
    # Get max GPU memory usage
    max_gpu_memory_usage = df['gpu_memory_used'].max()
    inactive_rows = df[df['gpu_memory_used'] < max_gpu_memory_usage * 0.9]

    # Get the rows where the process is active
    # GPU util > 95
    active_rows = df[df['gpu_util'] > 95]

    min_gpu_memory_used_exec = active_rows['gpu_memory_used'].min()
    process_active_rows = df[df['gpu_memory_util'] == 0][df['gpu_memory_used'] > 10][df['gpu_memory_used'] < min_gpu_memory_used_exec]
    process_inactive_rows = df[df['gpu_memory_used'] > 10][df['gpu_memory_used'] < min_gpu_memory_used_exec]
    # Load the average execution time of the model
    average_execution_time = get_average_execution_time(model_name, d)
    # Load the rate that the task was completed
    rate = get_rate(model_name, d)
    

    # Get the host cpu usage (cpu_util)
    cpu_freq_min = inactive_rows['cpu_freq_min'].values[0]
    cpu_freq_max = inactive_rows['cpu_freq_max'].values[0]

    cpu_freq_delta = (active_rows['cpu_freq_current'].max() - inactive_rows['cpu_freq_current'].min()) / 1000 
    host_process_compute_usage = (cpu_freq_delta) / (cpu_freq_max - cpu_freq_min)

    # Get the host memory usage (memory used based on memory_available)
    host_process_memory_usage = (active_rows['memory_total'].mean() - active_rows['memory_available'].mean()) - \
            (inactive_rows['memory_total'].mean() - inactive_rows['memory_available'].mean())

    # Get the host model cpu usage (0)
    host_model_compute_usage = 0
    # Get the host model memory usage (size of the model file)
    host_model_memory_usage = os.stat(os.path.join(model_folder,f"{model_name.replace('-','_')}.pt")).st_size

    gpu_process_memory_usage = (active_rows['gpu_memory_used'].mean() - inactive_rows['gpu_memory_used'].mean()) * MiB_TO_B
    gpu_process_compute_usage = (process_active_rows['gpu_util'].max() - process_inactive_rows['gpu_util'].min()) / 100

    gpu_exec_memory_usage = (active_rows['gpu_memory_used'].mean() - process_inactive_rows['gpu_memory_used'].mean()) * MiB_TO_B
    gpu_exec_compute_usage = calculate_gpu_compute_usage(average_execution_time, rate)

    assert gpu_exec_compute_usage + gpu_process_compute_usage <= 1.0, f"Expected {gpu_exec_compute_usage + gpu_process_compute_usage} to be <= 1.0"
    return {
        'gpu_process_memory_usage': int(gpu_process_memory_usage),
        'gpu_process_compute_usage': gpu_process_compute_usage,
        'host_process_compute_usage': host_process_compute_usage,
        'host_process_memory_usage': int(host_process_memory_usage),
        'host_model_compute_usage': host_model_compute_usage,
        'host_model_memory_usage': int(host_model_memory_usage),
        'gpu_exec_memory_usage': int(gpu_exec_memory_usage),
        'gpu_exec_compute_usage': gpu_exec_compute_usage,
        'model_name': model_name,
    }

BASE_DIR = 'logger/logs'
model_folder = '../measure-lms/codebert'

df = pd.DataFrame(columns=['gpu_process_memory_usage', 'gpu_process_compute_usage', 'host_process_compute_usage', 'host_process_memory_usage', 'host_model_compute_usage', 'host_model_memory_usage'])
for filename in os.listdir(BASE_DIR):
    filepath = os.path.join(BASE_DIR, filename)
    if (os.path.isfile(filepath)):
        model_name = '_'.join(filename.split('_')[:-1])
        d = int(filename.split('_')[-1].split('.')[0])
        data_dict = process_file(filepath, model_name, d, model_folder)
        df = df.append(data_dict, ignore_index=True)

output_path = 'logger/logs/summary/summary.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

            
