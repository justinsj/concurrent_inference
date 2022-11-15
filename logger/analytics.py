from functools import reduce
def get_memory_efficiency(data):
    '''
    Returns the efficiency of memory utilization percentage.

    get_memory_efficiency: (dictof Str (anyof Str Float)) -> Float
    '''
    return (data['memory_total'] - data['memory_available']) / (data['memory_total']) * 100 if data['memory_total'] != 0 else 100

def get_cpu_efficiency(data):
    '''
    Returns the efficiency of CPU utilization percentage.

    get_cpu_efficiency: (dictof Str (anyof Str Float)) -> Float
    '''
    freq_current = data['cpu_freq_current']
    freq_max = data['cpu_freq_max']
    freq_min = data['cpu_freq_min']
    return (freq_current - freq_min) / (freq_max - freq_min)  * 100 if freq_max - freq_min != 0 else 100

def get_disk_efficiency(data):
    '''
    Returns the efficiency of disk utilization percentage.

    get_disk_efficiency: (dictof Str (anyof Str Float)) -> Float
    '''
    return (data['disk_total'] - data['disk_free']) / (data['disk_total']) * 100 if data['disk_total'] != 0 else 100

def get_network_efficiency(data):
    '''
    Returns the efficiency of network utilization percentage.

    get_network_efficiency: (dictof Str (anyof Str Float)) -> Float
    '''
    return (data['net_speed_used']) / (data['net_speed']) * 100 if data['net_speed'] != 0 else 100

def get_efficiencies(data):
    '''
    Returns a dictionary of efficiencies for 
    CPU, Memory, Disk, and Network.

    get_efficiencies: (dictof Str (anyof Str Float)) -> (dictof Str (anyof Str Float))
    '''
    return {
        'cpu': get_cpu_efficiency(data),
        'memory': get_memory_efficiency(data),
        'disk': get_disk_efficiency(data),
        'network': get_network_efficiency(data),
    }
def efficiency(efficiencies, lambda_cpu = 1, lambda_memory = 1, lambda_disk = 1, lambda_network = 1):
    '''
    Returns the efficiency by the weightings of 
    lambda_cpu, lambda_memory, lambda_disk, lambda_network and calculated efficiencies.
    The lambdas are normalized to sum up to 1.

    efficiency: Float Float Float Float (dictof Str (anyof Str Float)) -> Float
    '''
    sum_lambda = lambda_cpu + lambda_memory + lambda_disk + lambda_network
    lambda_cpu = lambda_cpu / sum_lambda
    lambda_memory = lambda_memory / sum_lambda
    lambda_disk = lambda_disk / sum_lambda
    lambda_network = lambda_network / sum_lambda

    return (
        lambda_cpu * efficiencies['cpu'] + 
        lambda_memory * efficiencies['memory'] + 
        lambda_disk * efficiencies['disk'] + 
        lambda_network * efficiencies['network']
    )

'''
  {
    't1': [{}, {}],
    't2': [{}, {}],
  } => 
  {
    't1': {},
    't2': {},
  }
  '''
def get_efficiencies_per_timestamp(process_dict):
    '''
    Returns a new dictionary with keys of timestamps and values of 
    summed resource utilization logs from process_dict.

    get_efficiencies_per_timestamp: (dictof Str (listof (dictof (anyof Str Float)))) -> 
        (dictof Str (dictof Str (anyof Str Float)))
    '''
    results = {}
    for key in process_dict:
        lod = process_dict[key]
        results[key] = reduce(lambda t, n: dict_sum(t, n, ['unix']), lod)
    return results

def dict_sum(dict_one, dict_two, ignore_keys):
    '''
    Returns a new dictionary which has the same keys as dict_one
    and adds the value of dict_two for each key except the keys in ignore_keys.

    dict_sum: (dictof Str (anyof Str Float)) (dictof Str (anyof Str Float)) ->
        (dictof Str (anyof Str Float))
    '''
    result = {}
    for key in dict_one:
        result[key] = dict_one[key]
        if (key in ignore_keys): continue
        result[key] += dict_two[key]
    return result

def log_efficiency(data, lambda_cpu = 1, lambda_memory = 1, lambda_disk = 1, lambda_network = 1):
    efficiencies = get_efficiencies(data)

    print(efficiencies)
    result = efficiency(
            efficiencies, 
            lambda_cpu, 
            lambda_memory,
            lambda_disk,
            lambda_network
        )
    print(result)
    return result
def get_average_efficiency(efficiencies_per_timestamp, lambda_cpu = 1, lambda_memory = 1, lambda_disk = 1, lambda_network = 1):
    '''
    Returns the average efficiency of efficiencies_per_timestamp.

    get_average_efficiency: (dictof Str (dictof Str (anyof Str Float))) -> Float
    '''
    total_efficiency = sum(map(
        lambda data: log_efficiency(data, 
            lambda_cpu, 
            lambda_memory,
            lambda_disk,
            lambda_network
        ), 
        efficiencies_per_timestamp.values()
    ))
    average_efficiency = total_efficiency / len(efficiencies_per_timestamp)

    return average_efficiency
