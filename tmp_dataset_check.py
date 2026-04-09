from datasets import load_dataset, get_dataset_config_names
try:
    print(get_dataset_config_names('livebench/math'))
except Exception as e:
    print('Failed livebench/math:', e)

try:
    print(get_dataset_config_names('LiveBench/LiveBench'))
except Exception as e:
    print('Failed LiveBench/LiveBench:', e)
