from datasets import load_dataset, get_dataset_config_names
import json

out = {}
try:
    out['livebench/math'] = get_dataset_config_names('livebench/math')
except Exception as e:
    out['livebench/math'] = str(e)

try:
    out['LiveBench/LiveBench'] = get_dataset_config_names('LiveBench/LiveBench')
except Exception as e:
    out['LiveBench/LiveBench'] = str(e)

with open('tmp_out.json', 'w') as f:
    json.dump(out, f, indent=2)
