from datasets import load_dataset
import json

try:
    ds = load_dataset('livebench/math', split='test')
except Exception as e:
    ds = load_dataset('livebench/math', 'default', split='test')

out = {}
if len(ds) > 0:
    for k, v in ds[0].items():
        out[k] = str(type(v)) + " | " + str(v)[:100]
else:
    out['error'] = 'Dataset empty'

with open('tmp_out_2.json', 'w') as f:
    json.dump({'len': len(ds), 'sample': out}, f, indent=2)
