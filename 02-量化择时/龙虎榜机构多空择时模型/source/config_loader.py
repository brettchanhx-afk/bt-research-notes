import json
import os
import sys
from pathlib import Path

cur_root = Path(__file__).absolute().parent

os.chdir(cur_root)

sys.path.append(str(cur_root.parent))

# print(os.listdir('..'))
__all__ = ['ts_token']

with open(r'config.json', 'r') as file:

    config = json.loads(file.read())

ts_token = config['ts_token']
