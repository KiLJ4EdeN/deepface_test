import os
import json
import shutil
import numpy as np
from distutils.dir_util import copy_tree

BASE_PATH = '/home/parsa/Desktop/arman/bankwork/'
DATA_PATH = 'data'
DATA_PATH = os.path.join(BASE_PATH, DATA_PATH)
OUT_PATH = './not_verified'

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)
else:
    shutil.rmtree(OUT_PATH)
    os.makedirs(OUT_PATH)

with open('verification.json', 'r') as f:
    data = json.load(f)

print(len(data))

for uid, ver in data.items():
    if ver <= 0.7:
        person_path = os.path.join(DATA_PATH, uid)
        out_path = os.path.join(OUT_PATH, f'{np.around(ver * 100, 2)}--' + uid)
        os.makedirs(out_path, exist_ok=True)
        copy_tree(person_path, out_path)

