import json
from collections import defaultdict
import pickle as pkl
from tqdm.auto import tqdm
import os.path
from tqdm import tqdm

with open('inspired/train_data_processed.jsonl', 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        dialog = json.loads(line)
        print(dialog)
        break

