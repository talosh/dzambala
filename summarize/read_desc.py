import os
import sys
import json

def sanitize_name(name_to_sanitize):
    if name_to_sanitize is None:
        return None
    
    import re

    stripped_name = name_to_sanitize.strip()
    exp = re.compile(u'[^\w\.-]', re.UNICODE)

    result = exp.sub('_', stripped_name)
    return re.sub('_\_+', '_', result)


file_path = os.path.join(
    os.path.dirname(__file__),
    'descriptions.txt'
    )

out_dir = os.path.join(
    os.path.dirname(__file__),
    'desc'
    )

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.read().strip().split('\n')

data = []
block = []

for line in lines:
    if line.strip() == '':
        if len(block) == 2:
            data.append({'prompt': block[0], 'summ': block[1]})
        block = []
    else:
        block.append(line.strip())

# Handle the last block if the file does not end with an empty line
if len(block) == 2:
    data.append({'prompt': block[0], 'summ': [block[1]]})

# Example: print the result
for item in data:
    json_file_name = f'{sanitize_name(item["prompt"].lower())}.json'
    with open(os.path.join(out_dir, json_file_name), "w") as f:
        json.dump(item, f, indent=4, default=str)
    print (f'{json_file_name} ', end='')
    print(item)
