import os
import sys
import time
import yfinance as yf

from pprint import pprint

import trafilatura
import uuid
import json
import random

from openai import OpenAI

client = OpenAI(
    api_key = "yoursecretkey"
)

full_path  = '/mnt/StorageMedia/dzambala_summ/part_001'

from llm.gemma import tokenizer
tokenizer = tokenizer.Tokenizer(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'llm',
        'checkpoints',
        'tokenizer.model'
    )
)

'''
models = client.models.list()
for model in models.data:
    print(model.id)
'''

def ask_gpt(prompt: str, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

uuids = set()

while True:
    with open(os.path.dirname(__file__) + "/topics.txt", "r") as f:
        for line in f:
            line = line.strip()
            if line:
                news = yf.Search(line, news_count=4).news
                for count, item in enumerate(news):
                    try:
                        uid = item.get("uuid", str(uuid.uuid4()))
                        filename = f'{line.replace(" ", "_").lower()}.{uid}.json'
                        if os.path.isfile(os.path.join(full_path, filename)):
                            continue
                        if uid in uuids:
                            continue
                        uuids.add(uid)
                        url = item.get('link')
                        downloaded = trafilatura.fetch_url(url)
                        text = trafilatura.extract(downloaded)
                        downloaded = 'None'
                        text = text.split('More News')[0]
                        text_enc = tokenizer.encode(text, bos=False)
                        text = tokenizer.decode(text_enc[:384])
                        text = f'{item["title"]} {text}'
                        summ = ask_gpt(f'Summarise in {random.randint(24, 32)} tokens: {text}')
                        item['prompt'] = text
                        item['summ'] = summ.strip()
                        with open(os.path.join(full_path, filename), "w") as f:
                            json.dump(item, f, indent=4, default=str)
                        print ('=====')
                        print (filename)
                        print (item.get('title'))
                        print (item.get('summ'))
                    except Exception as e:
                        print (f'!===\n{e}\n!===')

                    time.sleep(1)

    for item in random.sample(uuids, min(2, len(uuids))):
        uuids.remove(item)
