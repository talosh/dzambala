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

full_path  = '/mnt/StorageMedia/dzambala_cnews/part_001'
if not os.path.isdir(full_path):
    os.makedirs(full_path)

from llm.gemma import tokenizer
tokenizer = tokenizer.Tokenizer(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'llm',
        'checkpoints',
        'tokenizer.model'
    )
)

import signal
def create_graceful_exit():
    def graceful_exit(signum, frame):
        print ('')
        exit(0)
    return graceful_exit
signal.signal(signal.SIGINT, create_graceful_exit())

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

def uid_short(text: str, length: int = 24) -> str:
    import hashlib
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:length]

def get_cryptonews_com():
    import feedparser
    from bs4 import BeautifulSoup


    FEED_URL = "https://cryptonews.com/news/feed/"

    feed = feedparser.parse(FEED_URL)
    articles = []

    for entry in feed.entries:
        title = entry.title
        html_summary = entry.summary
        soup = BeautifulSoup(html_summary, "html.parser")
        # Extract only <p> tags and join them as the article summary
        text = ' '.join(p.get_text(strip=True) for p in soup.find_all('p'))

        articles.append({
            "title": title,
            "text_body": text,
            "uuid": uid_short(title)
        })

    return articles

gpt_prompt = '''
            Summarise, 
            '''

VMIN = 4
VMAX = 6

while True:
    ask_gpt('Preserve consistent formatting key: values, Do not report number of tokens')
    # CryptoNews
    try:
        news = get_cryptonews_com()
        for count, item in enumerate(news):
            uid = item.get("uuid", str(uid_short(str(item.get("title")))))
            filename = f'cryptonews.{uid}.json'
            if os.path.isfile(os.path.join(full_path, filename)):
                continue
            if uid in uuids:
                continue
            text = f'{item["title"]} {item["text_body"]}'
            text_enc = tokenizer.encode(text, bos=False)
            text = tokenizer.decode(text_enc[:96])

            for idx in range(random.randint(VMIN, VMAX)):
                # ask_gpt('')
                summ = ask_gpt(f'{gpt_prompt} return in 11 tokens: {text}')
                item['prompt'] = text
                item['summ'] = summ.strip()
                filename_add, ext = os.path.splitext(filename)
                filename_add = f'{filename_add}.{idx:02d}{ext}'
                with open(os.path.join(full_path, filename_add), "w") as f:
                    json.dump(item, f, indent=4, default=str)
                print ('=====')
                print (filename_add)
                print (item.get('title'))
                print (item.get('summ'))
                print ('----------')
                print (item.get('prompt'))

            uuids.add(uid)
            # time.sleep(90)

    except Exception as e:
        print (f'!===\n{e}\n!===')
        time.sleep(90)

