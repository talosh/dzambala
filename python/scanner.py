import os
import time
from datetime import datetime
import yfinance as yf
import json

# Define your dataset path
dataset_path = '/mnt/projects/dzambala_data'

now = datetime.now()
folder_name = now.strftime('%Y%m%d_%H%M')  # Format: YYYYMMDD_HHMM
full_path = os.path.join(dataset_path, folder_name)
os.makedirs(full_path, exist_ok=True)

cycle_start = time.time()

try:

    # ===

    x = yf.Lookup("XMR-GBP").get_cryptocurrency(count=1)
    x.to_csv(os.path.join(full_path, "crypto_xmr_gbp.csv"), index=False)
    x = yf.Lookup("XMR-USD").get_cryptocurrency(count=1)
    x.to_csv(os.path.join(full_path, "crypto_xmr_usd.csv"), index=False)

    # ===

    x = yf.Lookup("GSPC").index
    x.to_csv(os.path.join(full_path, "index_gspc.csv"), index=False)
    x = yf.Lookup("DX-Y.NYB").index
    x.to_csv(os.path.join(full_path, "index_usd.csv"), index=False)
    x = yf.Lookup("^DJI").index
    x.to_csv(os.path.join(full_path, "index_usd.csv"), index=False)

    # ===

    x = yf.Search("Crypto", news_count=10).news
    with open(os.path.join(full_path, "news_crypto.json"), "w") as f:
        json.dump(x, f, indent=4, default=str)
    x = yf.Search("Finance", news_count=10).news
    with open(os.path.join(full_path, "news_finance.json"), "w") as f:
        json.dump(x, f, indent=4, default=str)
    x = yf.Search("Asia", news_count=10).news
    with open(os.path.join(full_path, "news_asia.json"), "w") as f:
        json.dump(x, f, indent=4, default=str)
    x = yf.Search("AI", news_count=10).news
    with open(os.path.join(full_path, "news_ai.json"), "w") as f:
        json.dump(x, f, indent=4, default=str)
    x = yf.Search("Senate", news_count=10).news
    with open(os.path.join(full_path, "news_senate.json"), "w") as f:
        json.dump(x, f, indent=4, default=str)
    x = yf.Search("Parliament", news_count=10).news
    with open(os.path.join(full_path, "news_parliament.json"), "w") as f:
        json.dump(x, f, indent=4, default=str)
    x = yf.Search("UK", news_count=10).news
    with open(os.path.join(full_path, "news_uk.json"), "w") as f:
        json.dump(x, f, indent=4, default=str)
    x = yf.Search("Government", news_count=10).news
    with open(os.path.join(full_path, "news_gov.json"), "w") as f:
        json.dump(x, f, indent=4, default=str)
    x = yf.Search("EU", news_count=10).news
    with open(os.path.join(full_path, "news_eu.json"), "w") as f:
        json.dump(x, f, indent=4, default=str)

    # ===

except Exception as e:
    print (e)

print(f"Completed: {full_path}")
elapsed = time.time() - cycle_start
sleep_time = max(0, 300 - elapsed)
time.sleep(sleep_time)