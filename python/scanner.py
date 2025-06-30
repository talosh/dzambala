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
    x.to_csv(os.path.join(full_path, "index_dji.csv"), index=False)
    x = yf.Lookup("^IXIC").index
    x.to_csv(os.path.join(full_path, "index_nasdaq.csv"), index=False)
    x = yf.Lookup("^NYA").index
    x.to_csv(os.path.join(full_path, "index_nya.csv"), index=False)
    x = yf.Lookup("^XAX").index
    x.to_csv(os.path.join(full_path, "index_xax.csv"), index=False)
    x = yf.Lookup("^BUK100P").index
    x.to_csv(os.path.join(full_path, "index_buk100p.csv"), index=False)
    x = yf.Lookup("^RUT").index
    x.to_csv(os.path.join(full_path, "index_rut.csv"), index=False)
    x = yf.Lookup("^VIX").index
    x.to_csv(os.path.join(full_path, "index_vix.csv"), index=False)
    x = yf.Lookup("^FTSE").index
    x.to_csv(os.path.join(full_path, "index_ftse.csv"), index=False)
    x = yf.Lookup("^GDAXI").index
    x.to_csv(os.path.join(full_path, "index_gdaxi.csv"), index=False)
    x = yf.Lookup("^FCHI").index
    x.to_csv(os.path.join(full_path, "index_fchi.csv"), index=False)
    x = yf.Lookup("^STOXX50E").index
    x.to_csv(os.path.join(full_path, "index_stoxx50e.csv"), index=False)
    x = yf.Lookup("^N100").index
    x.to_csv(os.path.join(full_path, "index_n100.csv"), index=False)
    x = yf.Lookup("^BFX").index
    x.to_csv(os.path.join(full_path, "index_bfx.csv"), index=False)
    x = yf.Lookup("MOEX.ME").index
    x.to_csv(os.path.join(full_path, "index_mdexme.csv"), index=False)
    x = yf.Lookup("^HSI").index
    x.to_csv(os.path.join(full_path, "index_hsi.csv"), index=False)
    x = yf.Lookup("^STI").index
    x.to_csv(os.path.join(full_path, "index_sti.csv"), index=False)
    x = yf.Lookup("^AXJO").index
    x.to_csv(os.path.join(full_path, "index_axjo.csv"), index=False)
    x = yf.Lookup("^AORD").index
    x.to_csv(os.path.join(full_path, "index_aord.csv"), index=False)
    x = yf.Lookup("^BSESN").index
    x.to_csv(os.path.join(full_path, "index_bsesn.csv"), index=False)
    x = yf.Lookup("^JKSE").index
    x.to_csv(os.path.join(full_path, "index_jkse.csv"), index=False)
    x = yf.Lookup("^KLSE").index
    x.to_csv(os.path.join(full_path, "index_klse.csv"), index=False)
    x = yf.Lookup("^NZ50").index
    x.to_csv(os.path.join(full_path, "index_nz50.csv"), index=False)
    x = yf.Lookup("^KS11").index
    x.to_csv(os.path.join(full_path, "index_ks11.csv"), index=False)
    x = yf.Lookup("^TWII").index
    x.to_csv(os.path.join(full_path, "index_twii.csv"), index=False)
    x = yf.Lookup("^GSPTSE").index
    x.to_csv(os.path.join(full_path, "index_gsptse.csv"), index=False)
    x = yf.Lookup("^BVSP").index
    x.to_csv(os.path.join(full_path, "index_bvsp.csv"), index=False)
    x = yf.Lookup("^MXX").index
    x.to_csv(os.path.join(full_path, "index_mxx.csv"), index=False)
    x = yf.Lookup("^IPSA").index
    x.to_csv(os.path.join(full_path, "index_ipsa.csv"), index=False)
    x = yf.Lookup("^MERV").index
    x.to_csv(os.path.join(full_path, "index_merv.csv"), index=False)
    x = yf.Lookup("^TA125.TA").index
    x.to_csv(os.path.join(full_path, "index_ta125.csv"), index=False)
    x = yf.Lookup("^CASE30").index
    x.to_csv(os.path.join(full_path, "index_case30.csv"), index=False)
    x = yf.Lookup("^JN0U.JO").index
    x.to_csv(os.path.join(full_path, "index_jn0u.csv"), index=False)
    x = yf.Lookup("^125904-USD-STRD").index
    x.to_csv(os.path.join(full_path, "index_125904.csv"), index=False)
    x = yf.Lookup("^XDB").index
    x.to_csv(os.path.join(full_path, "index_xdb.csv"), index=False)
    x = yf.Lookup("^XDE").index
    x.to_csv(os.path.join(full_path, "index_xde.csv"), index=False)
    x = yf.Lookup("000001.SS").index
    x.to_csv(os.path.join(full_path, "index_000001.csv"), index=False)
    x = yf.Lookup("^N225").index
    x.to_csv(os.path.join(full_path, "index_n225.csv"), index=False)
    x = yf.Lookup("^XDN").index
    x.to_csv(os.path.join(full_path, "index_xdn.csv"), index=False)
    x = yf.Lookup("^XDA").index
    x.to_csv(os.path.join(full_path, "index_xda.csv"), index=False)

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
    x = yf.Search("Government", news_count=10).news
    with open(os.path.join(full_path, "news_gov.json"), "w") as f:
        json.dump(x, f, indent=4, default=str)
    x = yf.Search("Court", news_count=10).news
    with open(os.path.join(full_path, "news_court.json"), "w") as f:
        json.dump(x, f, indent=4, default=str)
    x = yf.Search("UK", news_count=10).news
    with open(os.path.join(full_path, "news_uk.json"), "w") as f:
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