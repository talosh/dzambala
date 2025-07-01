import os
import time
from datetime import datetime
import yfinance as yf
import json

import signal
def create_graceful_exit():
    def graceful_exit(signum, frame):
        print()
        exit(0)
        # signal.signal(signum, signal.SIG_DFL)
        # os.kill(os.getpid(), signal.SIGINT)
    return graceful_exit
signal.signal(signal.SIGINT, create_graceful_exit())

# Define your dataset path
dataset_path = '/mnt/StorageMedia/dzambala_data'

def wait_until_next_5min():
    now = datetime.now()
    seconds_since_hour = now.minute * 60 + now.second
    seconds_until_next_5min = (300 - (seconds_since_hour % 300)) % 300
    if seconds_until_next_5min > 0:
        print(f"Waiting {seconds_until_next_5min:.2f} seconds until next 5-minute mark...")
        time.sleep(seconds_until_next_5min)

wait_until_next_5min()

while True:
    now = datetime.now()
    folder_name = now.strftime('%Y%m%d_%H%M')  # Format: YYYYMMDD_HHMM
    full_path = os.path.join(dataset_path, folder_name)
    os.makedirs(full_path, exist_ok=True)

    cycle_start = time.time()

    try:

        # ===

        x = yf.Lookup("XMR-GBP").get_cryptocurrency(count=1)
        x.to_csv(os.path.join(full_path, "crypto_xmrgbp.csv"), index=False)
        x = yf.Lookup("XMR-USD").get_cryptocurrency(count=1)
        x.to_csv(os.path.join(full_path, "crypto_xmrusd.csv"), index=False)

        x = yf.Lookup("BTC-GBP").get_cryptocurrency(count=1)
        x.to_csv(os.path.join(full_path, "crypto_btcgbp.csv"), index=False)
        x = yf.Lookup("BTC-USD").get_cryptocurrency(count=1)
        x.to_csv(os.path.join(full_path, "crypto_btcusd.csv"), index=False)
        x = yf.Lookup("BTC-EUR").get_cryptocurrency(count=1)
        x.to_csv(os.path.join(full_path, "crypto_btceur.csv"), index=False)
        x = yf.Lookup("BTC-CAD").get_cryptocurrency(count=1)
        x.to_csv(os.path.join(full_path, "crypto_btccad.csv"), index=False)

        x = yf.Lookup("ETH-GBP").get_cryptocurrency(count=1)
        x.to_csv(os.path.join(full_path, "crypto_ethgbp.csv"), index=False)
        x = yf.Lookup("ETH-USD").get_cryptocurrency(count=1)
        x.to_csv(os.path.join(full_path, "crypto_ethusd.csv"), index=False)
        x = yf.Lookup("ETH-EUR").get_cryptocurrency(count=1)
        x.to_csv(os.path.join(full_path, "crypto_etheur.csv"), index=False)
        x = yf.Lookup("ETH-CAD").get_cryptocurrency(count=1)
        x.to_csv(os.path.join(full_path, "crypto_ethcad.csv"), index=False)

        x = yf.Lookup("XRP-GBP").get_cryptocurrency(count=1)
        x.to_csv(os.path.join(full_path, "crypto_xrpgbp.csv"), index=False)
        x = yf.Lookup("XRP-USD").get_cryptocurrency(count=1)
        x.to_csv(os.path.join(full_path, "crypto_xrpusd.csv"), index=False)
        x = yf.Lookup("XRP-EUR").get_cryptocurrency(count=1)
        x.to_csv(os.path.join(full_path, "crypto_xrpeur.csv"), index=False)
        x = yf.Lookup("XRP-CAD").get_cryptocurrency(count=1)
        x.to_csv(os.path.join(full_path, "crypto_xrpcad.csv"), index=False)

        x = yf.Lookup("CFX-USD").get_cryptocurrency(count=1)
        x.to_csv(os.path.join(full_path, "crypto_cfxusd.csv"), index=False)
        # x = yf.Lookup("CFX-EUR").get_cryptocurrency(count=1)
        # x.to_csv(os.path.join(full_path, "crypto_cfxeur.csv"), index=False)
        # x = yf.Lookup("CFX-CAD").get_cryptocurrency(count=1)
        # x.to_csv(os.path.join(full_path, "crypto_cfxcad.csv"), index=False)

        x = yf.Lookup("PEPE24478-USD").get_cryptocurrency(count=1)
        x.to_csv(os.path.join(full_path, "crypto_pepeusd.csv"), index=False)

        # === Indexes

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
        # x = yf.Lookup("MOEX.ME").index
        # x.to_csv(os.path.join(full_path, "index_mdexme.csv"), index=False)
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

        # === News

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
        x = yf.Search("Britain", news_count=10).news
        with open(os.path.join(full_path, "news_uk.json"), "w") as f:
            json.dump(x, f, indent=4, default=str)
        x = yf.Search("EU", news_count=10).news
        with open(os.path.join(full_path, "news_eu.json"), "w") as f:
            json.dump(x, f, indent=4, default=str)

        # === Features

        x = yf.Lookup("GC=F").future
        x.to_csv(os.path.join(full_path, "future_gold.csv"), index=False)
        x = yf.Lookup("CL=F").future
        x.to_csv(os.path.join(full_path, "future_oil.csv"), index=False)
        x = yf.Lookup("SI=F").future
        x.to_csv(os.path.join(full_path, "future_silver.csv"), index=False)
        x = yf.Lookup("HG=F").future
        x.to_csv(os.path.join(full_path, "future_copper.csv"), index=False)
        x = yf.Lookup("BZ=F").future
        x.to_csv(os.path.join(full_path, "future_brent.csv"), index=False)
        x = yf.Lookup("NG=F").future
        x.to_csv(os.path.join(full_path, "future_gas.csv"), index=False)
        x = yf.Lookup("ZC=F").future
        x.to_csv(os.path.join(full_path, "future_corn.csv"), index=False)
        x = yf.Lookup("ZO=F").future
        x.to_csv(os.path.join(full_path, "future_oat.csv"), index=False)
        x = yf.Lookup("KE=F").future
        x.to_csv(os.path.join(full_path, "future_wheat.csv"), index=False)
        x = yf.Lookup("ZS=F").future
        x.to_csv(os.path.join(full_path, "future_soya.csv"), index=False)
        x = yf.Lookup("GF=F").future
        x.to_csv(os.path.join(full_path, "future_wisdomtree.csv"), index=False)
        x = yf.Lookup("GF=F").future
        x.to_csv(os.path.join(full_path, "future_wisdomtree.csv"), index=False)
        x = yf.Lookup("HE=F").future
        x.to_csv(os.path.join(full_path, "future_lean.csv"), index=False)
        x = yf.Lookup("LE=F").future
        x.to_csv(os.path.join(full_path, "future_cattle.csv"), index=False)
        x = yf.Lookup("CC=F").future
        x.to_csv(os.path.join(full_path, "future_cocoa.csv"), index=False)
        x = yf.Lookup("KC=F").future
        x.to_csv(os.path.join(full_path, "future_coffee.csv"), index=False)
        x = yf.Lookup("CT=F").future
        x.to_csv(os.path.join(full_path, "future_cotton.csv"), index=False)
        # x = yf.Lookup("LBS=F").future
        # x.to_csv(os.path.join(full_path, "future_limber.csv"), index=False)
        x = yf.Lookup("OJ=F").future
        x.to_csv(os.path.join(full_path, "future_orangejuice.csv"), index=False)
        x = yf.Lookup("SB=F").future
        x.to_csv(os.path.join(full_path, "future_sugar.csv"), index=False)

        # === Currencies

        x = yf.Lookup("GBPUSD=X").currency
        x.to_csv(os.path.join(full_path, "currency_gbpusd.csv"), index=False)
        x = yf.Lookup("GBPEUR=X").currency
        x.to_csv(os.path.join(full_path, "currency_gbpeur.csv"), index=False)
        x = yf.Lookup("EURUSD=X").currency
        x.to_csv(os.path.join(full_path, "currency_eurusd.csv"), index=False)
        x = yf.Lookup("GBPJPY=X").currency
        x.to_csv(os.path.join(full_path, "currency_gbpjpy.csv"), index=False)
        x = yf.Lookup("JPY=X").currency
        x.to_csv(os.path.join(full_path, "currency_usdjpy.csv"), index=False)
        x = yf.Lookup("GBP=X").currency
        x.to_csv(os.path.join(full_path, "currency_usdgbp.csv"), index=False)
        x = yf.Lookup("GBPAUD=X").currency
        x.to_csv(os.path.join(full_path, "currency_gbpaud.csv"), index=False)
        x = yf.Lookup("GBPBRL=X").currency
        x.to_csv(os.path.join(full_path, "currency_gbpbrl.csv"), index=False)
        x = yf.Lookup("GBPCAD=X").currency
        x.to_csv(os.path.join(full_path, "currency_gbpcad.csv"), index=False)
        x = yf.Lookup("GBPCHF=X").currency
        x.to_csv(os.path.join(full_path, "currency_gbpchf.csv"), index=False)
        x = yf.Lookup("GBPCNY=X").currency
        x.to_csv(os.path.join(full_path, "currency_gbpcny.csv"), index=False)
        x = yf.Lookup("GBPINR=X").currency
        x.to_csv(os.path.join(full_path, "currency_gbpinr.csv"), index=False)
        x = yf.Lookup("GBPNOK=X").currency
        x.to_csv(os.path.join(full_path, "currency_gbpnok.csv"), index=False)
        x = yf.Lookup("GBPQAR=X").currency
        x.to_csv(os.path.join(full_path, "currency_gbpqar.csv"), index=False)
        x = yf.Lookup("GBPZAR=X").currency
        x.to_csv(os.path.join(full_path, "currency_gbpzar.csv"), index=False)
        x = yf.Lookup("EURCHF=X").currency
        x.to_csv(os.path.join(full_path, "currency_eurchf.csv"), index=False)
        x = yf.Lookup("EURCAD=X").currency
        x.to_csv(os.path.join(full_path, "currency_eurcad.csv"), index=False)
        x = yf.Lookup("EURJPY=X").currency
        x.to_csv(os.path.join(full_path, "currency_eurjpy.csv"), index=False)
        x = yf.Lookup("EURSEK=X").currency
        x.to_csv(os.path.join(full_path, "currency_eursek.csv"), index=False)
        x = yf.Lookup("EURHUF=X").currency
        x.to_csv(os.path.join(full_path, "currency_eurhuf.csv"), index=False)
        x = yf.Lookup("CAD=X").currency
        x.to_csv(os.path.join(full_path, "currency_usdcad.csv"), index=False)
        # x = yf.Lookup("USDHKD=X").currency
        # x.to_csv(os.path.join(full_path, "currency_usdhkd.csv"), index=False)
        # x = yf.Lookup("USDSGD=X").currency
        # x.to_csv(os.path.join(full_path, "currency_usdsgd.csv"), index=False)
        x = yf.Lookup("INR=X").currency
        x.to_csv(os.path.join(full_path, "currency_usdinr.csv"), index=False)
        # x = yf.Lookup("USDMXN=X").currency
        # x.to_csv(os.path.join(full_path, "currency_usdmxn.csv"), index=False)
        x = yf.Lookup("CNY=X").currency
        x.to_csv(os.path.join(full_path, "currency_usdcny.csv"), index=False)
        x = yf.Lookup("CHF=X").currency
        x.to_csv(os.path.join(full_path, "currency_usdchf.csv"), index=False)
        x = yf.Lookup("RUB=X").currency
        x.to_csv(os.path.join(full_path, "currency_usdrub.csv"), index=False)
        x = yf.Lookup("UAH=X").currency
        x.to_csv(os.path.join(full_path, "currency_usduah.csv"), index=False)


    except Exception as e:
        print (e)

    print(f"Completed: {full_path}")
    elapsed = time.time() - cycle_start
    sleep_time = max(0, 300 - elapsed)
    time.sleep(sleep_time)