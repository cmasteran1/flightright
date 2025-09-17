# src/fetch_all_raw.py

#python fetch_all_flight.py --start 2024-03 --end 2024-06

import os, zipfile
from io import BytesIO
from pathlib import Path
import pandas as pd, requests

BASE_URL = "https://transtats.bts.gov/PREZIP/On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_{y}_{m}.zip"
RAW_DIR = Path("../../data/raw_bts")

KEEP = [
"Year","Quarter","Month","DayofMonth","DayOfWeek","FlightDate","Marketing_Airline_Network","Operated_or_Branded_Code_Share_Partners","DOT_ID_Marketing_Airline","IATA_Code_Marketing_Airline","Flight_Number_Marketing_Airline","Originally_Scheduled_Code_Share_Airline","DOT_ID_Originally_Scheduled_Code_Share_Airline","IATA_Code_Originally_Scheduled_Code_Share_Airline","Flight_Num_Originally_Scheduled_Code_Share_Airline","Operating_Airline ","DOT_ID_Operating_Airline","IATA_Code_Operating_Airline","Tail_Number","Flight_Number_Operating_Airline","OriginAirportID","OriginAirportSeqID","OriginCityMarketID","Origin","OriginCityName","OriginState","OriginStateFips","OriginStateName","OriginWac","DestAirportID","DestAirportSeqID","DestCityMarketID","Dest","DestCityName","DestState","DestStateFips","DestStateName","DestWac","CRSDepTime","DepTime","DepDelay","DepDelayMinutes","DepDel15","DepartureDelayGroups","DepTimeBlk","TaxiOut","WheelsOff","WheelsOn","TaxiIn","CRSArrTime","ArrTime","ArrDelay","ArrDelayMinutes","ArrDel15","ArrivalDelayGroups","ArrTimeBlk","Cancelled","CancellationCode","Diverted","CRSElapsedTime","ActualElapsedTime","AirTime","Flights","Distance","DistanceGroup","CarrierDelay","WeatherDelay","NASDelay","SecurityDelay","LateAircraftDelay","FirstDepTime","TotalAddGTime","LongestAddGTime"
]

def fetch_month(year, month):
    url = BASE_URL.format(y=year, m=month)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(BytesIO(r.content)) as z:
        csv = [n for n in z.namelist() if n.endswith(".csv")][0]
        with z.open(csv) as f:
            df = pd.read_csv(f, low_memory=False)
    return df

def save_parquet(df, year, month):
    out_dir = RAW_DIR / f"Year={year}" / f"Month={month:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # keep minimal columns, coerce FlightDate to datetime
    df = df[KEEP].copy()
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    df.to_parquet(out_dir / "flights.parquet", index=False)
    print("[OK]", out_dir)

if __name__ == "__main__":
    import argparse, datetime
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)  # YYYY-MM
    p.add_argument("--end", required=True)    # YYYY-MM
    args = p.parse_args()

    from dateutil.relativedelta import relativedelta
    cur = pd.Timestamp(args.start + "-01")
    end = pd.Timestamp(args.end + "-01")

    while cur <= end:
        y, m = cur.year, cur.month
        try:
            df = fetch_month(y, m)
            save_parquet(df, y, m)
        except Exception as e:
            print("[WARN]", y, m, e)
        cur += relativedelta(months=1)
