# tools/build_tail_age.py
# python tools/build_tail_age.py \
#  --infile data/MASTER.txt \
#  --outfile data/aircraft_registry_clean.csv


import argparse
from pathlib import Path
import pandas as pd
import re

DEFAULT_IN = Path("data/external/faa/MASTER.txt")
DEFAULT_OUT = Path("data/aircraft_registry_clean.csv")

def read_with_header(path: Path) -> pd.DataFrame:
    # Your file is comma-delimited with headers; keep a robust fallback anyway
    for sep in [",","|",";","\t"]:
        for enc in ["utf-8","latin1"]:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, dtype=str)
                if df.shape[1] >= 2:
                    return df
            except Exception:
                pass
    raise RuntimeError("Could not read MASTER.txt with a header.")

from typing import Optional

def normalize_tail(v: str) -> Optional[str]:
    if pd.isna(v):
        return None
    s = str(v).strip().upper().replace(" ", "")
    if not s:
        return None
    # FAA MASTER often omits the leading 'N' in this column
    if not s.startswith("N"):
        s = "N" + s
    # keep only alphanumerics after the leading N
    s = "N" + re.sub(r"[^A-Z0-9]", "", s[1:])
    return s if len(s) > 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default=str(DEFAULT_IN))
    ap.add_argument("--outfile", default=str(DEFAULT_OUT))
    # Exact header names based on your sample:
    ap.add_argument("--tail-col", default="N-NUMBER")
    ap.add_argument("--year-col", default="YEAR MFR")
    args = ap.parse_args()

    path = Path(args.infile)
    if not path.exists():
        raise FileNotFoundError(path)

    df = read_with_header(path)

    # Use the exact columns you pasted
    if args.tail_col not in df.columns or args.year_col not in df.columns:
        raise RuntimeError(
            f"Required columns not found. Have: {list(df.columns)[:10]} ..."
        )

    out = df[[args.tail_col, args.year_col]].rename(
        columns={args.tail_col: "Tail_Number", args.year_col: "Year_Mfr"}
    ).copy()

    # Normalize tails (add leading 'N', strip spaces, etc.)
    out["Tail_Number"] = out["Tail_Number"].apply(normalize_tail)

    # Coerce year; keep rows even if Year_Mfr is NaN (we can bucket as Unknown later)
    out["Year_Mfr"] = pd.to_numeric(out["Year_Mfr"], errors="coerce")

    # Drop rows with missing tail after normalization
    out = out[out["Tail_Number"].notna()]

    # (Optional) sanity: keep plausible years; but DON'T drop NaN years here
    this_year = pd.Timestamp.now().year
    valid_year_mask = out["Year_Mfr"].between(1940, this_year + 1, inclusive="both") | out["Year_Mfr"].isna()
    out = out[valid_year_mask]

    # Deduplicate
    out = out.drop_duplicates(subset=["Tail_Number"])

    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[OK] Wrote {out_path} with {len(out)} rows")

if __name__ == "__main__":
    main()
