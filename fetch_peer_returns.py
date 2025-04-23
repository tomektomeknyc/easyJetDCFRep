import refinitiv.dataplatform.eikon as ek
import pandas as pd
import os
import json
from datetime import datetime, timedelta

# Set your Refinitiv App Key once at import time
ek.set_app_key("cb64a413a4f04804b8a6a82bde9c35087f7f819c")

CACHE_PATH = "attached_assets/analyst_targets.json"


def _load_cache() -> dict[str, float]:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict[str, float]) -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def fetch_analyst_target_refinitiv(ticker: str) -> float | None:
    """
    Fetch or retrieve from cache the consensus mean analyst target price for a given ticker.
    """
    cache = _load_cache()

    try:
        df, err = ek.get_data(ticker, ["TR.PriceTargetMean"])
    except Exception:
        # network/API error: fall back to cache
        return cache.get(ticker)

    col = "Price Target - Mean"
    if err or df.empty or df[col].isna().all():
        # no data: maybe return cache
        return cache.get(ticker)

    # got a fresh value
    target = float(df[col].iat[0])
    cache[ticker] = target
    _save_cache(cache)
    return target

def main() -> None:
    """
    Fetch 10 years of daily returns for a set of peer tickers and
    save each to a CSV, and demonstrate fetching a Refinitiv target.
    """
    # Example usage of the Fetch function
    example_ticker = "EZJ.L"
    target = fetch_analyst_target_refinitiv(example_ticker)
    print(f"{example_ticker} analyst mean target: {target}")

    # Define the date range: last 10 years until today
    today = datetime.today().strftime("%Y-%m-%d")
    ten_years_ago = (datetime.today() - timedelta(days=10 * 365)).strftime("%Y-%m-%d")

    # List of peer tickers
    tickers = ["RYA.I", "WIZZ.L", "LHAG.DE", "ICAG.L", "AIRF.PA", "JET2.L", "KNIN.S"]

    for ticker in tickers:
        print(f"Fetching data for {ticker} from {ten_years_ago} to {today}...")
        data = ek.get_timeseries(
            rics=ticker,
            fields="CLOSE",
            start_date=ten_years_ago,
            end_date=today,
            interval="daily"
        )

        if data is None or data.empty:
            print(f"No data returned or data is empty for {ticker}.")
            continue

        # Calculate daily returns and drop NaNs
        data["Returns"] = data["CLOSE"].pct_change()
        data.dropna(subset=["Returns"], inplace=True)

        # Write out to CSV
        safe_ticker = ticker.replace(".", "_")
        csv_filename = f"attached_assets/{safe_ticker}_returns.csv"
        data.to_csv(csv_filename)
        print(f"Saved returns for {ticker} to {csv_filename}")


if __name__ == "__main__":
    main()
