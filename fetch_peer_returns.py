import refinitiv.dataplatform.eikon as ek
import pandas as pd
from datetime import datetime, timedelta

def main():
    # Replace with your actual Refinitiv App Key
    YOUR_API_KEY = "cb64a413a4f04804b8a6a82bde9c35087f7f819c"
    ek.set_app_key(YOUR_API_KEY)

    # Define the date range: last 10 years until today
    today = datetime.today().strftime("%Y-%m-%d")
    ten_years_ago = (datetime.today() - timedelta(days=10 * 365)).strftime("%Y-%m-%d")

    # List of tickers for the peer companies (change periods '.' in filenames later)
    tickers = ["RYA.I", "WIZZ.L", "LHAG.DE", "ICAG.L", "AIRF.PA", "JET2.L", "KNIN.S"]

    # Loop through each ticker, get data, calculate returns and save to CSV
    for ticker in tickers:
        print(f"Fetching data for {ticker} from {ten_years_ago} to {today}...")
        # Fetch daily close prices using the Eikon function
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

        # Calculate daily returns based on the "CLOSE" column
        data["Returns"] = data["CLOSE"].pct_change()
        data.dropna(subset=["Returns"], inplace=True)

        # Create a filename; replace periods with underscores to avoid issues in filenames
        csv_filename = f"attached_assets/{ticker.replace('.', '_')}_returns.csv"
        data.to_csv(csv_filename)
        print(f"Historical returns for {ticker} saved to {csv_filename}")

if __name__ == "__main__":
    main()
