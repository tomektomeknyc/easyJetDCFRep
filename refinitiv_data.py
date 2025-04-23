# #!/usr/bin/env python3

# """
# Fetch EasyJet (EZJ.L) historical daily prices from Refinitiv Eikon Data API,
# calculate daily returns, and display the results.
# """

import refinitiv.dataplatform.eikon as ek
import pandas as pd
from datetime import datetime, timedelta

def main():
    # Set your Refinitiv App Key (replace with your actual key)
    YOUR_API_KEY = "cb64a413a4f04804b8a6a82bde9c35087f7f819c"
    ek.set_app_key(YOUR_API_KEY)

    # Define the look-back period: last 10 years
    today = datetime.today().strftime("%Y-%m-%d")
    ten_years_ago = (datetime.today() - timedelta(days=10 * 365)).strftime("%Y-%m-%d")

    print(f"Fetching EasyJet data from {ten_years_ago} to {today}...")

    # Fetch daily close prices for EasyJet using RIC "EZJ.L"
    # Note: The function now returns only the data
    data = ek.get_timeseries(
        rics="EZJ.L",
        fields="CLOSE",
        start_date=ten_years_ago,
        end_date=today,
        interval="daily"
    )

    if data is None or data.empty:
        print("No data returned or data is empty.")
        return

    # Calculate daily returns
    data["Returns"] = data["CLOSE"].pct_change()
    data.dropna(subset=["Returns"], inplace=True)

    # Display the first few rows
    print("First few rows of EasyJet historical data:")
    print(data.head())

    # Save the data to a CSV file for future use

    data.to_csv("attached_assets/easyjet_returns.csv")

    print("Historical returns saved to easyjet_returns.csv")

if __name__ == "__main__":
    main()
