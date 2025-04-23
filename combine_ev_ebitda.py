import os
import pandas as pd

folder = "attached_assets"

# Only process files matching these base names
valid_tickers = {
    "EZJ.L", "RYA.I", "WIZZ.L", "LHAG.DE", "ICAG.L", "AIRF.PA", "JET2.L"
}

results = []

for ticker in valid_tickers:
    filename = f"{ticker}.xlsx"
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print(f"File not found for ticker: {ticker}")
        results.append({"Ticker": ticker, "EV_EBITDA": None})
        continue

    try:
        # Read the default sheet from the Excel file
        df = pd.read_excel(filepath)

        # EV/EBITDA is at row 64 (index 63 in pandas)
        ev_row = df.iloc[63]

        # Drop first column (label) and get the latest non-null value
        value_series = ev_row.iloc[1:]
        non_null_values = value_series.dropna()
        ev_ebitda_value = non_null_values.iloc[-1] if not non_null_values.empty else None

        print(f"{ticker} EV/EBITDA extracted: {ev_ebitda_value}")
        results.append({"Ticker": ticker, "EV_EBITDA": ev_ebitda_value})

    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        results.append({"Ticker": ticker, "EV_EBITDA": None})

# Save combined results
combined_df = pd.DataFrame(results)
output_csv = os.path.join(folder, "ev_ebitda_combined.csv")
combined_df.to_csv(output_csv, index=False)
print(f"\n✅ Combined EV/EBITDA data saved to {output_csv}")
