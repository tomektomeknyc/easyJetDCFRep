import refinitiv.dataplatform.eikon as ek
import pandas as pd
import os

# Set your Refinitiv Eikon API key here
ek.set_app_key("cb64a413a4f04804b8a6a82bde9c35087f7f819c")  # Replace with your actual API key

# Define tickers and company names
tickers = {
    "EZJ.L": "EasyJet",
    "RYA.I": "Ryanair",
    "WIZZ.L": "Wizz Air",
    "LHAG.DE": "Lufthansa",
    "ICAG.L": "IAG",
    "AIRF.PA": "Air France-KLM",
    "JET2.L": "Jet2"
}

# Fields to fetch
fields = ["TR.TotalEnterpriseValue", "TR.EBITDA"]

# Create output directory if it doesn't exist
output_dir = "attached_assets"
os.makedirs(output_dir, exist_ok=True)


try:
    df, err = ek.get_data(
    instruments="EZJ.L",
    fields=["*"]
)

    print(df.columns)



    print("Returned columns:")
    print(df.columns)
    print("\nData:")
    print(df)

except Exception as e:
   print(f"Error fetching test data: {e}")
