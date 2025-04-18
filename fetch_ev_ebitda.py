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

# Fetch and save EV/EBITDA data
# for ric, company in tickers.items():
#     try:
#         df = ek.get_data(
#             instruments=ric,
#             fields=fields,
#             parameters={"SDate": "0", "EDate": "0"}
#         )[0]

#         if df is not None and not df.empty:
#             df["Company"] = company
#             df = df.rename(columns={
#                 "Total Enterprise Value": "EV",
#                 "EBITDA": "EBITDA"
#             })
#             df = df[["Company", "EV", "EBITDA"]]
#             filename = f"{ric.replace('.', '_')}_ev.csv"
#             df.to_csv(os.path.join(output_dir, filename), index=False)
#             print(f"Saved EV/EBITDA data for {company} to {filename}")
#         else:
#             print(f"No data returned for {company} ({ric})")

#     except Exception as e:
#         print(f"Error fetching data for {company} ({ric}): {e}")

# TEMPORARY TEST LOOP – Just for EZJ.L
# TEMPORARY TEST LOOP – Just for EZJ.L
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
