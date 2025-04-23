import streamlit as st
# Set page title
st.set_page_config(page_title="EasyJet DCF Model", layout="wide")

import pandas as pd
import os
from utils import load_excel_file
from dcf_analyzer import DCFAnalyzer


st.title("EasyJet DCF Model Analysis")

# Load the Excel file
file_path = "attached_assets/EasyJet- complete.xlsx"

if os.path.exists(file_path):
    st.info(f"Found Excel file: {file_path}")
    try:
        # Load Excel file
        excel_data = pd.read_excel(file_path, sheet_name=None)

        # Print available sheets
        st.write("Available sheets:", list(excel_data.keys()))

        if 'DCF' in excel_data:
            st.success("Found DCF tab in the Excel file")

            # Display raw DCF data
            with st.expander("Show raw DCF data (first 20 rows)", expanded=False):
                st.dataframe(excel_data['DCF'].head(20))

            # Create DCF Analyzer
            dcf_analyzer = DCFAnalyzer(excel_data['DCF'])

            # Display all visualizations
            dcf_analyzer.display_all_visualizations()
        else:
            st.error("No 'DCF' tab found in the Excel file")

    except Exception as e:
        st.error(f"Error loading the file: {str(e)}")
        st.exception(e)
else:
    st.error(f"Excel file not found: {file_path}")
