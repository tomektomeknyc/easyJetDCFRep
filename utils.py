import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

def load_excel_file(uploaded_file):
    """
    Load Excel file and return a dictionary of DataFrames (one per sheet)

    Args:
        uploaded_file: The uploaded Excel file or file path

    Returns:
        dict: Dictionary containing DataFrames for each sheet
        str: Filename
    """
    try:
        # Read all sheets into a dictionary of DataFrames
        excel_file = pd.ExcelFile(uploaded_file)
        df_dict = {sheet_name: excel_file.parse(sheet_name) for sheet_name in excel_file.sheet_names}

        # Handle both file objects and path strings
        if hasattr(uploaded_file, 'name'):
            filename = uploaded_file.name
        else:
            # For a string path, extract just the filename
            import os
            filename = os.path.basename(str(uploaded_file))

        return df_dict, filename
    except Exception as e:
        st.error(f"Error reading Excel file: {str(e)}")
        return None, None

def extract_dcf_variables(df):
    """
    Extract DCF variables from specific cells in the DataFrame

    Args:
        df: DataFrame containing the DCF tab data

    Returns:
        dict: Dictionary of extracted DCF variables
    """
    try:
        # The cells might be 0-indexed in pandas vs 1-indexed in Excel
        # We'll try both approaches in case of issues

        # First attempt - direct Excel coordinates (0-indexed)
        try:
            # Excel cell E17 -> row 16, column 4 in 0-indexed DataFrame
            wacc = extract_numeric_value(df, 16, 4)

            # Excel cell K19 -> row 18, column 10 in 0-indexed DataFrame
            terminal_fcf_growth_rate = extract_numeric_value(df, 18, 10)

            # Excel cell E11 -> row 10, column 4
            valuation_date = extract_date_value(df, 10, 4)

            # Excel cell E14 -> row 13, column 4
            current_share_price = extract_numeric_value(df, 13, 4)

            # Excel cell E15 -> row 14, column 4
            diluted_shares_outstanding = extract_numeric_value(df, 14, 4)

            # Excel cell K24 -> row 23, column 10
            ev_multiples = extract_numeric_value(df, 23, 10)

            # Excel cell P24 -> row 23, column 15
            ev_perpetuity = extract_numeric_value(df, 23, 15)

            # Excel cell K39 -> row 38, column 10
            share_price_multiples = extract_numeric_value(df, 38, 10)

            # Excel cell P39 -> row 38, column 15
            share_price_perpetuity = extract_numeric_value(df, 38, 15)

        except Exception as e:
            # Alternative attempt - try searching for cells by header values if direct access fails
            st.warning(f"Attempting alternative cell extraction method due to: {str(e)}")

            # Search for cell values by looking for headers and relative positions
            wacc_row = locate_row_with_text(df, "Discount Rate (WACC)")
            terminal_growth_row = locate_row_with_text(df, "Implied Terminal FCF Growth Rate")
            valuation_date_row = locate_row_with_text(df, "Valuation Date")
            share_price_row = locate_row_with_text(df, "Current Share Price")
            shares_outstanding_row = locate_row_with_text(df, "Diluted Shares Outstanding")
            ev_row = locate_row_with_text(df, "Implied Enterprise Value")
            implied_share_row = locate_row_with_text(df, "Implied Share Price")

            # Extract values based on relative positions
            wacc = extract_numeric_from_row(df, wacc_row, 4) if wacc_row is not None else 0.1  # Default 10%
            terminal_fcf_growth_rate = extract_numeric_from_row(df, terminal_growth_row, 10) if terminal_growth_row is not None else 0.02  # Default 2%
            valuation_date = extract_date_from_row(df, valuation_date_row, 4) if valuation_date_row is not None else datetime.now().strftime("%Y-%m-%d")
            current_share_price = extract_numeric_from_row(df, share_price_row, 4) if share_price_row is not None else 0
            diluted_shares_outstanding = extract_numeric_from_row(df, shares_outstanding_row, 4) if shares_outstanding_row is not None else 0

            # Enterprise values and implied share prices could be in multiple rows, find the right one
            ev_multiples = extract_numeric_from_row(df, ev_row, 10) if ev_row is not None else 0
            ev_perpetuity = extract_numeric_from_row(df, ev_row, 15) if ev_row is not None else 0
            share_price_multiples = extract_numeric_from_row(df, implied_share_row, 10) if implied_share_row is not None else 0
            share_price_perpetuity = extract_numeric_from_row(df, implied_share_row, 15) if implied_share_row is not None else 0

        return {
            'wacc': wacc,
            'terminal_fcf_growth_rate': terminal_fcf_growth_rate,
            'valuation_date': valuation_date,
            'current_share_price': current_share_price,
            'diluted_shares_outstanding': diluted_shares_outstanding,
            'ev_multiples': ev_multiples,
            'ev_perpetuity': ev_perpetuity,
            'share_price_multiples': share_price_multiples,
            'share_price_perpetuity': share_price_perpetuity
        }

    except Exception as e:
        st.error(f"Error extracting DCF variables: {str(e)}")
        # Return reasonable defaults for sensitivity analysis
        return {
            'wacc': 0.1,  # 10%
            'terminal_fcf_growth_rate': 0.02,  # 2%
            'valuation_date': datetime.now().strftime("%Y-%m-%d"),
            'current_share_price': 5.0,
            'diluted_shares_outstanding': 1000,
            'ev_multiples': 5000,
            'ev_perpetuity': 5500,
            'share_price_multiples': 6.0,
            'share_price_perpetuity': 6.5
        }

def extract_numeric_value(df, row, col):
    """Extract a numeric value from a specific cell, handling different formats"""
    value = df.iloc[row, col]

    if pd.isna(value):
        return 0

    if isinstance(value, (int, float)):
        return value

    # Try to convert string to numeric
    try:
        # Remove any currency symbols, commas, and percentage signs
        if isinstance(value, str):
            value = value.replace('$', '').replace('£', '').replace('€', '')
            value = value.replace(',', '')

            if '%' in value:
                value = value.replace('%', '')
                return float(value) / 100  # Convert percentage to decimal

            return float(value)

        return 0
    except:
        return 0

def extract_date_value(df, row, col):
    """Extract a date value from a specific cell, handling different formats"""
    value = df.iloc[row, col]

    if pd.isna(value):
        return datetime.now().strftime("%Y-%m-%d")

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%d")

    # If it's a string, try to parse it
    try:
        if isinstance(value, str):
            return pd.to_datetime(value).strftime("%Y-%m-%d")

        return datetime.now().strftime("%Y-%m-%d")
    except:
        return datetime.now().strftime("%Y-%m-%d")

def locate_row_with_text(df, text):
    """Find row index containing the specified text"""
    for i in range(len(df)):
        row_values = df.iloc[i].astype(str).str.contains(text, case=False, na=False)
        if any(row_values):
            return i
    return None

def extract_numeric_from_row(df, row, col):
    """Extract numeric value from specified row and column"""
    if row is None:
        return 0

    try:
        return extract_numeric_value(df, row, col)
    except:
        return 0

def extract_date_from_row(df, row, col):
    """Extract date value from specified row and column"""
    if row is None:
        return datetime.now().strftime("%Y-%m-%d")

    try:
        return extract_date_value(df, row, col)
    except:
        return datetime.now().strftime("%Y-%m-%d")

def format_currency(value):
    """Format a numeric value as currency"""
    if pd.isna(value) or value == 0:
        return "£0.00"

    if value >= 1_000_000:
        return f"£{value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"£{value/1_000:.2f}K"
    else:
        return f"£{value:.2f}"

def format_percentage(value):
    """Format a numeric value as percentage"""
    if pd.isna(value):
        return "0.00%"

    return f"{value*100:.2f}%"
