import streamlit as st

st.set_page_config(
    page_title="EasyJet DCF Model Analysis",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from utils_base64 import get_base64_image


# Load the EasyJet banner image (used only in the header)
base64_plane = get_base64_image("attached_assets/easyJet.png")

st.markdown(f"""
<link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css" rel="stylesheet">
<div style="position: relative; height: 320px; background-image: url('data:image/png;base64,{base64_plane}'); background-size: cover; background-position: center; border-radius: 10px; margin-bottom: 30px;">
    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -60%); text-align: center;">
        <h1 style="color: white; font-size: 38px; margin: 0;">
            EasyJet Financial DCF Analysis<br/>
            <span style="display: block;">Dashboard</span>
        </h1>
        <h3 style="color: white; font-weight: normal; margin-top: 10px;"><br/>
            Interactive analysis of EasyJet's Discounted Cash Flow model
        </h3>
    </div>
</div>
""", unsafe_allow_html=True)
import refinitiv.dataplatform.eikon as ek
ek.set_app_key(st.secrets["refinitiv"]["app_key"])



import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import feedparser
from pathlib import Path
from utils import load_excel_file
from dcf_analyzer import DCFAnalyzer
from advanced_visualizations import AdvancedVisualizations
from monte_carlo import run_monte_carlo
from generate_report import generate_html_report
from fetch_peer_returns import fetch_analyst_target_refinitiv
from dcf_sensitivity import run_dcf_sensitivity
from easy_jet_news_scraper import EasyJetNewsScraper
from easy_jet_news_scraper import fetch_easyjet_news
import requests
from datetime import datetime



def parse_date(d):
    s = d.replace(" +0000", "")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        dt = datetime.strptime(s, "%a, %d %b %Y %H:%M:%S")
    return dt



scraper = EasyJetNewsScraper()



# ------------------ THEME TOGGLE ------------------

theme = st.sidebar.selectbox("Select Theme", ["Dark Mode", "Light Mode"], index=0)

if theme == "Dark Mode":
    app_bg = "#000"
    plot_bg = "#000"
    text_color = "#fff"
    metric_text = "#fff"
else:
    app_bg = "#001f3f"
    plot_bg = "#001f3f"
    text_color = "#fff"
    metric_text = "#fff"

# ------------------ DYNAMIC CSS ------------------
st.markdown(f"""
<style>
.stApp {{ background-color: {app_bg} !important; color: {text_color} !important; }}
h1, h2, h3, h4, p, span {{ font-family: 'Inter', sans-serif; color: {text_color} !important; }}
[data-testid="metric-container"] {{ background-color: transparent !important; color: {metric_text} !important; }}
[data-testid="metric-container"] * {{ color: {metric_text} !important; fill: {metric_text} !important; }}
div[data-testid="stylable_container"]#current_price_container,
div[data-testid="stylable_container"]#multiples_price_container,
div[data-testid="stylable_container"]#perpetuity_price_container,
div[data-testid="stylable_container"]#wacc_growth_container {{ background-color: #333 !important; border-radius: 10px !important; padding: 15px !important; }}
div[data-testid="stylable_container"]#current_price_container *,
div[data-testid="stylable_container"]#multiples_price_container *,
div[data-testid="stylable_container"]#perpetuity_price_container *,
div[data-testid="stylable_container"]#wacc_growth_container * {{ color: #00BFFF !important; fill: #00BFFF !important; }}
.dcf-key-variables [data-testid="stMetricValue"],
.valuation-results [data-testid="stMetricValue"],
[data-testid="stMetricValue"] {{ color: #00BFFF !important; fill: #00BFFF !important; }}
.stTabs [data-baseweb="tab"] {{ background-color: #FFA500 !important; color: #000 !important; border-radius: 4px 4px 0px 0px; padding: 10px; margin-right: 2px; }}
.stTabs [aria-selected="true"] {{ background-color: #FF6600 !important; color: #000 !important; border-bottom: 2px solid #FF6600; }}
.stTabs [data-baseweb="tab"] > div {{ color: #000 !important; }}
div.stButton > button:first-child {{ background-color: #1E88E5; color: white; border-radius: 5px; border: none; padding: 10px 25px; font-size: 16px; }}
div.stButton > button:hover {{ background-color: #1565C0; color: white; }}
[data-testid="stSidebar"] > div:first-child {{ background-color: #001f3f !important; color: #fff !important; }}
[data-testid="stSidebar"] * {{ color: #fff !important; fill: #fff !important; }}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
      /* Hint box next to the sidebar toggle button */
      .unfold-hint {
        position: fixed;
        top: 60px;    /* move it higher */
        left: 8px;    /* same horizontal position as the chevron */
        background: rgba(0, 191, 255, 0.9);
        color: white;
        padding: 2px 6px;     /* smaller padding */
        border-radius: 4px;
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;   /* slightly smaller text */
        z-index: 1000;
        pointer-events: none;
      }
    </style>
    <div class="unfold-hint">
      <i class="bi bi-sliders"></i> Unfold
    </div>
    """,
    unsafe_allow_html=True,
)
# ——— Show Refinitiv consensus target for EasyJet in the sidebar ———
target = fetch_analyst_target_refinitiv("EZJ.L")
if target is not None:
    st.sidebar.metric("Analyst Mean Target", f"GBp{target:.2f}")
else:
    st.sidebar.warning("No consensus price available")



@st.cache_data(ttl=24 * 60 * 60)
def get_all_news() -> pd.DataFrame:
    """Fetch easyJet news from Yahoo RSS, Google News, Finviz plus Refinitiv (if available)."""
    rows = []

    # 1) Yahoo Finance RSS
    feed = feedparser.parse(
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=EZJ.L&region=GB&lang=en-GB"
    )
    for entry in feed.entries:
        title = entry.get("title", "")
        if "EZJ.L" in title or "easyJet" in title:
            rows.append({
                "Date":     entry.get("published", ""),
                "Headline": title,
                "Link":     entry.get("link", ""),
                "Source":   "Yahoo RSS"
            })

    # 2) Google News RSS
    feed = feedparser.parse("https://news.google.com/rss/search?q=EZJ.L")
    for entry in feed.entries:
        rows.append({
            "Date":     entry.get("published", ""),
            "Headline": entry.get("title", ""),
            "Link":     entry.get("link", ""),
            "Source":   "Google News"
        })

    # 3) Finviz RSS
    feed = feedparser.parse("https://finviz.com/news.ashx?t=EZJ.L")
    for entry in feed.entries:
        rows.append({
            "Date":     entry.get("published", ""),
            "Headline": entry.get("title", ""),
            "Link":     entry.get("link", ""),
            "Source":   "Finviz"
        })

    # Build the DataFrame for these three
    df_yahoo = pd.DataFrame(rows)

    # 4) Refinitiv via fetch_peer_returns (if available)
    try:
        df_ref = fetch_easyjet_news(count=10).assign(Source="Refinitiv")
    except Exception:
        # silently fall back to empty if something goes wrong
        df_ref = pd.DataFrame(columns=df_yahoo.columns)

    # 5) Combine and return
    return pd.concat([df_yahoo, df_ref], ignore_index=True)


# ------------------ MAIN APP ------------------
def main():


    #ticker = st.sidebar.text_input("Ticker", value="EZJ.L")


    EXCEL_PATH = "attached_assets/EasyJet- complete.xlsx"

    # Load local Excel file or request upload
    if os.path.exists(EXCEL_PATH):
        try:
            df_dict, _ = load_excel_file(EXCEL_PATH)
            if 'DCF' not in df_dict:
                st.error("The Excel file does not contain a 'DCF' tab.")
                return
            dcf_analyzer = DCFAnalyzer(df_dict['DCF'])
            adv_viz = AdvancedVisualizations(dcf_analyzer)
          #  st.write("DCF variable keys:", list(dcf_analyzer.variables.keys()))

            # ── DEBUG: show all extracted DCF variables ──
            with st.expander("🛠️ Debug: show all DCF variables", expanded=False):
                 st.write(dcf_analyzer.variables)

        except Exception as e:
            st.error(f"Error processing local Excel file: {e}")
            return
    else:
        st.info("No local Excel file found. Please upload your EasyJet financial model Excel file.")
        uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
        if uploaded_file:
            try:
                df_dict, _ = load_excel_file(uploaded_file)
                if 'DCF' not in df_dict:
                    st.error("The uploaded file does not contain a 'DCF' tab.")
                    return
                dcf_analyzer  = DCFAnalyzer(df_dict['DCF'])
                adv_viz = AdvancedVisualizations(dcf_analyzer)
            except Exception as e:
                st.error(f"Error processing the uploaded file: {e}")
                return
        else:
            st.warning("Please upload a valid Excel file.")
            return

  # now compute the comps & M&A implied upside/downside %
            current_price = dcf_analyzer.variables.get("current_share_price", 0)  # or your existing current_price var
        if current_price:
           comp_diff_pct = (comp_price - current_price) / current_price * 100
           mna_diff_pct  = (mna_price  - current_price) / current_price * 100
        else:
           comp_diff_pct = 0.0
           mna_diff_pct  = 0.0



# —————— ensure comp_price and mna_price always exist ——————
    try:
        peer_df    = pd.read_csv("attached_assets/ev_ebitda_combined.csv")
        comp_price = dcf_analyzer.implied_by_comps(peer_df)
    except Exception:
        comp_price = 0.0

    # pick an M&A multiple from precedent transactions
    mna_multiple = st.sidebar.slider(
    "🤝 M&A EV/EBITDA multiple (EasyJet precedent median: 8.7×)",
    min_value=3.5,
    max_value=24.9,
    value=8.7,
    step=0.1
)




    # initialize these so they're always defined
    comp_diff_pct = 0.0
    mna_diff_pct  = 0.0

    try:
        mna_price = dcf_analyzer.implied_by_precedent_mna(mna_multiple)
    except Exception:
        mna_price = 0.0
            # compute comps & M&A implied upside/downside %
    current_price = dcf_analyzer.variables.get("current_share_price", 0)
    if current_price:
        comp_diff_pct = (comp_price - current_price) / current_price * 100
        mna_diff_pct  = (mna_price  - current_price) / current_price * 100
    else:
        comp_diff_pct = 0.0
        mna_diff_pct  = 0.0

# compute & stash peer multiple
    peer_multiple = peer_df["EV_EBITDA"].median()
    dcf_analyzer.variables["peer_multiple"] = peer_multiple


    # ========================
    # New Feature: Buy/Hold/Sell Recommendation
    # ========================
    # Extract required prices from DCF variables
    current_price = dcf_analyzer.variables.get("current_share_price", 0)
    implied_mult_price = dcf_analyzer.variables.get("share_price_multiples", 0)
    implied_perp_price = dcf_analyzer.variables.get("share_price_perpetuity", 0)
    pe_ratio = dcf_analyzer.variables.get("current_pe_ratio", 0)
    pb_ratio = dcf_analyzer.variables.get("current_pb_ratio", 0)
    total_debt = dcf_analyzer.variables["total_debt"]
    cash_and_investments = dcf_analyzer.variables["cash_and_investments"]

    net_debt = total_debt - cash_and_investments
    ev = dcf_analyzer.variables["ev_multiples"]
    net_debt_pct_of_ev = (net_debt / ev * 100) if ev else 0
    shares_outstanding = dcf_analyzer.variables["diluted_shares_outstanding"]


    def styled_label_with_icon(label, icon_class, value):
        return f"""
            <div style="font-size:20px;">
                <strong>
                    <i class="bi {icon_class}" style="margin-right: 8px;"></i>{label}
                </strong>
                <span style="
                    background-color: black;
                    color: #00ff88;
                    padding: 4px 10px;
                    border-radius: 5px;
                    font-family: monospace;">
                    {value}
                </span>
            </div>
        """
# === split those icon‐labels into two columns ===
    col1, col2 = st.columns(2)

    with col1:
       st.write(styled_label_with_icon("Model EV/EBITDA:", "bi-graph-up",dcf_analyzer.variables.get("current_model_ev_ebitda", 0)), unsafe_allow_html=True)
       st.write(styled_label_with_icon("Industry EV/EBITDA:", "bi-building",dcf_analyzer.variables.get("current_industry_ev_ebitda", 0)), unsafe_allow_html=True)
       st.write(styled_label_with_icon("EV Diff Percent:", "bi-percent",dcf_analyzer.variables.get("current_ev_diff_percent", 0)), unsafe_allow_html=True)
       st.write(styled_label_with_icon("P/E:", "bi-cash-coin",dcf_analyzer.variables.get("current_pe_ratio", 0)), unsafe_allow_html=True)

    with col2:
      st.write(styled_label_with_icon("P/B:", "bi-journal-richtext",dcf_analyzer.variables.get("current_pb_ratio", 0)), unsafe_allow_html=True)
      st.write(styled_label_with_icon("Peer multiple EV/EBITDA (median):", "bi-bar-chart-fill",dcf_analyzer.variables.get("peer_multiple", 0)), unsafe_allow_html=True)
      st.write(styled_label_with_icon("Total Debt/Cash&Investments:","bi-wallet2",f"£{total_debt:,.2f}/£{cash_and_investments:,.2f}"),unsafe_allow_html=True)

      st.write(styled_label_with_icon("Net Debt:","bi-bank",f"£{net_debt:,.0f} ({net_debt_pct_of_ev:.1f}% of EV)"), unsafe_allow_html=True)

      st.write(styled_label_with_icon("Shares Outstanding:", "bi-people-fill",f"{shares_outstanding:,.0f}"), unsafe_allow_html=True)


      ev_diff_percent=  dcf_analyzer.variables.get("current_ev_diff_percent", 0)
    # Calculate the average implied share price (you could adjust weighting if needed)
    if current_price > 0 and implied_mult_price > 0 and implied_perp_price > 0:
        avg_implied_price = (implied_mult_price + implied_perp_price) / 2
        # Compute percentage difference between average implied price and current price
        diff_pct = (avg_implied_price - current_price) / current_price * 100
    else:
        avg_implied_price = 0
        diff_pct = 0

        peer_df = pd.read_csv("attached_assets/ev_ebitda_combined.csv")
        comp_price = dcf_analyzer.implied_by_comps(peer_df)
        mna_price  = dcf_analyzer.implied_by_precedent_mna(mna_multiple=8.0)  # say 8×



# 1) Compute your comp & M&A % moves
        comp_diff_pct = (comp_price - current_price) / current_price * 100
        mna_diff_pct  = (mna_price  - current_price) / current_price * 100
# 2) Compute net-debt as % of EV
        total_debt= dcf_analyzer.variables["total_debt"]
        cash_and_investments = dcf_analyzer.variables["cash_and_investments"]
        net_debt = total_debt - cash_and_investments
        ev = dcf_analyzer.variables["ev_multiples"] or 1  # avoid divide by zero
        net_debt_pct_of_ev = (net_debt / ev) * 100


# 3) Buy when DCF, comps & M&A all look undervalued by ≥20%, and PE/PB are in range
    if (
    diff_pct >= 20
    and comp_diff_pct  >= 20
    and mna_diff_pct   >= 20
    and ev_diff_percent >= 5
    and 0 < pe_ratio < 25
    and 0 < pb_ratio < 2
    and net_debt_pct_of_ev < 20
):
       recommendation, rec_color = "Buy", "#2ecc71"

# 4) Sell if ANY of DCF, peers or comps/M&A signal ≥10% downside, or P/E/PB are warning‐signs
    elif (
    diff_pct       <= -10
    or ev_diff_percent <= -10
    or comp_diff_pct  <= -10
    or mna_diff_pct   <= -10
    or (pe_ratio < 0 and diff_pct <= 0)
    or pb_ratio    > 3
    or net_debt_pct_of_ev > 40
):
       recommendation, rec_color = "Sell", "#e74c3c"

# 5) Otherwise you’re in the “no clear signal” bucket
    else:
       recommendation, rec_color = "Hold", "#95a5a6"



    # Pull the EV/EBITDA numbers into locals
    easyjet_ev_ebitda   = dcf_analyzer.variables.get("current_model_ev_ebitda", 0.0)
    peers_avg_ev_ebitda = dcf_analyzer.variables.get("current_industry_ev_ebitda", 0.0)

    st.markdown(f"""
<div style="padding:20px; border-radius:8px; background-color:{rec_color}; box-shadow:0 4px 8px rgba(0,0,0,0.3);">
  <h2 style="text-align:center; color:#fff; margin-bottom:12px;">
    Recommendation: {recommendation}
  </h2>
  <p style="text-align:center; color:#fff; margin:4px 0;">
    📈 Current Price: £{current_price:.2f} | 🔢 Implied (Multiples): £{implied_mult_price:.2f} | 🧮 Implied (Perpetuity): £{implied_perp_price:.2f}
  </p>
  <p style="text-align:center; color:#fff; margin:4px 0;">
    🧮 Average Implied Price: £{avg_implied_price:.2f} | ⬆️ Difference: {diff_pct:.1f}%
  </p>
  <p style="text-align:center; color:#fff; margin:4px 0;">
    📊 EV/EBITDA (Model): {easyjet_ev_ebitda:.2f} | 🏢 EV/EBITDA (Industry Avg): {peers_avg_ev_ebitda:.2f} | 🔻 EV/EBITDA Δ: {ev_diff_percent:.1f}%
  </p>
  <p style="text-align:center; color:#fff; margin:4px 0;">
    💵 P/E: {pe_ratio:.2f} | 📚 P/B: {pb_ratio:.2f}
  </p>
  <p style="text-align:center; color:#fff; margin:4px 0;">
    🏦 Net Debt/EV: {net_debt_pct_of_ev:.1f}
  </p>
  <p style="text-align:center; color:#fff; margin:4px 0;">
  🔍 Comps implied: £{comp_price:.2f} | 🤝 M&A implied: £{mna_price:.2f}</p>

</div>
""", unsafe_allow_html=True)






    # ========================
    # Main Tabs
    # ========================


    st.markdown("""
<style>
/* Custom tab look (Streamlit default tabs override) */
.stTabs [role="tab"] {
    background-color: #f7931e;
    color: #ffffff;
    padding: 12px 20px;
    margin-right: 8px;
    border: none;
    border-radius: 10px 10px 0 0;
    font-weight: 600;
    font-size: 16px;
    transition: all 0.3s ease-in-out;
    box-shadow: 0px 3px 6px rgba(0,0,0,0.2);
}

.stTabs [role="tab"]:hover {
    background-color: #ffa94d;
    color: #fff;
    transform: translateY(-1px);
}

.stTabs [role="tab"][aria-selected="true"] {
    background-color: #ff5f00;
    color: #ffffff;
    border-bottom: 4px solid #00d084;
    box-shadow: inset 0 -4px 0 0 #00d084;
}
</style>
""", unsafe_allow_html=True)



# Tabs
    main_tab1, main_tab2, main_tab3, main_tab4, main_tab5, main_tab6 = st.tabs([
        "\U0001F4CA Interactive DCF Dashboard",
        "\U0001F4DD Documentation",
        "\U0001F3B2 Monte Carlo",
        "\U0001F4C8 NPV Simulation",
        "\U0001F4C4 Report",
        "\U0001F4F0 News"

    ])

    # Tab 1: DCF Dashboard
    with main_tab1:
        st.subheader("Main Financial Analysis")
        st.write("### Key Metrics")
        with st.container():
            st.markdown('<div class="dcf-key-variables">', unsafe_allow_html=True)
        dcf_analyzer.display_key_metrics()
        st.markdown('</div>', unsafe_allow_html=True)
        st.write("---")
        with st.container():
            st.markdown('<div class="valuation-results">', unsafe_allow_html=True)
        # Call get_enterprise_value_chart() to display the chart
        dcf_analyzer.get_enterprise_value_chart()
        st.markdown('</div>', unsafe_allow_html=True)
        st.write("---")
        with st.container():
            st.markdown('<div class="valuation-results">', unsafe_allow_html=True)
            dcf_analyzer.display_share_price_chart()


            st.markdown('</div>', unsafe_allow_html=True)
        st.write("---")
        if adv_viz is not None:
            adv_viz.display_visual_dashboard()
        st.subheader("Additional Advanced Visualizations")
        adv_tab1, adv_tab2, adv_tab3, adv_tab4, adv_tab5, adv_tab6, adv_tab7 = st.tabs([
            "3D EV Sensitivity",
            "Share Price Sunburst",
            "WACC Analysis",
            "Two-Factor Heatmap",
            "Peer Analysis",
            "EV/EBITDA Comparables",
            "Debt Structure"
        ])
        with adv_tab1:
            adv_viz._display_3d_sensitivity_with_real_data()
        with adv_tab2:
            adv_viz.display_share_price_sunburst()
        with adv_tab3:
            adv_viz.display_wacc_analysis_dashboard()
        with adv_tab4:
            adv_viz.display_two_factor_heatmap()
        with adv_tab5:
            adv_viz.display_peer_analysis()
        with adv_tab6:
            adv_viz.display_ev_ebitda_comparables()
        with adv_tab7:
            adv_viz.display_debt_structure()


    # Tab 2: Documentation
    with main_tab2:
        st.header("📘 Documentation and Help")
        with st.expander("📊 What is a DCF Analysis?", expanded=False):
            st.markdown("""
**Discounted Cash Flow (DCF)** is a fundamental valuation approach used to determine the present value of an asset based on its future expected cash flows.

In this dashboard, the DCF methodology is applied to EasyJet using:
- Projected **Free Cash Flows (FCFs)** from the financial model.
- A **discount rate** (Weighted Average Cost of Capital or WACC) to account for the time value of money and risk.
- A **Terminal Value**, capturing value beyond the forecast period using either:
  - A **perpetual growth model** (Gordon Growth)
  - Or a **multiples-based approach** (e.g. EV/EBITDA).

The output is an estimated **Enterprise Value (EV)** and **Implied Share Price**.
            """)
        with st.expander("🛠️ How to Use This Dashboard"):
            st.markdown("""
This interactive dashboard allows you to explore EasyJet's DCF valuation in depth.

**Navigation Tips:**
- Use the **DCF Dashboard** to see charts and KPIs for enterprise value, share price, and assumptions.
- Try out different **WACC** and **growth rate** sensitivities.
- Run a **Monte Carlo simulation** to test thousands of random scenarios using historical returns.
- Go to the **Report** tab to generate a ready-to-download HTML summary.

**Best used on desktop for full visibility.**
            """)
        with st.expander("📐 Methodology & Calculations"):
            st.markdown("""
The valuation is built using a structured Excel financial model which is parsed and visualized in real-time.

**Key components:**
- **Historical financial data** and analyst assumptions.
- **Forecasted Free Cash Flows (FCFs)** for a 5-10 year period.
- **Terminal Value Estimation**:
    - **Perpetuity Growth Method** (using a long-term FCF growth rate)
    - **Exit Multiple Method** (applying a terminal EV/EBITDA multiple)
- **Discounting via WACC**, which blends the cost of equity and debt.
- **Implied Equity Value** and **Share Price** are derived by subtracting debt, adding cash, and dividing by diluted shares.
- Each simulation run is a path of 365 trading days (1 year).
  For each day, a random return is generated using a normal distribution:
  𝑅𝑡 ∼ 𝑁(𝜇, 𝜎)

    where:

  μ = average daily return (based on your historical data)

  σ = daily volatility (standard deviation from your CSV)

These daily returns are compounded, producing a final simulated price at the end of 1 year.

This process is repeated N times (e.g., 1000 simulations) to generate a distribution of possible outcomes.
Outputs are shown visually with sensitivity to changes in assumptions.
            """)
        with st.expander("✈️ About EasyJet"):
            st.markdown("""
**EasyJet plc** is a leading low-cost airline headquartered in the UK, serving short-haul routes across Europe.

**Key Facts:**
- Founded: 1995
- Head Office: London Luton Airport
- Fleet Size: ~300 aircraft
- Destinations: Over 150 airports
- Business Model: Point-to-point low-fare flights

EasyJet is traded on the London Stock Exchange under the ticker **EZJ.L**.
            """)

    # Tab 3: Monte Carlo
    with main_tab3:
        st.header("Monte Carlo Simulation")
        try:
            returns_df = pd.read_csv("attached_assets/EZJ_L_returns.csv", index_col=0, parse_dates=True)
            st.markdown("### Ten years of historical returns data from Refinitiv API")
            st.dataframe(returns_df, height=300)
            returns_array = returns_df["Returns"].dropna().values
        except Exception as e:
            st.error(f"Error loading historical returns CSV: {e}")
            returns_array = None

        default_price = dcf_analyzer.variables.get("current_share_price", 1.0) if dcf_analyzer else 1.0

        if returns_array is not None:
            n_sims = st.slider("Number of Simulations", 100, 5000, 1000, 100)
            horizon = st.slider("Simulation Horizon (Days)", 30, 365, 252, 10)
            initial_price = st.number_input("Starting Price", value=float(default_price))

            if st.button("Run Monte Carlo Simulation"):
                final_prices = run_monte_carlo(returns_array, n_sims, horizon, initial_price)
                st.write(f"Mean Final Price: £{np.mean(final_prices):.2f}")
                st.write(f"Median Final Price: £{np.median(final_prices):.2f}")
                st.write(f"Max Final Price: £{max(final_prices):.2f}")
                st.write(f"Min Final Price: £{min(final_prices):.2f}")

                df_prices = pd.DataFrame({"Final Price": final_prices})
                fig = px.histogram(df_prices, x="Final Price", nbins=50, title="Distribution of Final Simulated Prices")
                fig.update_traces(marker_color="#00BFFF")
                fig.update_layout(
                    title_font_color="#00BFFF",
                    yaxis_title="Number of Simulations",
                    paper_bgcolor=plot_bg,
                    plot_bgcolor=plot_bg,
                    font_color=text_color
                )
                fig.update_xaxes(tickfont=dict(color=text_color))
                fig.update_yaxes(tickfont=dict(color=text_color))
                st.plotly_chart(fig, use_container_width=True)



        # Tab 4: NPV Simulation
    with main_tab4:
        st.header("NPV Sensitivity Simulation (5 years)")

        # place your sliders
        col1, col2 = st.columns(2)
        with col1:
            g = st.slider("Growth rate (g)", 0.0, 0.10, 0.02, step=0.002)
        with col2:
            r = st.slider("Discount rate (r)", 0.04, 0.15, 0.08, step=0.002)

        # now render a button at the tab level (two spaces in from `with main_tab4`)
        if st.button("Run NPV Simulation", key="npv_sim"):
            # this is indented four spaces from `with main_tab4`
            st.write("→ Trying to open:", EXCEL_PATH)
            fig = run_dcf_sensitivity(str(EXCEL_PATH))
            st.plotly_chart(fig, use_container_width=True)


    # Tab 5: Report
    with main_tab5:
        st.header("📄 Generate HTML Report")
        if st.button("Create Report"):
            if dcf_analyzer is not None and returns_array is not None:
                try:
                    generate_html_report(dcf_analyzer, returns_array)
                    st.success("✅ Report generated: EasyJet_DCF_Report.html")
                except Exception as e:
                    st.error(f"❌ Report generation failed: {e}")
            else:
                st.error("DCF Analyzer is not initialized.")

        html_path = "attached_assets/EasyJet_DCF_Report.html"
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                st.download_button("📥 Download HTML Report", f, file_name="EasyJet_DCF_Report.html", mime="text/html")

                st.markdown("""
<div style="background-color:#FFA500; padding:10px; border-radius:5px; margin-top:20px; text-align:center;">
  <p style="margin:0; font-size:14px; color:#000;">
    This interactive DCF analysis dashboard is for educational and analytical purposes only.
    It is not financial advice. Data is based on historical information and financial projections.
  </p>
</div>
        """, unsafe_allow_html=True)
    

        # Tab 6: News (updates once every 24 hours)
       
        # Tab 6: News (updates once every 24 hours)
    # with main_tab6:
    #     st.header("📰 Latest easyJet News (Yahoo + Google+ FinViz)")

    #     # 1) Load & cache
    #     all_news = get_all_news()

    #     # 2) Tidy up dates & sort newest first
    #     def parse_date(d: str) -> datetime:
    #         s = d.replace(" +0000", "").replace(" GMT", "")
    #         try:
    #             return datetime.fromisoformat(s)
    #         except ValueError:
    #             return datetime.strptime(s, "%a, %d %b %Y %H:%M:%S")

    #     all_news["_dt"] = all_news["Date"].apply(parse_date)
    #     all_news = all_news.sort_values("_dt", ascending=False).reset_index(drop=True)
    #     all_news["Date"] = all_news["_dt"].dt.strftime("%Y-%m-%d %H:%M")
    #     all_news = all_news.drop(columns=["_dt"])

    #     # 3) Combine headline + link into one “News” cell
    #     all_news["News"] = (
    #         all_news["Headline"]
    #         + "<br>"
    #         + all_news["Link"].apply(lambda u: f'<a href="{u}" target="_blank">{u}</a>')
    #     )

    #     # 4) Build display DF with exactly three columns
    #     df_display = all_news[["Date", "News", "Source"]]

    #     # 5) Render scrollable HTML table with CSS tweaks
    #     st.markdown(
    #         """
    #         <style>
    #           .news-table { width:100%; border-collapse: collapse; }
    #           .news-table th, .news-table td { padding: 8px; vertical-align: top; }
    #           .news-table th:nth-child(1), .news-table td:nth-child(1) { width: 20%; }
    #           .news-table th:nth-child(2), .news-table td:nth-child(2) { width: 60%; }
    #           .news-table th:nth-child(3), .news-table td:nth-child(3) { width: 20%; text-align:center; }
    #           .news-table th { background-color: #FFA500; color: #000; }
    #           .scrollable-news { height: 400px; overflow-y: auto; border: 1px solid #444; border-radius: 4px; }
    #         </style>
    #         """,
    #         unsafe_allow_html=True,
    #     )
    #     html = df_display.to_html(escape=False, index=False, classes="news-table")
    #     st.markdown(f'<div class="scrollable-news">{html}</div>', unsafe_allow_html=True)

    #     # 6) If empty, warn; otherwise show bullet-list of just the headlines
    #     if all_news.empty:
    #         st.warning("⚠️ No news items found.")
    #     else:
    #         st.markdown("### Headlines")
    #         for _, row in all_news.iterrows():
    #             # extract pure headline text (before the <br>)
    #             headline_text = row["Headline"]
    #             if "<br>" in headline_text:
    #                 headline_text = headline_text.split("<br>")[0]
    #             st.markdown(f"- **{row['Date']}**: [{headline_text}]({row['Link']})")

        # Tab 6: News (updates once every 24 hours)
    with main_tab6:
        st.header("📰 Latest easyJet News (Yahoo + Google + Finviz)")

        # 1) Load & cache
        all_news = get_all_news()

        # 2) Tidy up dates, sort most recent first
        def parse_date(d: str) -> datetime:
            s = d.replace(" +0000", "").replace(" GMT", "")
            try:
                return datetime.fromisoformat(s)
            except ValueError:
                return datetime.strptime(s, "%a, %d %b %Y %H:%M:%S")

        all_news["_dt"] = all_news["Date"].apply(parse_date)
        all_news = all_news.sort_values("_dt", ascending=False).reset_index(drop=True)
        all_news["Date"] = all_news["_dt"].dt.strftime("%Y-%m-%d %H:%M")
        all_news = all_news.drop(columns=["_dt"])

        # 3) Merge headline + link into one “News” cell
        all_news["News"] = (
            all_news["Headline"]
            + "<br>"
            + all_news["Link"].apply(
                lambda u: f'<a href="{u}" target="_blank">{u}</a>'
            )
        )

        # 4) Build display DF with exactly three columns
        df_display = all_news[["Date", "News", "Source"]]

        # 5) Render scrollable HTML table with CSS tweaks
        st.markdown(
            """
            <style>
  .news-table { width:100%; border-collapse: collapse; }
  .news-table th, .news-table td { padding:8px; vertical-align: top; border-bottom:1px solid #444; }
  .news-table td { white-space: normal; word-break: break-word; }

  /* center-align columns */
  .news-table th:nth-child(1), .news-table td:nth-child(1) {
    width:20%;
    text-align: center;
  }
  .news-table th:nth-child(2), .news-table td:nth-child(2) {
    width:60%;
    text-align: center;
  }
  .news-table th:nth-child(3), .news-table td:nth-child(3) {
    width:20%;
    text-align: center;
  }

  /* scroll container */
  .scrollable-news {
    max-height: 400px;
    overflow-y: auto;
    overflow-x: auto;
    border:1px solid #444;
    border-radius:4px;
    padding:8px;
  }

  /* header style */
  .news-table th { background-color: #FFA500; color: #000; }
</style>

            """,
            unsafe_allow_html=True,
        )
        html_table = df_display.to_html(
            escape=False,
            index=False,
            classes="news-table"
        )
        st.markdown(f'<div class="scrollable-news">{html_table}</div>', unsafe_allow_html=True)

        # 6) Headlines list (optional)
        if all_news.empty:
            st.warning("⚠️ No news items found.")
        else:
            st.markdown("### Headlines")
            for _, row in all_news.iterrows():
                # split out the text part before the <br>
                title = row["News"].split("<br>")[0]
                st.markdown(f"- **{row['Date']}**: [{title}]({row['Link']})")


# Final CSS override for certain containers
        st.markdown("""
<style>
[data-testid="stylable_container"]#current_price_container,
[data-testid="stylable_container"]#multiples_price_container,
[data-testid="stylable_container"]#perpetuity_price_container,
[data-testid="stylable_container"]#wacc_growth_container {
    background-color: #444 !important;
}
</style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
