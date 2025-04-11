import streamlit as st
st.set_page_config(
    page_title="EasyJet DCF Model Analysis",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils import load_excel_file
from dcf_analyzer import DCFAnalyzer
from advanced_visualizations import AdvancedVisualizations
from monte_carlo import run_monte_carlo
from generate_report import generate_html_report


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

# ------------------ MAIN APP ------------------
def main():
    EXCEL_PATH = "attached_assets/EasyJet- complete.xlsx"
    dcf_analyzer = None
    adv_viz = None

    if os.path.exists(EXCEL_PATH):
        try:
            df_dict, _ = load_excel_file(EXCEL_PATH)
            if 'DCF' not in df_dict:
                st.error("The Excel file does not contain a 'DCF' tab.")
                return
            dcf_analyzer = DCFAnalyzer(df_dict['DCF'])
            adv_viz = AdvancedVisualizations(dcf_analyzer)
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
                dcf_analyzer = DCFAnalyzer(df_dict['DCF'])
                adv_viz = AdvancedVisualizations(dcf_analyzer)
            except Exception as e:
                st.error(f"Error processing the uploaded file: {e}")
                return
        else:
            st.warning("Please upload a valid Excel file.")
            return

    # Header
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://logos-download.com/wp-content/uploads/2016/03/EasyJet_logo_logotype_emblem.png", width=100)
    with col2:
        st.title("EasyJet Financial DCF Analysis Dashboard")
        st.subheader("Interactive analysis of EasyJet's Discounted Cash Flow model")

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Tabs
    main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
        "\U0001F4CA Interactive DCF Dashboard",
        "\U0001F4DD Documentation",
        "\U0001F3B2 Monte Carlo",
        "\U0001F4C4 Report"
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
            dcf_analyzer.display_enterprise_value_chart()
            st.markdown('</div>', unsafe_allow_html=True)
        st.write("---")
        with st.container():
            st.markdown('<div class="valuation-results">', unsafe_allow_html=True)
            dcf_analyzer.display_share_price_chart()
            st.markdown('</div>', unsafe_allow_html=True)
        st.write("---")
        if adv_viz:
            adv_viz.display_visual_dashboard()
        st.subheader("Additional Advanced Visualizations")
        adv_tab1, adv_tab2, adv_tab3, adv_tab4, adv_tab5 = st.tabs([
            "3D EV Sensitivity",
            "Share Price Sunburst",
            "WACC Analysis",
            "Two-Factor Heatmap",
            "Peer Analysis"
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
        - Use the **Interactive DCF Dashboard** to see charts and KPIs for enterprise value, share price, and assumptions.
        - Try out different **WACC** and **growth rate** sensitivities.
        - Run a **Monte Carlo simulation** to test thousands of random scenarios using historical returns.
        - Go to the **Report** tab to generate a ready-to-download PDF summary.

        **Best used on desktop for full visibility.**
        """)

         with st.expander("📐 Methodology & Calculations"):
           st.markdown("""
        The valuation is built using a structured Excel financial model which is parsed and visualized in real-time.

        **Key components:**
        - **Historical financial data** and analyst assumptions.
        - **Forecasted Free Cash Flows (FCFs)** for a 5–10 year period.
        - **Terminal Value Estimation**:
            - **Perpetuity Growth Method** (using a long-term FCF growth rate)
            - **Exit Multiple Method** (applying a terminal EV/EBITDA multiple)
        - **Discounting via WACC**, which blends the cost of equity and debt.
        - **Implied Equity Value** and **Share Price** are derived by subtracting debt, adding cash, and dividing by diluted shares.

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

    # Tab 4: Report
    with main_tab4:
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

# Final CSS
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
