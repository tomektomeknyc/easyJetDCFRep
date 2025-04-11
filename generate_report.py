import os
import io
import base64
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def generate_html_report(dcf_analyzer, returns_array):
    output_path = "attached_assets/EasyJet_DCF_Report.html"
    os.makedirs("attached_assets", exist_ok=True)

    metrics = dcf_analyzer.variables

    # Extract metrics
    wacc = metrics.get("wacc", 0)
    terminal_growth = metrics.get("terminal_growth", 0)
    current_price = metrics.get("current_share_price", 0)
    diluted_shares = metrics.get("diluted_shares_outstanding", 0)
    ev_multiples = metrics.get("ev_multiples", 0)
    ev_perpetuity = metrics.get("ev_perpetuity", 0)
    share_price_multiples = metrics.get("share_price_multiples", 0)
    share_price_perpetuity = metrics.get("share_price_perpetuity", 0)

    # Monte Carlo metrics
    mean_return = f"{returns_array.mean()*100:.2f}%" if returns_array is not None else "N/A"
    volatility = f"{returns_array.std()*100:.2f}%" if returns_array is not None else "N/A"

    # ---------- Graphs ----------
    def save_plot(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return f'<img src="data:image/png;base64,{encoded}" width="600"/>'

    # Enterprise Value Bar Chart
    fig1, ax1 = plt.subplots()
    ax1.bar(["EV (Multiples)", "EV (Perpetuity)"], [ev_multiples, ev_perpetuity], color=["#1E88E5", "#FF7043"])
    ax1.set_title("Enterprise Value Comparison")
    ax1.set_ylabel("EV (£m)")
    ev_chart = save_plot(fig1)

    # Share Price Bar Chart
    fig2, ax2 = plt.subplots()
    ax2.bar(["Share Price (Multiples)", "Share Price (Perpetuity)"], [share_price_multiples, share_price_perpetuity], color=["#43A047", "#FBC02D"])
    ax2.set_title("Implied Share Price Comparison")
    ax2.set_ylabel("Share Price (£)")
    share_chart = save_plot(fig2)

    # Monte Carlo Histogram
    fig3, ax3 = plt.subplots()
    if returns_array is not None:
        sims = np.random.normal(loc=returns_array.mean(), scale=returns_array.std(), size=1000)
        final_prices = current_price * np.exp(np.cumsum(sims))
        ax3.hist(final_prices, bins=40, color="#8E24AA")
        ax3.set_title("Monte Carlo Final Price Distribution")
        ax3.set_xlabel("Simulated Price (£)")
        ax3.set_ylabel("Frequency")
        mc_chart = save_plot(fig3)
    else:
        mc_chart = "<p>Monte Carlo data unavailable.</p>"

    # ---------- HTML ----------
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset='UTF-8'>
        <title>EasyJet DCF Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 30px; line-height: 1.6; }}
            h1 {{ color: #E67E22; }}
            h2, h3 {{ color: #2980B9; }}
            .section {{ margin-bottom: 40px; }}
            .metric {{ margin-bottom: 10px; }}
        </style>
    </head>
    <body>
        <h1>EasyJet DCF Valuation Report</h1>
        <p><b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

        <div class='section'>
            <h2>1. DCF Key Inputs</h2>
            <p class='metric'><b>WACC:</b> {wacc*100:.2f}%</p>
            <p class='metric'><b>Terminal Growth:</b> {terminal_growth*100:.2f}%</p>
            <p class='metric'><b>Current Share Price:</b> £{current_price:.2f}</p>
            <p class='metric'><b>Diluted Shares Outstanding:</b> {diluted_shares:,.0f}</p>
        </div>

        <div class='section'>
            <h2>2. Valuation Results</h2>
            <p class='metric'><b>EV (Multiples):</b> £{ev_multiples:.2f}</p>
            <p class='metric'><b>EV (Perpetuity):</b> £{ev_perpetuity:.2f}</p>
            <p class='metric'><b>Implied Share Price (Multiples):</b> £{share_price_multiples:.2f}</p>
            <p class='metric'><b>Implied Share Price (Perpetuity):</b> £{share_price_perpetuity:.2f}</p>
        </div>

        <div class='section'>
            <h2>3. Enterprise Value Comparison</h2>
            {ev_chart}
        </div>

        <div class='section'>
            <h2>4. Share Price Breakdown</h2>
            {share_chart}
        </div>

        <div class='section'>
            <h2>5. Monte Carlo Simulation Summary</h2>
            <p><b>Mean Daily Return:</b> {mean_return}</p>
            <p><b>Volatility:</b> {volatility}</p>
            {mc_chart}
        </div>

        <p><i>This report was automatically generated from the Streamlit DCF Dashboard for EasyJet plc.</i></p>
    </body>
    </html>
    """

    with open(output_path, "w") as f:
        f.write(html)

    return output_path
