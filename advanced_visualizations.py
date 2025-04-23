import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
import math
import random
from dcf_analyzer import slice_section, parse_pct



# Optional extras – if you have these packages installed, they add extra styling.
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.chart_container import chart_container
from dcf_analyzer import slice_section, parse_pct


class AdvancedVisualizations:
    """
    A class providing advanced financial visualizations using data extracted from the Excel DCF model.
    """
    def __init__(self, dcf_analyzer):
        self.dcf = dcf_analyzer
        self.variables = dcf_analyzer.variables
        self.color_palette = {
            'primary': '#2196F3',
            'secondary': '#FF9800',
            'tertiary': '#4CAF50',
            'quaternary': '#9C27B0',
            'negative': '#F44336',
            'positive': '#4CAF50',
            'neutral': '#9E9E9E',
            'background': '#F5F5F5',
            'grid': 'rgba(0,0,0,0.05)'
        }
        self.gradient_palette = self._generate_gradient_palette()

    def _generate_gradient_palette(self, num_colors=20):
        colors = []
        for i in range(num_colors):
            r = int(33 + (242 - 33) * i / (num_colors - 1))
            g = int(150 + (153 - 150) * i / (num_colors - 1))
            b = int(243 + (0 - 243) * i / (num_colors - 1))
            colors.append(f'rgb({r},{g},{b})')
        return colors

    def format_currency(self, value):
        if isinstance(value, str):
            return value
        if math.isnan(value):
            return "N/A"
        if abs(value) >= 1e6:
            return f"£{value/1_000_000:.2f}M"
        elif abs(value) >= 1e3:
            return f"£{value/1_000:.2f}K"
        else:
            return f"£{value:.2f}"

    def format_percentage(self, value):
        if isinstance(value, str):
            return value
        if math.isnan(value):
            return "N/A"
        return f"{value*100:.2f}%"

    def display_header_dashboard(self):
        st.markdown("""
        <style>
        .metric-header {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        </style>
        <div class="metric-header">EasyJet Financial Summary</div>
        """, unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        # 1) Current Price
        with col1:
            current_price = self.variables.get('current_share_price', 0)
            with stylable_container(
                key="current_price_container",
                css_styles="{ border-radius: 10px; padding: 15px; }"
            ):
                st.metric(label="Current Share Price", value=f"£{current_price:.2f}")
        # 2) DCF Multiples Price
        with col2:
            multiples_price = self.variables.get('share_price_multiples', 0)
            base_price = current_price if current_price != 0 else 1
            pct_diff_multiples = ((multiples_price / base_price) - 1) * 100
            with stylable_container(
                key="multiples_price_container",
                css_styles="{ border-radius: 10px; padding: 15px; }"
            ):
                st.metric(
                    label="DCF Multiples Price",
                    value=f"£{multiples_price:.2f}",
                    delta=f"{pct_diff_multiples:.1f}%",
                    delta_color="normal"
                )
        # 3) DCF Perpetuity Price
        with col3:
            perpetuity_price = self.variables.get('share_price_perpetuity', 0)
            pct_diff_perpetuity = ((perpetuity_price / base_price) - 1) * 100
            with stylable_container(
                key="perpetuity_price_container",
                css_styles="{ border-radius: 10px; padding: 15px; }"
            ):
                st.metric(
                    label="DCF Perpetuity Price",
                    value=f"£{perpetuity_price:.2f}",
                    delta=f"{pct_diff_perpetuity:.1f}%",
                    delta_color="normal"
                )
        # 4) WACC / Terminal Growth (Updated label)
        with col4:
            wacc = self.variables.get('wacc', 0) * 100
            growth = self.variables.get('terminal_growth', 0) * 100
            with stylable_container(
                key="wacc_growth_container",
                css_styles="{ border-radius: 10px; padding: 15px; }"
            ):
                st.metric(label="WACC / Terminal Growth", value=f"{wacc:.2f}% / {growth:.2f}%")

    def _display_3d_sensitivity_with_real_data(self):
        base_wacc = self.variables.get("wacc", 0.10)
        base_growth = self.variables.get("terminal_growth", 0.02)
        base_ev = self.variables.get("ev_perpetuity", 5000)
        st.markdown("#### 3D EV Sensitivity (Using Excel-derived values)")
        wacc_range = np.linspace(base_wacc * 0.5, base_wacc * 1.5, 30)
        growth_range = np.linspace(base_growth * 0.5, base_growth * 1.5, 30)
        wacc_grid, growth_grid = np.meshgrid(wacc_range, growth_range)
        ev_surface = np.zeros_like(wacc_grid)
        for i in range(wacc_grid.shape[0]):
            for j in range(wacc_grid.shape[1]):
                w = wacc_grid[i, j]
                g = growth_grid[i, j]
                ratio = (base_wacc / w) ** 1.2 * ((1 + g) / (1 + base_growth))
                ev_surface[i, j] = base_ev * ratio
        fig = go.Figure(data=[go.Surface(
            x=wacc_grid,
            y=growth_grid,
            z=ev_surface,
            colorscale='Viridis',
            hovertemplate=(
                "WACC: %{x:.2%}<br>" +
                "Terminal Growth: %{y:.2%}<br>" +
                "Enterprise Value: £%{z:.2f}M<extra></extra>"
            )
        )])
        fig.update_layout(
            scene=dict(
                xaxis=dict(title="WACC"),
                yaxis=dict(title="Terminal Growth"),
                zaxis=dict(title="Enterprise Value (M)")
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=500,
            paper_bgcolor="#000",
            plot_bgcolor="#000"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("This chart uses real WACC, Growth, and EV values from Excel to display a sensitivity surface.")

    def display_share_price_sunburst(self):
        st.subheader("Share Price Sunburst Chart")
        enterprise_value = max(self.variables.get('ev_perpetuity', 0), self.variables.get('ev_multiples', 0))
        net_debt = self.variables.get('net_debt', enterprise_value * 0.3)
        equity_value = enterprise_value - net_debt
        sunburst_data = {
            'labels': ['Total Enterprise Value', 'Net Debt', 'Equity Value', 'Historical FCF', 'Terminal Value'],
            'parents': ['', 'Total Enterprise Value', 'Total Enterprise Value', 'Equity Value', 'Equity Value'],
            'values': [enterprise_value, net_debt, equity_value, equity_value * 0.35, equity_value * 0.65]
        }
        fig = go.Figure(go.Sunburst(
            labels=sunburst_data['labels'],
            parents=sunburst_data['parents'],
            values=sunburst_data['values'],
            branchvalues='total',
            texttemplate='<b>%{label}</b><br>£%{value:.1f}M<br>%{percentEntry:.1%}',
            hovertemplate='<b>%{label}</b><br>Value: £%{value:.2f}M<br>%{percentEntry:.2%}<extra></extra>',
            marker=dict(colors=['#1E88E5', '#F44336', '#4CAF50', '#9C27B0', '#FF9800'],
                        line=dict(color='black', width=1)),
            rotation=90
        ))
        fig.update_traces(textfont=dict(color="black"))
        fig.update_layout(
            height=600,
            margin=dict(t=10, l=10, r=10, b=10),
            paper_bgcolor="#000",
            plot_bgcolor="#000"
        )
        st.plotly_chart(fig, use_container_width=True)

    def display_wacc_analysis_dashboard(self):
        st.subheader("WACC Analysis Dashboard")
        # 1️⃣ North‑Star: the WACC loaded from your Excel model
        original_wacc = self.variables.get("wacc", 0.10)
        st.write(f"**Model‑calculated WACC:** {original_wacc*100:.2f}%")
        # ─── WACC INPUT SLIDERS ──────────────────────────
        equity_weight = st.slider("Equity Weight (%)", 0, 100, 70) / 100
        cost_equity   = st.slider("Cost of Equity (%)", 0.0, 20.0, 8.11) / 100
        cost_debt     = st.slider("Cost of Debt (%)",   0.0, 20.0, 3.48) / 100
        tax_rate      = st.slider("Corporate Tax Rate (%)", 0.0, 50.0, 21.0) / 100
        debt_weight   = 1.0 - equity_weight
        # 3️⃣ Compute WACC from the sliders (what‑if scenario)
        simulated_wacc = equity_weight * cost_equity \
        + debt_weight   * cost_debt * (1 - tax_rate)
                # ─── Waterfall: compare Original vs. Simulated WACC ───────────────────
        fig = go.Figure()
            # ─── Colors ─────────────────────────────────────────
        original_color  = "#636EFA"   # blue
        simulated_color = "#EF553B"   # red

         # ─── Prepare Original vs Simulated values ──────────
        orig_equity = original_wacc * equity_weight
        orig_debt   = original_wacc * debt_weight
        orig_total  = original_wacc

        sim_equity = equity_weight * cost_equity
        sim_debt   = debt_weight * cost_debt * (1 - tax_rate)
        sim_total  = simulated_wacc

    # ─── Add two bar traces ────────────────────────────
        fig.add_trace(go.Bar(
        x=["Equity", "After‑Tax Debt", "Total WACC"],
        y=[orig_equity, orig_debt, orig_total],
        name="Original",
        marker_color=original_color,
        text=[f"{v*100:.2f}%" for v in (orig_equity, orig_debt, orig_total)],
        textposition="outside"
    ))
        fig.add_trace(go.Bar(
        x=["Equity", "After‑Tax Debt", "Total WACC"],
        y=[sim_equity, sim_debt, sim_total],
        name="Simulated",
        marker_color=simulated_color,
        text=[f"{v*100:.2f}%" for v in (sim_equity, sim_debt, sim_total)],
        textposition="outside"
    ))

    # ─── Final layout tweaks ───────────────────────────
        fig.update_layout(
        barmode="group",
        title="WACC Build‑up: Original vs. Simulated",
        template="plotly_dark",
        height=500,
        legend=dict(title="", orientation="h", y=-0.2),
        yaxis=dict(tickformat=".0%")
    )


        st.plotly_chart(fig, use_container_width=True)
        st.info("Toggle between the two traces in the legend to compare build‑ups.")
        # # ───────────────────────────────────────────────────────────────────────


    def display_two_factor_heatmap(self):
        st.subheader("Two-Factor Sensitivity Heatmap")
        base_wacc = self.variables.get('wacc', 0.10)
        base_growth = self.variables.get('terminal_growth', 0.02)
        base_price = self.variables.get('share_price_perpetuity', 0)
        wacc_range = np.linspace(base_wacc * 0.9, base_wacc * 1.1, 20)
        growth_range = np.linspace(base_growth * 0.9, base_growth * 1.1, 20)
        wacc_grid, growth_grid = np.meshgrid(wacc_range, growth_range)
        price_grid = base_price * (base_wacc / wacc_grid) * ((1 + growth_grid) / (1 + base_growth))
        pct_change = ((price_grid / base_price) - 1) * 100
        fig = go.Figure(data=go.Heatmap(
            z=pct_change,
            x=np.round(wacc_range*100,2),
            y=np.round(growth_range*100,2),
            colorscale='Viridis',
            colorbar=dict(
                title=dict(text="% Change"),
                tickfont=dict()
            )
        ))
        fig.update_layout(
            title="Sensitivity of Share Price to WACC and Terminal Growth",
            xaxis_title="WACC (%)",
            yaxis_title="Terminal Growth (%)",
            height=500,
            margin=dict(l=50, r=50, t=80, b=50),
            paper_bgcolor="#000",
            plot_bgcolor="#000"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("This heatmap shows the impact on share price when varying WACC and Terminal Growth.")

    # === New Peer Analysis Function ===
    def display_peer_analysis(self):
        """
        Displays weekly returns for EasyJet and its peers over the last X years
        (controlled by a slider). Each ticker is loaded from a CSV in attached_assets,
        and all data is combined in a single line chart.
        """
        st.subheader("Peer Analysis")
        st.write("This section displays weekly returns for EasyJet and its peers over the selected time period.")
        # 1) Let the user pick how many years back to display (1-20, default 10)
        n_years = st.slider("Select number of years to display", min_value=1, max_value=20, value=10)
        # 2) Determine date range
        end_date = pd.to_datetime("today")
        start_date = end_date - pd.DateOffset(years=n_years)
        st.write(f"Displaying data from {start_date.date()} to {end_date.date()}")
        # 3) List of tickers (including EasyJet and its peers)
        tickers = [
            "EZJ.L",   # EasyJet
            "RYA.I",   # Ryanair
            "WIZZ.L",  # Wizz Air
            "LHAG.DE", # Lufthansa
            "ICAG.L",  # IAG
            "AIRF.PA", # Air France-KLM
            "JET2.L",  # Jet2
            "KNIN.S"   # Unknown/Others
        ]
        # 4) Prepare an array of custom colors so each ticker has a distinct color
        color_sequence = [

    "#1E88E5",  # Strong Blue
    "#FF7043",  # Distinctive Orange
    "#9C27B0",  # Bold Purple
    "#FBC02D",  # Brighter Yellow-Gold
    "#E91E63",  # Hot Pink / Fuchsia
    "#964B00",  # Rich Brown
    "#00E676",  # Neon Green
    "#00B8D4"  # Bright Turquoise

]


        df_list = []
        # 5) For each ticker, load the CSV, compute weekly returns, add to df_list
        for i, ticker in enumerate(tickers):
            file_name = f"attached_assets/{ticker.replace('.', '_')}_returns.csv"
            try:
                df = pd.read_csv(file_name, index_col=0, parse_dates=True)
                if "CLOSE" not in df.columns:
                    st.error(f"File {file_name} missing 'CLOSE' column. Skipping.")
                    continue
                df = df.loc[df.index >= start_date]
                weekly_close = df["CLOSE"].resample("W").last()
                weekly_returns = weekly_close.pct_change().dropna()
                weekly_df = weekly_returns.to_frame(name="Returns").reset_index()
                weekly_df.rename(columns={"index": "Date"}, inplace=True)
                weekly_df["Ticker"] = ticker
                df_list.append(weekly_df)
            except Exception as e:
                st.error(f"Error reading file {file_name}: {e}")
        if df_list:
            combined_df = pd.concat(df_list)
            fig_peer = px.line(
                combined_df,
                x="Date",
                y="Returns",
                color="Ticker",
                title=f"Weekly Returns for the Last {n_years} Year(s)",
                color_discrete_sequence=color_sequence
            )
            fig_peer.update_layout(
                paper_bgcolor="#000",
                plot_bgcolor="#000",
                font=dict(color="#fff", size=14, family="Arial, sans-serif"),
                title_font_color="#fff",
                legend_title_text="Peers",
                legend=dict(
    font=dict(color='white')  # Makes legend (ticker labels) white
)

            )
            fig_peer.update_xaxes(
                tickfont=dict(color="#fff"),
                title="Date",
                title_font=dict(color="#fff")
            )
            fig_peer.update_yaxes(
                tickfont=dict(color="#fff"),
                title="Weekly Returns (%)",
                title_font=dict(color="#fff")
            )
            st.plotly_chart(fig_peer, use_container_width=True)
        else:
            st.write("No return data available for the selected tickers.")


      # === Add this function to advanced_visualizations.py ===
    def display_ev_ebitda_comparables(self):
        """
        Display EV/EBITDA multiples of EasyJet and peer companies from a combined CSV.
        """
        st.subheader("EV/EBITDA Multiples - Comparables")
        st.write("This chart compares the EV/EBITDA valuation multiples for EasyJet and peer airlines.")

        filepath = "attached_assets/ev_ebitda_combined.csv"
        if not os.path.exists(filepath):
            st.warning("Combined EV/EBITDA CSV not found.")
            return

        df = pd.read_csv(filepath)
        if not {"Ticker", "EV_EBITDA"}.issubset(df.columns):
            st.warning("Combined CSV is missing required columns.")
            return

        fig = px.bar(
            df,
            x="Ticker",
            y="EV_EBITDA",
            color="Ticker",
            text="EV_EBITDA",
            template="plotly_dark",              # <-- use the dark template
            color_discrete_sequence=[            # <-- optional: your own palette
                "#66CCFF", "#0066CC", "#FF9999",
                "#FF3333", "#99FF99", "#00CC99", "#FFCC66"
            ],
        )

        # make the text labels nice and the bars fat(ter)
        fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside",
            width=0.6                       # <-- controls bar thickness
        )

        # close up the gaps between bars
        fig.update_layout(
            bargap=0.15,                    # <-- smaller = fatter bars
            margin=dict(t=80, b=40, l=40, r=40),
            yaxis=dict(automargin=True),
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)


    def display_debt_structure(self):

        # 1) Header
        st.subheader("Debt Structure as of " +
            pd.to_datetime(self.variables.get("valuation_date", None) or pd.Timestamp.today()).strftime("%d-%b-%Y")
        )

        # 2) Load the Excel summary table
        df = pd.read_excel(
            "attached_assets/debt_structure.xlsx",
            sheet_name="Summary",
            header=11,                           # row 12 is your header
            usecols="A:D",                       # A=Name, B=Issues, C=Outstanding, D=Issued
            names=["Name", "Issues", "Outstanding", "Issued"]
        )

        # coerce any non-numeric cells to 0 so the chart displays them
        df["Outstanding"] = pd.to_numeric(df["Outstanding"], errors="coerce").fillna(0)
        df["Issued"]      = pd.to_numeric(df["Issued"],      errors="coerce").fillna(0)

        # 3) Show raw table
        st.write("### Debt Details")
        st.dataframe(df, use_container_width=True)

        # 4) Build grouped bar chart
        fig = px.bar(
            df,
            x="Name",
            y=["Outstanding", "Issued"],
            barmode="group",
            labels={
                "value": "£ Amount",
                "Name": "Debt Instrument",
                "variable": "Status"
            },
            title="Outstanding vs. Issued Debt by Instrument",
            template="plotly_dark",
            color_discrete_map={
                "Outstanding": "#66c2a5",  # light teal
                "Issued":      "#fc8d62"   # salmon
            }
        )
        fig.update_layout(
            xaxis_tickangle=45,
            legend_title_text="",
            height=500
        )

        # 5) Draw Refinitiv DataPlatform total_debt marker (if available)
        total_debt_m = self.variables.get("total_debt", None)
        if total_debt_m is not None:
            # convert millions to full pounds
            total_debt_full = total_debt_m * 1_000_000

            fig.add_shape(
                type="line",
                x0="Total", x1="Total",
                y0=0, y1=total_debt_full,
                xref="x", yref="y",
                line=dict(color="yellow", width=2, dash="dash")
            )
            fig.add_annotation(
                x="Total", y=total_debt_full,
                text=f"DataPlatform total: £{total_debt_m:,} m",
                showarrow=True,
                arrowhead=2,
                arrowcolor="yellow",
                font=dict(color="yellow", size=12),
                yshift=10
            )

        # 6) Render chart and footnote
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            * “Outstanding” is the remaining balance from the Refinitiv Excel model.
            * “Issued” is original principal from the Refinitiv Excel model.
            * Yellow dashed line is the total debt pulled from Refinitiv DataPlatform live up-to-minute aggregate (in £ m).
            """,
            unsafe_allow_html=True
        )

    def display_visual_dashboard(self):

        self.display_header_dashboard()


    def display_share_price_chart(self):
        current_price = self.variables["current_share_price"]
        price_multiples = self.variables["share_price_multiples"]
        price_perpetuity = self.variables["share_price_perpetuity"]
        wacc = self.variables["wacc"]
        terminal_growth = self.variables["terminal_growth"]

        upside_multiples = ((price_multiples / current_price) - 1) * 100 if current_price else 0
        upside_perpetuity = ((price_perpetuity / current_price) - 1) * 100 if current_price else 0

        st.subheader("Share Price Analysis")
        tab1, tab2 , tab3 = st.tabs(["Price Comparison", "Upside Potential", "Ownership Summary"])
        with tab1:
            col1, col2 = st.columns([3, 2])
            with col1:
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=["Current Price", "Multiples", "Perpetuity"],
                    y=[current_price, price_multiples, price_perpetuity],
                    marker_color=["#455A64", "#1E88E5", "#FFC107"]
                ))
                fig_bar.update_layout(
                    title="Comparison of Current Price vs. Implied Prices",
                    xaxis_title="Method",
                    yaxis_title="Price (£)",
                    height=400,
                    font=dict(
                        color="#ffffff",
                        size=14,
                        family="Arial, sans-serif"
                    )
                )
                fig_bar.update_xaxes(
                    tickfont=dict(color="#ffffff"),
                    title_font=dict(color="#ffffff")
                )
                fig_bar.update_yaxes(
                    tickfont=dict(color="#ffffff"),
                    title_font=dict(color="#ffffff")
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            with col2:
                avg_price = (price_multiples + price_perpetuity) / 2
                st.metric("Current Price", f"£{current_price:.2f}")
                st.metric("Multiples Price", f"£{price_multiples:.2f}", f"{upside_multiples:.1f}%", delta_color="normal")
                st.metric("Perpetuity Price", f"£{price_perpetuity:.2f}", f"{upside_perpetuity:.1f}%", delta_color="normal")
                st.metric("Average Implied Price", f"£{avg_price:.2f}")
                st.write("### Key Inputs")
                st.write(f"- WACC: {wacc * 100:.2f}%")
                st.write(f"- Terminal Growth: {terminal_growth * 100:.2f}%")
        with tab2:
            max_upside = max(upside_multiples, upside_perpetuity)
            min_upside = min(upside_multiples, upside_perpetuity)
            fig_up = go.Figure()
            fig_up.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=max_upside,
                title={"text": "Max Upside", "font": {"size": 14}},
                gauge={"axis": {"range": [-50, 200]}, "bar": {"color": "#4CAF50"}},
                delta={"reference": 0, "relative": False},
                domain={"row": 0, "column": 0}
            ))
            fig_up.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=min_upside,
                title={"text": "Min Upside", "font": {"size": 14}},
                gauge={"axis": {"range": [-50, 200]}, "bar": {"color": "#FFC107"}},
                delta={"reference": 0, "relative": False},
                domain={"row": 1, "column": 0}
            ))
            fig_up.update_layout(
                grid={"rows": 2, "columns": 1},
                height=600,
                font=dict(
                    color="#ffffff",
                    size=14,
                    family="Arial, sans-serif"
                )
            )
            st.plotly_chart(fig_up, use_container_width=True)
            #--Ownership Summary tab--
        with tab3:
            st.header("Ownership Summary")
            # Load the Data sheet from Excel
            raw = pd.read_excel("attached_assets/EZJ.L Firm Ownership Summary.xlsx",
            sheet_name= "Data",header = None)

            # 1 Top 10 Shareholders

            inv = slice_section(raw, "Investor Rank", "Type", list(range(1,8))).head(10)
            inv["% O/S"] = parse_pct(inv["% O/S"])
            fig = go.Figure(go.Bar(x=inv["Investor Name"], y=inv["% O/S"]))
            fig.update_layout(title="Top 10 Shareholders by % Ownership",
            xaxis_tickangle=-45, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            # 2) Ownership % by Investor Type
            typ = slice_section(raw, "Type", "Style", [1, 2, 3])
            typ.columns = ['Type', 'Investors', '% O/S']
            typ['% O/S'] = parse_pct(typ['% O/S'])
            fig2 = go.Figure(
            data=[go.Pie(
                labels=typ['Type'],
                values=typ['% O/S'],
                hole=0.3
            )]
        )
            fig2.update_layout(
            title="Ownership % by Investor Type",
            template="plotly_dark"
        )
            st.plotly_chart(fig2, use_container_width=True)

            # 3) Ownership % by Investment Style
            sty = slice_section(raw, "Style", "Region", [1, 2, 3])
            sty.columns = ['Style', 'Investors', '% O/S']
            sty['% O/S'] = parse_pct(sty['% O/S'])
            fig3 = go.Figure(
            data=[go.Bar(
                x=sty['Style'],
                y=sty['% O/S'],
                marker_color="#FFC107"
            )]
        )
            fig3.update_layout(
            title="Ownership % by Investment Style",
            xaxis_tickangle=-45,
            template="plotly_dark"
        )
            st.plotly_chart(fig3, use_container_width=True)

            # 4) Ownership % by Region
            reg = slice_section(raw, "Region", "Recent Activity", [1, 2, 3])
            reg.columns = ['Region', 'Investors', '% O/S']
            reg['% O/S'] = parse_pct(reg['% O/S'])
            fig4 = go.Figure(
            data=[go.Bar(
                x=reg['Region'],
                y=reg['% O/S'],
                marker_color="#4CAF50"
            )]
        )
            fig4.update_layout(
            title="Ownership % by Region",
            xaxis_tickangle=-45,
            template="plotly_dark"
        )
            st.plotly_chart(fig4, use_container_width=True)

            # 5) Recent Trading Activity (Buys vs Sells)
            ra_block = raw.iloc[
            raw[0].tolist().index("Recent Activity")+2 :
            raw[0].tolist().index("Concentration"),
            [0, 1, 2]
            ].dropna(how='all')
            ra_block.columns = ['Investor', 'Value', 'Shares']
            ra_block['Shares'] = pd.to_numeric(ra_block['Shares'], errors='coerce')
            buys = ra_block[ra_block['Value'].astype(str).str.startswith('+')]
            sells = ra_block[ra_block['Value'].astype(str).str.startswith('-')]

            fig5 = go.Figure()
            fig5.add_trace(go.Bar(
            x=buys['Shares'],
            y=buys['Investor'],
            orientation='h',
            name='Buys',
            marker_color='#1E88E5'
        ))
            fig5.add_trace(go.Bar(
            x=-sells['Shares'],
            y=sells['Investor'],
            orientation='h',
            name='Sells',
            marker_color='#E53935'
        ))
            fig5.update_layout(
            title="Recent Trading Activity (Shares)",
            barmode='relative',
            template="plotly_dark"
        )
            st.plotly_chart(fig5, use_container_width=True)

            # 6) Holdings Concentration
            conc = slice_section(raw, "Concentration", None, [0, 2])
            conc.columns = ['Concentration', '% O/S']
            conc['% O/S'] = parse_pct(conc['% O/S'])
            fig6 = go.Figure(
            data=[go.Scatter(
                x=conc['Concentration'],
                y=conc['% O/S'],
                mode='lines+markers',
                line=dict(color='#FFC107')
            )]
        )
            fig6.update_layout(
            title="Holdings Concentration",
            template="plotly_dark"
        )
            st.plotly_chart(fig6, use_container_width=True)


    def display_sensitivity_analysis(self):
        pass

    def _display_wacc_sensitivity(self):
        pass

    def _display_growth_sensitivity(self):
        pass

    def _display_revenue_sensitivity(self):
        pass

    def _display_margin_sensitivity(self):
        pass

    def _display_two_factor_analysis(self, factor1, factor2):
        pass

    def _calculate_price_for_factors(self, factor1_key, val1, factor2_key, val2, factor_values):
        pass

    def _calculate_custom_scenario(self, wacc, growth, revenue_growth, margin):
        pass

    def _display_spider_chart(self, scenario):
        pass

    def display_all_visualizations(self):
        try:
            st.success("✅ Successfully loaded DCF model data!")
            with st.expander("Show extracted variables (debug)", expanded=False):
                st.write(self.variables)
            self.display_header_dashboard()
            st.header("DCF Model Visualizations")
            self.display_enterprise_value_chart()
            self.display_share_price_chart()
            self.display_sensitivity_analysis()
        except Exception as e:
            st.error(f"ERROR: Problem displaying visualizations: {str(e)}")
            st.exception(e)

