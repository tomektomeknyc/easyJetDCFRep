# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import altair as alt
# import math
# import random

# # Optional extras – if you have these packages installed, they add extra styling.
# from streamlit_extras.metric_cards import style_metric_cards
# from streamlit_extras.stylable_container import stylable_container
# from streamlit_extras.chart_container import chart_container

# class AdvancedVisualizations:
#     """
#     A class providing advanced financial visualizations using data extracted from the Excel DCF model.
#     """

#     def __init__(self, dcf_analyzer):
#         self.dcf = dcf_analyzer
#         self.variables = dcf_analyzer.variables
#         self.color_palette = {
#             'primary': '#2196F3',
#             'secondary': '#FF9800',
#             'tertiary': '#4CAF50',
#             'quaternary': '#9C27B0',
#             'negative': '#F44336',
#             'positive': '#4CAF50',
#             'neutral': '#9E9E9E',
#             'background': '#F5F5F5',
#             'grid': 'rgba(0,0,0,0.05)'
#         }
#         self.gradient_palette = self._generate_gradient_palette()

#     def _generate_gradient_palette(self, num_colors=20):
#         colors = []
#         for i in range(num_colors):
#             r = int(33 + (242 - 33) * i / (num_colors - 1))
#             g = int(150 + (153 - 150) * i / (num_colors - 1))
#             b = int(243 + (0 - 243) * i / (num_colors - 1))
#             colors.append(f'rgb({r},{g},{b})')
#         return colors

#     def format_currency(self, value):
#         if isinstance(value, str):
#             return value
#         if math.isnan(value):
#             return "N/A"
#         if abs(value) >= 1e6:
#             return f"£{value/1_000_000:.2f}M"
#         elif abs(value) >= 1e3:
#             return f"£{value/1_000:.2f}K"
#         else:
#             return f"£{value:.2f}"

#     def format_percentage(self, value):
#         if isinstance(value, str):
#             return value
#         if math.isnan(value):
#             return "N/A"
#         return f"{value*100:.2f}%"

#     def display_header_dashboard(self):
#         st.markdown("""
#         <style>
#         .metric-header {
#             font-size: 1.5rem;
#             font-weight: 600;
#             margin-bottom: 1rem;
#         }
#         </style>
#         <div class="metric-header">EasyJet Financial Summary</div>
#         """, unsafe_allow_html=True)

#         col1, col2, col3, col4 = st.columns(4)

#         # 1) Current Price
#         with col1:
#             current_price = self.variables.get('current_share_price', 0)
#             with stylable_container(
#                 key="current_price_container",
#                 css_styles="{ border-radius: 10px; padding: 15px; }"
#             ):
#                 st.metric(label="Current Share Price", value=f"£{current_price:.2f}")

#         # 2) DCF Multiples Price
#         with col2:
#             multiples_price = self.variables.get('share_price_multiples', 0)
#             base_price = current_price if current_price != 0 else 1
#             pct_diff_multiples = ((multiples_price / base_price) - 1) * 100
#             with stylable_container(
#                 key="multiples_price_container",
#                 css_styles="{ border-radius: 10px; padding: 15px; }"
#             ):
#                 st.metric(
#                     label="DCF Multiples Price",
#                     value=f"£{multiples_price:.2f}",
#                     delta=f"{pct_diff_multiples:.1f}%",
#                     delta_color="normal"
#                 )

#         # 3) DCF Perpetuity Price
#         with col3:
#             perpetuity_price = self.variables.get('share_price_perpetuity', 0)
#             pct_diff_perpetuity = ((perpetuity_price / base_price) - 1) * 100
#             with stylable_container(
#                 key="perpetuity_price_container",
#                 css_styles="{ border-radius: 10px; padding: 15px; }"
#             ):
#                 st.metric(
#                     label="DCF Perpetuity Price",
#                     value=f"£{perpetuity_price:.2f}",
#                     delta=f"{pct_diff_perpetuity:.1f}%",
#                     delta_color="normal"
#                 )

#         # 4) WACC / Terminal Growth (Updated label)
#         with col4:
#             wacc = self.variables.get('wacc', 0) * 100
#             growth = self.variables.get('terminal_growth', 0) * 100
#             with stylable_container(
#                 key="wacc_growth_container",
#                 css_styles="{ border-radius: 10px; padding: 15px; }"
#             ):
#                 st.metric(label="WACC / Terminal Growth", value=f"{wacc:.2f}% / {growth:.2f}%")

#     def _display_3d_sensitivity_with_real_data(self):
#         base_wacc = self.variables.get("wacc", 0.10)
#         base_growth = self.variables.get("terminal_growth", 0.02)
#         base_ev = self.variables.get("ev_perpetuity", 5000)

#         st.markdown("#### 3D EV Sensitivity (Using Excel-derived values)")
#         wacc_range = np.linspace(base_wacc * 0.5, base_wacc * 1.5, 30)
#         growth_range = np.linspace(base_growth * 0.5, base_growth * 1.5, 30)
#         wacc_grid, growth_grid = np.meshgrid(wacc_range, growth_range)
#         ev_surface = np.zeros_like(wacc_grid)
#         for i in range(wacc_grid.shape[0]):
#             for j in range(wacc_grid.shape[1]):
#                 w = wacc_grid[i, j]
#                 g = growth_grid[i, j]
#                 ratio = (base_wacc / w) ** 1.2 * ((1 + g) / (1 + base_growth))
#                 ev_surface[i, j] = base_ev * ratio

#         fig = go.Figure(data=[go.Surface(
#             x=wacc_grid,
#             y=growth_grid,
#             z=ev_surface,
#             colorscale='Viridis',
#             hovertemplate=(
#                 "WACC: %{x:.2%}<br>" +
#                 "Terminal Growth: %{y:.2%}<br>" +
#                 "Enterprise Value: £%{z:.2f}M<extra></extra>"
#             )
#         )])
#         fig.update_layout(
#             scene=dict(
#                 xaxis=dict(title="WACC"),
#                 yaxis=dict(title="Terminal Growth"),
#                 zaxis=dict(title="Enterprise Value (M)")
#             ),
#             margin=dict(l=0, r=0, t=30, b=0),
#             height=500,
#             paper_bgcolor="#000",
#             plot_bgcolor="#000"

#         )
#         st.plotly_chart(fig, use_container_width=True)
#         st.info("This chart uses real WACC, Growth, and EV values from Excel to display a sensitivity surface.")

#     def display_share_price_sunburst(self):
#         st.subheader("Share Price Sunburst Chart")
#         enterprise_value = max(self.variables.get('ev_perpetuity', 0), self.variables.get('ev_multiples', 0))
#         net_debt = self.variables.get('net_debt', enterprise_value * 0.3)
#         equity_value = enterprise_value - net_debt

#         sunburst_data = {
#             'labels': ['Total Enterprise Value', 'Net Debt', 'Equity Value', 'Historical FCF', 'Terminal Value'],
#             'parents': ['', 'Total Enterprise Value', 'Total Enterprise Value', 'Equity Value', 'Equity Value'],
#             'values': [enterprise_value, net_debt, equity_value, equity_value * 0.35, equity_value * 0.65]
#         }
#         fig = go.Figure(go.Sunburst(
#             labels=sunburst_data['labels'],
#             parents=sunburst_data['parents'],
#             values=sunburst_data['values'],
#             branchvalues='total',
#             texttemplate='<b>%{label}</b><br>£%{value:.1f}M<br>%{percentEntry:.1%}',
#             hovertemplate='<b>%{label}</b><br>Value: £%{value:.2f}M<br>%{percentEntry:.2%}<extra></extra>',
#             marker=dict(colors=['#1E88E5', '#F44336', '#4CAF50', '#9C27B0', '#FF9800'],
#                         line=dict(color='black', width=1)),
#             rotation=90
#         ))
#         # Update the sunburst trace so that the text (including the center label) is black.
#         fig.update_traces(textfont=dict(color="black"))
#         fig.update_layout(
#             height=600,
#             margin=dict(t=10, l=10, r=10, b=10),
#             paper_bgcolor="#000",
#             plot_bgcolor="#000"
#         )
#         st.plotly_chart(fig, use_container_width=True)

#     def display_wacc_analysis_dashboard(self):
#         st.subheader("WACC Analysis Dashboard")
#         base_wacc = self.variables.get('wacc', 0.10)
#         st.write(f"Current WACC: {base_wacc*100:.2f}%")
#         fig = go.Figure(go.Waterfall(
#             name="WACC Components",
#             orientation="v",
#             measure=["absolute", "relative", "total"],
#             x=["Equity Component", "Debt Component", "Total WACC"],
#             y=[base_wacc*70, base_wacc*30, 0],
#             text=[f"{base_wacc*70*100:.2f}%", f"{base_wacc*30*100:.2f}%", f"{base_wacc*100:.2f}%"],
#             connector={"line": {"color": "rgba(0,0,0,0.3)"}}
#         ))
#         fig.update_traces(textfont=dict())
#         fig.update_layout(
#             title="WACC Build-up",
#             showlegend=False,
#             height=500,
#             paper_bgcolor="#000",
#             plot_bgcolor="#000"
#         )
#         st.plotly_chart(fig, use_container_width=True)
#         st.info("Adjust parameters (if implemented) to see the impact of changes in WACC components.")

#     def display_two_factor_heatmap(self):
#         st.subheader("Two-Factor Sensitivity Heatmap")
#         base_wacc = self.variables.get('wacc', 0.10)
#         base_growth = self.variables.get('terminal_growth', 0.02)
#         base_price = self.variables.get('share_price_perpetuity', 0)

#         wacc_range = np.linspace(base_wacc * 0.9, base_wacc * 1.1, 20)
#         growth_range = np.linspace(base_growth * 0.9, base_growth * 1.1, 20)
#         wacc_grid, growth_grid = np.meshgrid(wacc_range, growth_range)
#         price_grid = base_price * (base_wacc / wacc_grid) * ((1 + growth_grid) / (1 + base_growth))
#         pct_change = ((price_grid / base_price) - 1) * 100

#         fig = go.Figure(data=go.Heatmap(
#             z=pct_change,
#             x=np.round(wacc_range*100,2),
#             y=np.round(growth_range*100,2),
#             colorscale='Viridis',
#             colorbar=dict(
#                 title=dict(text="% Change"),
#                 tickfont=dict()
#             )
#         ))
#         fig.update_layout(
#             title="Sensitivity of Share Price to WACC and Terminal Growth",
#             xaxis_title="WACC (%)",
#             yaxis_title="Terminal Growth (%)",
#             height=500,
#             margin=dict(l=50, r=50, t=80, b=50),
#             paper_bgcolor="#000",
#             plot_bgcolor="#000"
#         )
#         st.plotly_chart(fig, use_container_width=True)
#         st.info("This heatmap shows the impact on share price when varying WACC and Terminal Growth.")
# #########
# def display_peer_analysis(self):
#     """
#     Displays weekly returns for EasyJet and its peers over the last X years
#     (controlled by a slider). Each ticker is loaded from a CSV in attached_assets,
#     and all data is combined in a single line chart.
#     """
#     st.subheader("Peer Analysis")
#     st.write("This section displays weekly returns for EasyJet and its peers over the selected time period.")

#     # 1) Let the user pick how many years back to display (1-20, default 10)
#     n_years = st.slider("Select number of years to display", min_value=1, max_value=20, value=10)

#     # 2) Determine date range

#     end_date = pd.to_datetime("today")
#     start_date = end_date - pd.DateOffset(years=n_years)
#     st.write(f"Displaying data from {start_date.date()} to {end_date.date()}")

#     # 3) List of tickers (including EasyJet and its peers).
#     #    Adjust or expand this list as you add more CSV files.
#     tickers = [
#         "EZJ.L",   # EasyJet
#         "RYA.I",   # Ryanair
#         "WIZZ.L",  # Wizz Air
#         "LHAG.DE", # Lufthansa
#         "ICAG.L",  # IAG (British Airways parent)
#         "AIRF.PA", # Air France-KLM
#         "JET2.L",  # Jet2
#         "KNIN.S"   # ?
#     ]

#     # 4) Prepare an array of custom colors so each ticker has a distinct color
#     color_sequence = [
#         "#1E88E5",  # Blue
#         "#FF6633",  # Orange
#         "#6A1B9A",  # Purple
#         "#FDD835",  # Yellow
#         "#43A047",  # Green
#         "#E53935",  # Red
#         "#8E24AA",  # Deep Purple
#         "#00ACC1"   # Teal
#     ]

#     df_list = []  # we'll store each ticker's weekly returns in here

#     # 5) For each ticker, load the CSV, compute weekly returns, add to df_list
#     for i, ticker in enumerate(tickers):
#         # CSV file naming convention: periods replaced by underscores
#         file_name = f"attached_assets/{ticker.replace('.', '_')}_returns.csv"
#         try:
#             df = pd.read_csv(file_name, index_col=0, parse_dates=True)
#             if "CLOSE" not in df.columns:
#                 st.error(f"File {file_name} missing 'CLOSE' column. Skipping.")
#                 continue

#             # Filter data to the chosen date range
#             df = df.loc[df.index >= start_date]

#             # Resample to weekly, taking the last close each week
#             weekly_close = df["CLOSE"].resample("W").last()

#             # Calculate weekly returns (percentage change)
#             weekly_returns = weekly_close.pct_change().dropna()

#             # Convert to DataFrame for plotting with a uniform structure
#             weekly_df = weekly_returns.to_frame(name="Returns").reset_index()
#             weekly_df.rename(columns={"index": "Date"}, inplace=True)

#             # Add a 'Ticker' column
#             weekly_df["Ticker"] = ticker

#             df_list.append(weekly_df)
#         except Exception as e:
#             st.error(f"Error reading file {file_name}: {e}")

#     # 6) Combine data if we have any loaded
#     if df_list:
#         combined_df = pd.concat(df_list)
#         # Plotly Express line chart: x=Date, y=Returns, color=Ticker
#         fig_peer = px.line(
#             combined_df,
#             x="Date",
#             y="Returns",
#             color="Ticker",
#             title=f"Weekly Returns for the Last {n_years} Year(s)",
#             color_discrete_sequence=color_sequence
#         )

#         # 7) Customize layout for a dark background, bright text, and a custom legend title
#         #    We assume 'plot_bg' and 'text_color' are global or stored in self; if not, adjust accordingly.
#         fig_peer.update_layout(
#             paper_bgcolor="#000",
#             plot_bgcolor="#000",
#             font=dict(color="#fff", size=14, family="Arial, sans-serif"),
#             title_font_color="#fff",
#             legend_title_text="Peers"  # Custom legend title
#         )
#         fig_peer.update_xaxes(
#             tickfont=dict(color="#fff"),
#             title="Date",
#             title_font=dict(color="#fff")
#         )
#         fig_peer.update_yaxes(
#             tickfont=dict(color="#fff"),
#             title="Weekly Returns (%)",
#             title_font=dict(color="#fff")
#         )

#         # 8) Display the chart
#         st.plotly_chart(fig_peer, use_container_width=True)
#     else:
#         st.write("No return data available for the selected tickers.")
# ########




#     def display_visual_dashboard(self):
#         st.subheader("Advanced Visualizations (Real Data)")
#         self.display_header_dashboard()
#         st.write("Below are additional visualizations derived from Excel data:")

#     def display_share_price_chart(self):
#         current_price = self.variables["current_share_price"]
#         price_multiples = self.variables["share_price_multiples"]
#         price_perpetuity = self.variables["share_price_perpetuity"]
#         wacc = self.variables["wacc"]
#         terminal_growth = self.variables["terminal_growth"]

#         upside_multiples = ((price_multiples / current_price) - 1) * 100 if current_price else 0
#         upside_perpetuity = ((price_perpetuity / current_price) - 1) * 100 if current_price else 0

#         st.subheader("Share Price Analysis")

#         tab1, tab2 = st.tabs(["Price Comparison", "Upside Potential"])

#         with tab1:
#             col1, col2 = st.columns([3, 2])
#             with col1:
#                 fig_bar = go.Figure()
#                 fig_bar.add_trace(go.Bar(
#                     x=["Current Price", "Multiples", "Perpetuity"],
#                     y=[current_price, price_multiples, price_perpetuity],
#                     marker_color=["#455A64", "#1E88E5", "#FFC107"]
#                 ))
#                 fig_bar.update_layout(
#                     title="Comparison of Current Price vs. Implied Prices",
#                     xaxis_title="Method",
#                     yaxis_title="Price (£)",
#                     height=400,font = dict(
#                     color="#ffffff",
#                     size=14,
#                     family = "Arial, sans-serif"

#                 ))



#                 fig_bar.update_xaxes(
#     tickfont=dict(color="#ffffff"),
#     title_font=dict(color="#ffffff")
# )

#                 fig_bar.update_yaxes(
#     tickfont=dict(color="#ffffff"),
#     title_font=dict(color="#ffffff")
# )



#                 st.plotly_chart(fig_bar, use_container_width=True)
#             with col2:
#                 avg_price = (price_multiples + price_perpetuity) / 2
#                 st.metric("Current Price", f"£{current_price:.2f}")
#                 st.metric("Multiples Price", f"£{price_multiples:.2f}", f"{upside_multiples:.1f}%", delta_color="normal")
#                 st.metric("Perpetuity Price", f"£{price_perpetuity:.2f}", f"{upside_perpetuity:.1f}%", delta_color="normal")
#                 st.metric("Average Implied Price", f"£{avg_price:.2f}")
#                 st.write("### Key Inputs")
#                 st.write(f"- WACC: {wacc * 100:.2f}%")
#                 st.write(f"- Terminal Growth: {terminal_growth * 100:.2f}%")
#         with tab2:
#             max_upside = max(upside_multiples, upside_perpetuity)
#             min_upside = min(upside_multiples, upside_perpetuity)
#             fig_up = go.Figure()
#             fig_up.add_trace(go.Indicator(
#                 mode="gauge+number+delta",
#                 value=max_upside,
#                 title={"text": "Max Upside", "font": {"size": 14}},
#                 gauge={"axis": {"range": [-50, 200]}, "bar": {"color": "#4CAF50"}},
#                 delta={"reference": 0, "relative": False},
#                 domain={"row": 0, "column": 0}
#             ))
#             fig_up.add_trace(go.Indicator(
#                 mode="gauge+number+delta",
#                 value=min_upside,
#                 title={"text": "Min Upside", "font": {"size": 14}},
#                 gauge={"axis": {"range": [-50, 200]}, "bar": {"color": "#FFC107"}},
#                 delta={"reference": 0, "relative": False},
#                 domain={"row": 1, "column": 0}
#             ))
#             fig_up.update_layout(grid={"rows": 2, "columns": 1}, height=600,
#                     font = dict(
#                     color="#ffffff",
#                     size=14,
#                     family = "Arial, sans-serif"))
#             st.plotly_chart(fig_up, use_container_width=True)

#     def display_sensitivity_analysis(self):
#         pass

#     def _display_wacc_sensitivity(self):
#         pass

#     def _display_growth_sensitivity(self):
#         pass

#     def _display_revenue_sensitivity(self):
#         pass

#     def _display_margin_sensitivity(self):
#         pass

#     def _display_two_factor_analysis(self, factor1, factor2):
#         pass

#     def _calculate_price_for_factors(self, factor1_key, val1, factor2_key, val2, factor_values):
#         pass

#     def _calculate_custom_scenario(self, wacc, growth, revenue_growth, margin):
#         pass

#     def _display_spider_chart(self, scenario):
#         pass

#     def display_all_visualizations(self):
#         try:
#             st.success("✅ Successfully loaded DCF model data!")
#             with st.expander("Show extracted variables (debug)", expanded=False):
#                 st.write(self.variables)
#             self.display_header_dashboard()
#             st.header("DCF Model Visualizations")
#             self.display_enterprise_value_chart()
#             self.display_share_price_chart()
#             self.display_sensitivity_analysis()
#         except Exception as e:
#             st.error(f"ERROR: Problem displaying visualizations: {str(e)}")
#             st.exception(e)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
import math
import random

# Optional extras – if you have these packages installed, they add extra styling.
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.chart_container import chart_container

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
        base_wacc = self.variables.get('wacc', 0.10)
        st.write(f"Current WACC: {base_wacc*100:.2f}%")
        fig = go.Figure(go.Waterfall(
            name="WACC Components",
            orientation="v",
            measure=["absolute", "relative", "total"],
            x=["Equity Component", "Debt Component", "Total WACC"],
            y=[base_wacc*70, base_wacc*30, 0],
            text=[f"{base_wacc*70*100:.2f}%", f"{base_wacc*30*100:.2f}%", f"{base_wacc*100:.2f}%"],
            connector={"line": {"color": "rgba(0,0,0,0.3)"}}
        ))
        fig.update_traces(textfont=dict())
        fig.update_layout(
            title="WACC Build-up",
            showlegend=False,
            height=500,
            paper_bgcolor="#000",
            plot_bgcolor="#000"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("Adjust parameters (if implemented) to see the impact of changes in WACC components.")

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
                legend_title_text="Peers"
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

    def display_visual_dashboard(self):
        st.subheader("Advanced Visualizations (Real Data)")
        self.display_header_dashboard()
        st.write("Below are additional visualizations derived from Excel data:")

    def display_share_price_chart(self):
        current_price = self.variables["current_share_price"]
        price_multiples = self.variables["share_price_multiples"]
        price_perpetuity = self.variables["share_price_perpetuity"]
        wacc = self.variables["wacc"]
        terminal_growth = self.variables["terminal_growth"]

        upside_multiples = ((price_multiples / current_price) - 1) * 100 if current_price else 0
        upside_perpetuity = ((price_perpetuity / current_price) - 1) * 100 if current_price else 0

        st.subheader("Share Price Analysis")
        tab1, tab2 = st.tabs(["Price Comparison", "Upside Potential"])
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
