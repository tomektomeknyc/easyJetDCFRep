import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

class DCFAnalyzer:
    """
    A class to extract and visualize DCF model data from an Excel file.
    """

    def __init__(self, excel_df):
        """
        Initialize the DCF Analyzer with a DataFrame from the DCF tab

        Args:
            excel_df: DataFrame containing the DCF tab data
        """
        self.df = excel_df
        self.variables = self._extract_dcf_variables()

    def get_share_price_chart(self):
        """
        Returns a Plotly bar chart comparing current and implied share prices.
        """
        current_price = self.variables.get("current_share_price", 0)
        price_multiples = self.variables.get("share_price_multiples", 0)
        price_perpetuity = self.variables.get("share_price_perpetuity", 0)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Current Price", "Multiples", "Perpetuity"],
            y=[current_price, price_multiples, price_perpetuity],
            marker_color=["#455A64", "#1E88E5", "#FFC107"]
        ))
        fig.update_layout(
            title="Current vs. Implied Share Prices",
            xaxis_title="Valuation Method",
            yaxis_title="Price (£)",
            paper_bgcolor="#000",
            plot_bgcolor="#000",
            font=dict(color="#fff")
        )
        return fig


    def _extract_dcf_variables(self):
        """
        Extract DCF variables from specific cells in the DataFrame

        Returns:
            dict: Dictionary of extracted DCF variables
        """
        try:
            # Attempt direct indexing first
            try:
                wacc = self._extract_numeric_value(15, 4)
                terminal_growth = self._extract_numeric_value(17, 10)
                valuation_date = self._extract_date_value(9, 4)
                current_share_price = self._extract_numeric_value(12, 4)
                diluted_shares_outstanding = self._extract_numeric_value(13, 4)
                ev_multiples = self._extract_numeric_value(22, 10)
                ev_perpetuity = self._extract_numeric_value(22, 15)
                share_price_multiples = self._extract_numeric_value(37, 10)
                share_price_perpetuity = self._extract_numeric_value(37, 15)
            except Exception as e:
                st.warning(f"Attempting alternative cell extraction method due to: {str(e)}")

                wacc_row = self._locate_row_with_text("Discount Rate (WACC)")
                terminal_growth_row = self._locate_row_with_text("Implied Terminal FCF Growth Rate")
                valuation_date_row = self._locate_row_with_text("Valuation Date")
                share_price_row = self._locate_row_with_text("Current Share Price")
                shares_outstanding_row = self._locate_row_with_text("Diluted Shares Outstanding")
                ev_row = self._locate_row_with_text("Implied Enterprise Value")
                implied_share_row = self._locate_row_with_text("Implied Share Price")

                wacc = self._extract_numeric_from_row(wacc_row, 4) if wacc_row is not None else 0.1
                terminal_growth = self._extract_numeric_from_row(terminal_growth_row, 10) if terminal_growth_row is not None else 0.02
                valuation_date = self._extract_date_from_row(valuation_date_row, 4) if valuation_date_row is not None else datetime.now().strftime("%Y-%m-%d")
                current_share_price = self._extract_numeric_from_row(share_price_row, 4) if share_price_row is not None else 0
                diluted_shares_outstanding = self._extract_numeric_from_row(shares_outstanding_row, 4) if shares_outstanding_row is not None else 0
                ev_multiples = self._extract_numeric_from_row(ev_row, 10) if ev_row is not None else 0
                ev_perpetuity = self._extract_numeric_from_row(ev_row, 15) if ev_row is not None else 0
                share_price_multiples = self._extract_numeric_from_row(implied_share_row, 10) if implied_share_row is not None else 0
                share_price_perpetuity = self._extract_numeric_from_row(implied_share_row, 15) if implied_share_row is not None else 0

            # Ensure we never store None
            if wacc is None:
                wacc = 0.1
            if terminal_growth is None:
                terminal_growth = 0.02
            if current_share_price is None:
                current_share_price = 0
            if diluted_shares_outstanding is None:
                diluted_shares_outstanding = 0
            if ev_multiples is None:
                ev_multiples = 0
            if ev_perpetuity is None:
                ev_perpetuity = 0
            if share_price_multiples is None:
                share_price_multiples = 0
            if share_price_perpetuity is None:
                share_price_perpetuity = 0

            return {
                "wacc": wacc,
                "terminal_growth": terminal_growth,
                "valuation_date": valuation_date,
                "current_share_price": current_share_price,
                "diluted_shares_outstanding": diluted_shares_outstanding,
                "ev_multiples": ev_multiples,
                "ev_perpetuity": ev_perpetuity,
                "share_price_multiples": share_price_multiples,
                "share_price_perpetuity": share_price_perpetuity
            }
        except Exception as e:
            st.error(f"Error extracting DCF variables: {str(e)}")
            return {
                "wacc": 0.1,
                "terminal_growth": 0.02,
                "valuation_date": datetime.now().strftime("%Y-%m-%d"),
                "current_share_price": 5.0,
                "diluted_shares_outstanding": 1000,
                "ev_multiples": 5000,
                "ev_perpetuity": 5500,
                "share_price_multiples": 6.0,
                "share_price_perpetuity": 6.5
            }

    def _extract_numeric_value(self, row, col):
        try:
            value = self.df.iloc[row, col]
        except:
            return 0
        if pd.isna(value):
            return 0
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            temp = value.replace('$', '').replace('£', '').replace('€', '').replace(',', '')
            if '%' in temp:
                temp = temp.replace('%', '')
                try:
                    return float(temp) / 100.0
                except:
                    return 0
            else:
                try:
                    return float(temp)
                except:
                    return 0
        return 0

    def _extract_date_value(self, row, col):
        try:
            value = self.df.iloc[row, col]
        except:
            return datetime.now().strftime("%Y-%m-%d")
        if pd.isna(value):
            return datetime.now().strftime("%Y-%m-%d")
        if isinstance(value, (pd.Timestamp, datetime)):
            return value.strftime("%Y-%m-%d")
        if isinstance(value, str):
            try:
                return pd.to_datetime(value).strftime("%Y-%m-%d")
            except:
                return datetime.now().strftime("%Y-%m-%d")
        return datetime.now().strftime("%Y-%m-%d")

    def _locate_row_with_text(self, text):
        for i in range(len(self.df)):
            row_values = self.df.iloc[i].astype(str).str.contains(text, case=False, na=False)
            if any(row_values):
                return i
        return None

    def _extract_numeric_from_row(self, row, col):
        if row is None:
            return 0
        return self._extract_numeric_value(row, col)

    def _extract_date_from_row(self, row, col):
        if row is None:
            return datetime.now().strftime("%Y-%m-%d")
        return self._extract_date_value(row, col)

    def format_currency(self, value):
        if not value or pd.isna(value):
            return "£0.00"
        if value >= 1_000_000:
            return f"£{value/1_000_000:.2f}M"
        elif value >= 1_000:
            return f"£{value/1_000:.2f}K"
        else:
            return f"£{value:.2f}"

    def format_percentage(self, value):
        if not value or pd.isna(value):
            return "0.00%"
        return f"{value * 100:.2f}%"
    def extract_key_metrics_for_report(self):
        """
        Returns a dictionary of key DCF metrics for PDF report generation.
        """
        return {
            "Valuation Date": self.variables.get("valuation_date", ""),
            "Current Share Price": self.format_currency(self.variables.get("current_share_price", 0)),
            "Diluted Shares Outstanding": f"{self.variables.get('diluted_shares_outstanding', 0):,.2f}",
            "Discount Rate (WACC)": self.format_percentage(self.variables.get("wacc", 0)),
            "Terminal Growth Rate": self.format_percentage(self.variables.get("terminal_growth", 0)),
            "EV (Multiples)": self.format_currency(self.variables.get("ev_multiples", 0)),
            "EV (Perpetuity)": self.format_currency(self.variables.get("ev_perpetuity", 0)),
            "Share Price (Multiples)": self.format_currency(self.variables.get("share_price_multiples", 0)),
            "Share Price (Perpetuity)": self.format_currency(self.variables.get("share_price_perpetuity", 0)),
        }

    def get_enterprise_value_chart(self):
        """
        Returns a Plotly figure comparing Enterprise Value by method.
        """
        ev_multiples = self.variables.get("ev_multiples", 0)
        ev_perpetuity = self.variables.get("ev_perpetuity", 0)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Multiples Method",
            x=["EV"],
            y=[ev_multiples],
            marker_color="#1E88E5"
        ))
        fig.add_trace(go.Bar(
            name="Perpetuity Growth",
            x=["EV"],
            y=[ev_perpetuity],
            marker_color="#FFC107"
        ))

        fig.update_layout(
            title="Enterprise Value Comparison",
            barmode="group",
            paper_bgcolor="#000",
            plot_bgcolor="#000",
            font=dict(color="#fff")
        )
        return fig


    def display_key_metrics(self):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("DCF Model Key Variables")
            shares_out = self.variables["diluted_shares_outstanding"] or 0
            dcf_metrics = {
                "Valuation Date": self.variables["valuation_date"],
                "Current Share Price": self.format_currency(self.variables["current_share_price"]),
                "Diluted Shares Outstanding (millions)": f"{shares_out:,.2f}",
                "Discount Rate (WACC)": self.format_percentage(self.variables["wacc"]),
                "Implied Terminal FCF Growth Rate": self.format_percentage(self.variables["terminal_growth"])
            }
            for metric, value in dcf_metrics.items():
                st.metric(label=metric, value=value)
        with col2:
            st.subheader("Valuation Results")
            valuation_metrics = {
                "Implied Enterprise Value (Multiples)": self.format_currency(self.variables["ev_multiples"]),
                "Implied Enterprise Value (Perpetuity Growth)": self.format_currency(self.variables["ev_perpetuity"]),
                "Implied Share Price (Multiples)": self.format_currency(self.variables["share_price_multiples"]),
                "Implied Share Price (Perpetuity Growth)": self.format_currency(self.variables["share_price_perpetuity"])
            }
            for metric, value in valuation_metrics.items():
                st.metric(label=metric, value=value)

    ##########

    def display_key_metrics(self):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("DCF Model Key Variables")
            shares_out = self.variables["diluted_shares_outstanding"] or 0
            dcf_metrics = {
                "Valuation Date": self.variables["valuation_date"],
                "Current Share Price": self.format_currency(self.variables["current_share_price"]),
                "Diluted Shares Outstanding (millions)": f"{shares_out:,.2f}",
                "Discount Rate (WACC)": self.format_percentage(self.variables["wacc"]),
                "Implied Terminal FCF Growth Rate": self.format_percentage(self.variables["terminal_growth"])
            }
            for metric, value in dcf_metrics.items():
                st.metric(label=metric, value=value)
        with col2:
            st.subheader("Valuation Results")
            valuation_metrics = {
                "Implied Enterprise Value (Multiples)": self.format_currency(self.variables["ev_multiples"]),
                "Implied Enterprise Value (Perpetuity Growth)": self.format_currency(self.variables["ev_perpetuity"]),
                "Implied Share Price (Multiples)": self.format_currency(self.variables["share_price_multiples"]),
                "Implied Share Price (Perpetuity Growth)": self.format_currency(self.variables["share_price_perpetuity"])
            }
            for metric, value in valuation_metrics.items():
                st.metric(label=metric, value=value)
            wacc = self.variables.get('wacc', 0) * 100
            terminal_growth = self.variables.get('terminal_growth', 0) * 100
            st.metric(label="WACC / Terminal Growth", value=f"{wacc:.2f}% / {terminal_growth:.2f}%")


    ########## this place needs white outputs

    def display_enterprise_value_chart(self):
        ev_multiples = self.variables["ev_multiples"]
        ev_perpetuity = self.variables["ev_perpetuity"]
        ev_diff = ev_perpetuity - ev_multiples
        ev_pct_diff = (ev_diff / ev_multiples) * 100 if ev_multiples else 0
        st.subheader("Enterprise Value Analysis")
        col1, col2 = st.columns([3, 2])
        with col1:
            fig_ev = go.Figure()
            ev_multiples_components = {
                "Cash Flows": ev_multiples * 0.4,
                "Terminal Value": ev_multiples * 0.6
            }
            ev_perpetuity_components = {
                "Cash Flows": ev_perpetuity * 0.35,
                "Terminal Value": ev_perpetuity * 0.65
            }
            # Remove forced marker line color by not specifying the "line" property
            fig_ev.add_trace(go.Funnel(
                name="Enterprise Value Breakdown",
                y=["Enterprise Value (Multiples)", "Cash Flows", "Terminal Value",
                   "Enterprise Value (Perpetuity)", "Cash Flows", "Terminal Value"],
                x=[
                    ev_multiples,
                    ev_multiples_components["Cash Flows"],
                    ev_multiples_components["Terminal Value"],
                    ev_perpetuity,
                    ev_perpetuity_components["Cash Flows"],
                    ev_perpetuity_components["Terminal Value"]
                ],
                textposition="inside",
                textinfo="value+percent initial",
                textfont=dict(color="#ffffff"), #white font color
                opacity=1.00,
                marker={
                    "color": [" #3333ff", "#0aabf5", "#0D47A1",
                              "#99cc00", "#e6b800", "#FF8F00"]
                },
                connector={"line": {"color": "royalblue", "dash": "dot", "width": 3}},
                hoverinfo="text",
                hovertext=[
                    f"<b>Total EV (Multiples)</b>: {self.format_currency(ev_multiples)}<br>Method: EV/EBITDA Multiple",
                    f"<b>Cash Flows (M)</b>: {self.format_currency(ev_multiples_components['Cash Flows'])}",
                    f"<b>Terminal Value (M)</b>: {self.format_currency(ev_multiples_components['Terminal Value'])}",
                    f"<b>Total EV (Perpetuity)</b>: {self.format_currency(ev_perpetuity)}<br>Method: Perpetuity Growth",
                    f"<b>Cash Flows (P)</b>: {self.format_currency(ev_perpetuity_components['Cash Flows'])}",
                    f"<b>Terminal Value (P)</b>: {self.format_currency(ev_perpetuity_components['Terminal Value'])}"
                ]
            ))
            max_ev = max(ev_multiples, ev_perpetuity)
            fig_ev.add_annotation(
                x=1.0, y=1.0, xref="paper", yref="paper",
                text=f"Δ {self.format_currency(abs(ev_diff))}",
                showarrow=True, arrowhead=2, arrowcolor="royalblue", ax=-60
            )
            fig_ev.add_annotation(
                x=1.0, y=0.6, xref="paper", yref="paper",
                text=f"{abs(ev_pct_diff):.1f}% {'higher' if ev_perpetuity > ev_multiples else 'lower'}",
                showarrow=False
            )

            fig_ev.update_layout(
                title="Enterprise Value - Method Comparison",
                title_font_color="#ffffff",
                height=500,
                funnelmode="stack",
                showlegend=False,
                paper_bgcolor="#000",   # Set outer background to jet black
                plot_bgcolor="#000",    # Set inner plotting area to jet black
                font=dict(             # Specify the font dictionary explicitly
                color="#ffffff",      # Bright white text
                size=14,           # Adjust font size if needed
                family="Arial, sans-serif"  # Choose a font family that renders clearly
        )
    )
            fig_ev.update_xaxes(tickfont=dict(color="#ffffff"))
            fig_ev.update_yaxes(tickfont=dict(color="rgba(255,255,255,1.0)"))

            st.plotly_chart(fig_ev, use_container_width=True)

        with col2:
            max_val = max(ev_multiples, ev_perpetuity)
            min_val = min(ev_multiples, ev_perpetuity)
            fig_gauge = go.Figure()
            fig_gauge.add_trace(go.Indicator(
              mode="gauge+number+delta",
              value=ev_multiples,
              title={"text": "Multiples Method", "font": {"size": 14}},
              gauge={
                "axis": {"range": [0, max_val * 1.2]},
                "bar": {"color": "#3333ff"}  #### adjust color
              },
              delta={"reference": ev_perpetuity, "relative": True},
              domain={"x": [0, 1], "y": [0.55, 1]}
            ))
            fig_gauge.add_trace(go.Indicator(
              mode="gauge+number+delta",
              value=ev_perpetuity,
              title={"text": "Perpetuity Method", "font": {"size": 14}},
              gauge={
                "axis": {"range": [0, max_val * 1.2]},
                "bar": {"color": "#99cc00"}  ## adjust color
              },
              delta={"reference": ev_multiples, "relative": True},
              domain={"x": [0, 1], "y": [0, 0.45]}
            ))
            fig_gauge.update_layout(
              height=500,
              margin=dict(l=50, r=50, t=50, b=50),
              paper_bgcolor="#000",  # Outer background set to jet black
              plot_bgcolor="#000",   # Inner plotting area set to jet black
              font=dict(color="#ffffff")  # White text throughout I changed from fff
            )
            fig_gauge.update_xaxes(tickfont=dict(color="#ffffff"))
            fig_gauge.update_yaxes(tickfont=dict(color="#ffffff"))


            st.plotly_chart(fig_gauge, use_container_width=True)

            ev_pct_diff_abs = abs(ev_pct_diff)
            if ev_pct_diff_abs > 20:
                insight_level = "very significant"
                insight_color = "#d32f2f"
            elif ev_pct_diff_abs > 10:
                insight_level = "significant"
                insight_color = "#f57c00"
            elif ev_pct_diff_abs > 5:
                insight_level = "moderate"
                insight_color = "#fbc02d"
            else:
                insight_level = "minimal"
                insight_color = "#388e3c"
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, {insight_color}20, {insight_color}05);
                        border-left: 5px solid {insight_color};
                        padding: 15px;
                        border-radius: 5px;
                        margin-top: 20px;">
                <h4 style="margin-top:0; color: {insight_color};">Valuation Confidence: {insight_level.title()}</h4>
                <p>Difference between methods: <b>{abs(ev_pct_diff):.1f}%</b></p>
                <p>EV Range: {self.format_currency(min_val)} - {self.format_currency(max_val)}</p>
            </div>
            """, unsafe_allow_html=True)

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
                    paper_bgcolor="#000",   # Set outer background to jet black
                plot_bgcolor="#000",    # Set inner plotting area to jet black
                font = dict(
                    color="#ffffff",
                    size=14,
                    family = "Arial, sans-serif"
                )
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            with col2:
                avg_price = (price_multiples + price_perpetuity) / 2
                st.metric("Current Price", f"£{current_price:.2f}")
                st.metric("Multiples Price", f"£{price_multiples:.2f}", f"{upside_multiples:.1f}%")
                st.metric("Perpetuity Price", f"£{price_perpetuity:.2f}", f"{upside_perpetuity:.1f}%")
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
            fig_up.update_layout(grid={"rows": 2, "columns": 1},
                                 height=600,paper_bgcolor="#000", # Set outer background to jet black
                plot_bgcolor="#000", # Set inner plotting area to jet black
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
            self.display_key_metrics()
            st.header("DCF Model Visualizations")
            self.display_enterprise_value_chart()
            self.display_share_price_chart()
            self.display_sensitivity_analysis()
        except Exception as e:
            st.error(f"ERROR: Problem displaying visualizations: {str(e)}")
            st.exception(e)
