import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from utils_base64 import get_base64_image

def parse_pct(series):
    """Convert strings like '15.27%' to float."""
    return pd.to_numeric(series.astype(str).str.rstrip('%'), errors='coerce')

def slice_section(raw, start_label, end_label, cols):
    """
    Given a DataFrame `raw` whose first column contains header labels,
    find the row matching `start_label`, then return the block of rows
    up to (but not including) `end_label`, restricted to `cols`.
    """
    idx = { val.strip(): i for i, val in raw.iloc[:,0].items() if isinstance(val, str) }
    start = idx[start_label]
    end = idx[end_label] if end_label else len(raw)
    block = raw.iloc[start+1:end, cols].dropna(how='all')
    block.columns = raw.iloc[start, cols]
    return block


class DCFAnalyzer:
    """
    A class to extract and visualize DCF model data from an Excel file.
    """

    def __init__(self, excel_df):
        """
        Initialize the DCF Analyzer with a DataFrame from the DCF tab.
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
        Extract DCF variables from specific cells in the DataFrame.
        Returns a dictionary of extracted variables.
        """
        # Initialize optional variables to avoid "referenced before assignment" error
        pe_ratio = 0
        pb_ratio = 0
        ev_ebitda = 0
        sensitivity = 0
        current_model_ev_ebitda = 0
        current_industry_ev_ebitda = 0
        current_ev_diff_percent = 0
        # print("P/E cell raw value:", self.df.iloc[27, 5])
        # print("P/B cell raw value:", self.df.iloc[28, 5])

        try:
            try:
                # Existing extractions
                wacc = self._extract_numeric_value(15, 4)
                terminal_growth = self._extract_numeric_value(17, 10)
                valuation_date = self._extract_date_value(9, 4)
                current_share_price = self._extract_numeric_value(12, 4)
                diluted_shares_outstanding = self._extract_numeric_value(13, 4)
                ev_multiples = self._extract_numeric_value(22, 10)
                ev_perpetuity = self._extract_numeric_value(22, 15)
                share_price_multiples = self._extract_numeric_value(37, 10)
                share_price_perpetuity = self._extract_numeric_value(37, 15)
                total_debt= round(self._extract_numeric_value(20,4),4) or 0.0
                cash_and_investments = round(self._extract_numeric_value( 26, 10),4) or 0.0


                # NEW EXTRACTIONS: Adjust these row and column indices to match your Excel DCF tab layout.
                current_pe_ratio = round(self._extract_numeric_value(27, 5),4)       # Example: P/E Ratio at row 29, col 6
                current_pb_ratio = round(self._extract_numeric_value(28, 5),4)       # Example: P/B Ratio at row 30, col 6
               # EV/EBITDA difference between model and the industry will be applied
                current_model_ev_ebitda = round(self._extract_numeric_value(11, 10),4) #4.9 Baseline Terminal EBITDA
                current_industry_ev_ebitda = round(self._extract_numeric_value(8, 10),4)  #5.9 Industry EBITDA
                current_ev_diff_percent = round((((current_industry_ev_ebitda - current_model_ev_ebitda) / current_industry_ev_ebitda) * 100),4)





            except Exception as e:
                st.warning(f"Attempting alternative extraction due to: {str(e)}")
                wacc_row = self._locate_row_with_text("Discount Rate (WACC)")
                terminal_growth_row = self._locate_row_with_text("Implied Terminal FCF Growth Rate")
                valuation_date_row = self._locate_row_with_text("Valuation Date")
                share_price_row = self._locate_row_with_text("Current Share Price")
                shares_outstanding_row = self._locate_row_with_text("Diluted Shares Outstanding")
                ev_row = self._locate_row_with_text("Implied Enterprise Value")
                implied_share_row = self._locate_row_with_text("Implied Share Price")
                industry_ev_ebitda_row = self._locate_row_with_text("Median CY23 TEV / EBITDA of Comps")
                model_ev_ebitda_row = self._locate_row_with_text("Baseline Terminal EBITDA Multiple")
                pe_ratio_row= self._locate_row_with_text("P/E")
                pb_ratio_row= self._locate_row_with_text("P/B")
                total_debt_row= self.locate_row_with_text("Total Debt")
                cash_and_investments_row= self.locate_row_with_text("Cash & Investments")

                wacc = self._extract_numeric_from_row(wacc_row, 4) if wacc_row is not None else 0.1
                terminal_growth = self._extract_numeric_from_row(terminal_growth_row, 10) if terminal_growth_row is not None else 0.02
                valuation_date = self._extract_date_from_row(valuation_date_row, 4) if valuation_date_row is not None else datetime.now().strftime("%Y-%m-%d")
                current_share_price = self._extract_numeric_from_row(share_price_row, 4) if share_price_row is not None else 0
                diluted_shares_outstanding = self._extract_numeric_from_row(shares_outstanding_row, 4) if shares_outstanding_row is not None else 0
                ev_multiples = self._extract_numeric_from_row(ev_row, 10) if ev_row is not None else 0
                ev_perpetuity = self._extract_numeric_from_row(ev_row, 15) if ev_row is not None else 0
                share_price_multiples = self._extract_numeric_from_row(implied_share_row, 10) if implied_share_row is not None else 0
                share_price_perpetuity = self._extract_numeric_from_row(implied_share_row, 15) if implied_share_row is not None else 0
                # Fix it
                current_industry_ev_ebitda =self._extract_numeric_from_row(industry_ev_ebitda_row, 10) if industry_ev_ebitda_row is not None else 0.0
                current_model_ev_ebitda = self._extract_numeric_from_row(model_ev_ebitda_row, 10) if model_ev_ebitda_row is not None else 0.0

                current_pe_ratio = self._extract_numeric_from_row(pe_ratio_row, 5) if pe_ratio_row is not None else 0.0
                current_pb_ratio = self._extract_numeric_from_row(pb_ratio_row, 5) if pb_ratio_row is not None else 0.0
                # Set defa
                total_debt= self._extract_numeric_from_row(total_debt_row, 4) if total_debt_row is not None else 0.0
                cash_and_investments= self.extract_numerc_from_row(cash_and_investments_row,10) if cash_and_investments_row is not None else 0.0
                # Set default values for additional metrics if alternative extraction is used
                # pe_ratio = 0
                # pb_ratio = 0
                # ev_ebitda = 0
                # sensitivity = 0
                # model_ev_ebitda = 0
                # industry_ev_ebitda = 0
                # ev_diff_percent = 0

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
            if pe_ratio is None:
                pe_ratio = 0
            if pb_ratio is None:
                pb_ratio = 0
            if ev_ebitda is None:
                ev_ebitda = 0
            if sensitivity is None:
                sensitivity = 0
            if current_industry_ev_ebitda  is None:
                  current_industry_ev_ebitda  = 0.0
            if current_model_ev_ebitda is None:
                  current_model_ev_ebitda = 0.0
            if current_ev_diff_percent is None:
                  current_ev_diff_percent = 0.0
            if current_pb_ratio is None:
                  current_pb_ratio = 0.0
            if current_pe_ratio is None:
                  current_pe_ratio = 0.0
            if total_debt is None:
                  total_debt = 0.0
            if cash_and_investments is None:
                  cash_and_investments= 0.0







            return {
                "wacc": wacc,
                "terminal_growth": terminal_growth,
                "valuation_date": valuation_date,
                "current_share_price": current_share_price,
                "diluted_shares_outstanding": diluted_shares_outstanding,
                "ev_multiples": ev_multiples,
                "ev_perpetuity": ev_perpetuity,
                "share_price_multiples": share_price_multiples,
                "share_price_perpetuity": share_price_perpetuity,
                "current_pe_ratio": current_pe_ratio,
                "current_pb_ratio": current_pb_ratio,
                "ev_ebitda": ev_ebitda,
                "sensitivity": sensitivity,
                "current_industry_ev_ebitda": current_industry_ev_ebitda,
                "current_model_ev_ebitda": current_model_ev_ebitda,
                "current_ev_diff_percent": current_ev_diff_percent,
                "cash_and_investments": cash_and_investments,
                "total_debt": total_debt


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
                "share_price_perpetuity": 6.5,
                "pe_ratio": 0,
                "pb_ratio": 0,
                "ev_ebitda": 0,
                "sensitivity": 0,
                "ev_diff_percent":0,
                "current_pe_ratio":0,
                "current_pb_ratio":0,
                "current_industry_ev_ebitda":0.0,
                "current_model_ev_ebitda":0.0,
                "current_ev_diff_percent":0.0,
                "cash_and_investments":0.0,
                "total_debt":0.0

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
            "Share Price (Perpetuity)": self.format_currency(self.variables.get("share_price_perpetuity", 0))
        }

    def get_enterprise_value_chart(self):
        """
        Displays the Enterprise Value Comparison using Plotly charts.
        """
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
                textfont=dict(color="#ffffff"),
                opacity=1.00,
                marker={
                    "color": ["#3333ff", "#0aabf5", "#0D47A1",
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
                paper_bgcolor="#000",
                plot_bgcolor="#000",
                font=dict(
                    color="#ffffff",
                    size=14,
                    family="Arial, sans-serif"
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
                "bar": {"color": "#3333ff"}
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
                "bar": {"color": "#99cc00"}
              },
              delta={"reference": ev_multiples, "relative": True},
              domain={"x": [0, 1], "y": [0, 0.45]}
            ))
            fig_gauge.update_layout(
              height=500,
              margin=dict(l=50, r=50, t=50, b=50),
              paper_bgcolor="#000",
              plot_bgcolor="#000",
              font=dict(color="#ffffff")
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

        tab1, tab2, tab3 = st.tabs(["Price Comparison", "Upside Potential", "Ownership Summary"])

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
                    paper_bgcolor="#000",
                    plot_bgcolor="#000",
                    font=dict(
                        color="#ffffff",
                        size=14,
                        family="Arial, sans-serif"
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

        # with tab2:
        #     max_upside = max(upside_multiples, upside_perpetuity)
        #     min_upside = min(upside_multiples, upside_perpetuity)
        #     fig_up = go.Figure()
        #     fig_up.add_trace(go.Indicator(
        #         mode="gauge+number+delta",
        #         value=max_upside,
        #         title={"text": "Max Upside", "font": {"size": 14}},
        #         gauge={"axis": {"range": [-50, 200]}, "bar": {"color": "#4CAF50"}},
        #         delta={"reference": 0, "relative": False},
        #         domain={"row": 0, "column": 0}
        #     ))
        #     fig_up.add_trace(go.Indicator(
        #         mode="gauge+number+delta",
        #         value=min_upside,
        #         title={"text": "Min Upside", "font": {"size": 14}},
        #         gauge={"axis": {"range": [-50, 200]}, "bar": {"color": "#FFC107"}},
        #         delta={"reference": 0, "relative": False},
        #         domain={"row": 1, "column": 0}
        #     ))
        #     fig_up.update_layout(grid={"rows": 2, "columns": 1},
        #                          height=600,
        #                          paper_bgcolor="#000",
        #                          plot_bgcolor="#000")
        #     st.plotly_chart(fig_up, use_container_width=True)
        #     # after your with tab1: and with tab2: blocks

        with tab2:
            max_upside = max(upside_multiples, upside_perpetuity)
            min_upside = min(upside_multiples, upside_perpetuity)

            fig_up = go.Figure()

            # Max Upside on left half
            fig_up.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=max_upside,
                title={"text": "Max Upside", "font": {"size": 14}},
                gauge={"axis": {"range": [-50, 200]}, "bar": {"color": "#4CAF50"}},
                delta={"reference": 0, "relative": False},
                domain={"x": [0.0, 0.5], "y": [0, 1]}
            ))

            # Min Upside on right half
            fig_up.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=min_upside,
                title={"text": "Min Upside", "font": {"size": 14}},
                gauge={"axis": {"range": [-50, 200]}, "bar": {"color": "#FFC107"}},
                delta={"reference": 0, "relative": False},
                domain={"x": [0.5, 1.0], "y": [0, 1]}
            ))

            fig_up.update_layout(
                grid={"rows": 1, "columns": 2},
                height=400,
                paper_bgcolor="#000",
                plot_bgcolor="#000"
            )

            st.plotly_chart(fig_up, use_container_width=True)




        with tab3:
            st.header("Ownership Summary")

            # Load the “Data” sheet
            raw = pd.read_excel(
            "attached_assets/EZJ.L Firm Ownership Summary.xlsx",
            sheet_name="Data",
            header=None
        )

            # 1) Top 10 Shareholders
            inv = slice_section(raw, "Investor Rank", "Type", list(range(1,8))).head(10)
            inv["% O/S"] = parse_pct(inv["% O/S"])
            fig1 = go.Figure(go.Bar(
            x=inv["Investor Name"],
            y=inv["% O/S"],
            marker_color="#1E88E5"
        ))
            fig1.update_layout(
            title="Top 10 Shareholders by % Ownership",
            xaxis_tickangle=-45,
            template="plotly_dark"
        )
            st.plotly_chart(fig1, use_container_width=True)

            # 2) Ownership % by Investor Type
            typ = slice_section(raw, "Type", "Style", [0, 1, 2])
            typ.columns = ["Type", "Investors", "% O/S"]

            # parse the percentages
            typ["% O/S"] = parse_pct(typ["% O/S"])
            # parse the investor‐count as integers
            typ["Investors"] = (
            pd.to_numeric(typ["Investors"], errors="coerce").fillna(0).astype(int)
)

            # Build legend labels: "Type Name (X investors)"
            labels = [f"{investor_type} ({count} investors)"
            for investor_type, count in zip(typ["Type"], typ["Investors"])
]


            fig2 = go.Figure(go.Pie(
            labels=labels,
            values=typ["% O/S"],
            hole=0.3,
            textinfo="label+percent",
            hoverinfo="label+value+percent",
            domain={'x': [0.0, 0.8], 'y': [0.0, 1.0]}  # <— make the pie fill 80% of the width
))

            fig2.update_layout(
            title="Ownership % by Investor Type",
            template="plotly_dark",

            # make the overall figure bigger
            width=800,
            height=600,

            # tighten margins and give space for the legend on the right
            margin=dict(l=20, r=200, t=50, b=20),

            # keep the interactive legend on the right, no ncol needed
            legend=dict(
            orientation="v",
            y=1.0,
            yanchor="top",
            x=1.05,
            xanchor="left",
            font=dict(size=12)
    )
)
            st.plotly_chart(fig2, use_container_width=True)

            # 4) Ownership % by Region
            reg = slice_section(raw, "Region", "Rotation", [0, 1, 2])
            reg.columns = ["Region", "Investors", "% O/S"]

            # convert "% O/S" into a float (e.g. "83.34%" → 0.8334)
            reg["% O/S"] = parse_pct(reg["% O/S"])

            # build the bar chart
            fig4 = go.Figure(
            go.Bar(
            x=reg["Region"],
            y=reg["% O/S"],
            marker_color="#4CAF50",
            text=reg["% O/S"].map(lambda v: f"{v:.1%}"),
            textposition="outside"
    )
)

            fig4.update_layout(
            title="Ownership % by Region",
            template="plotly_dark",
            xaxis_tickangle=-45,
            yaxis=dict(
            title="% Ownership",
            tickformat=".0%",
            range=[0, 1]    # force 0–100%
    ),
            margin=dict(l=40, r=20, t=50, b=100)  # plenty of bottom room for labels
)

            st.plotly_chart(fig4, use_container_width=True)

            # 5) Recent Trading Activity (Buys vs Sells)
            ra = raw.iloc[
            raw[0].tolist().index("Recent Activity")+2 :
            raw[0].tolist().index("Concentration"),
            [0,1,2]
            ].dropna(how='all')
            ra.columns = ["Investor","Value","Shares"]
            ra["Shares"] = pd.to_numeric(ra["Shares"], errors="coerce")
            buys = ra[ra["Value"].astype(str).str.startswith("+")]
            sells = ra[ra["Value"].astype(str).str.startswith("-")]

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
            template="plotly_dark",
            xaxis=dict(
            title='Shares (Millions)',
            tickformat=',.1fM'   # show e.g. “1.0M”, “2.5M”
    ),
            height=500
)
            st.plotly_chart(fig5, use_container_width=True)


            # 6) Holdings Concentration
            conc = slice_section(raw, "Concentration", None, [0,2])
            conc.columns = ["Concentration","% O/S"]
            conc["% O/S"] = parse_pct(conc["% O/S"])
            fig6 = go.Figure(go.Scatter(
            x=conc["Concentration"], y=conc["% O/S"],
            mode="lines+markers",
            line=dict(color="#FFC107")
        ))
            fig6.update_layout(
            title="Holdings Concentration",
            template="plotly_dark"
        )
            st.plotly_chart(fig6, use_container_width=True)



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


    def implied_by_comps(self, peer_df: pd.DataFrame) -> float:
        """
        Apply the median peer EV/EBITDA multiple to our model EBITDA
        to back out an implied share price.
        """
        # 1. Grab our model EBITDA and peer multiple
        model_ebitda = self.variables.get("current_model_ev_ebitda", 0.0)
        peer_multiple = peer_df["EV_EBITDA"].median()

        # 2. Compute implied enterprise value
        implied_ev = model_ebitda * peer_multiple

        # 3. Subtract net debt (assumes you have these in your variables)
        debt = self.variables.get("total_debt", 0.0)
        cash = self.variables.get("cash_and_equivalents", 0.0)
        implied_equity = implied_ev - (debt - cash)

        # 4. Divide by shares
        shares = self.variables.get("diluted_shares_outstanding", 1.0)
        return implied_equity / shares if shares else 0.0

    def implied_by_precedent_mna(self, mna_multiple: float) -> float:
        """
        Apply a hand‑picked M&A EV/EBITDA multiple to model EBITDA.
        """
        model_ebitda = self.variables.get("current_model_ev_ebitda", 0.0)
        implied_ev    = model_ebitda * mna_multiple

        debt = self.variables.get("total_debt", 0.0)
        cash = self.variables.get("cash_and_investments", 0.0)
        implied_equity = implied_ev - (debt - cash)

        shares = self.variables.get("diluted_shares_outstanding", 1.0)
        return implied_equity / shares if shares else 0.0
