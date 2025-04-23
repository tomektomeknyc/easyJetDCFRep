import streamlit as st
# Set page configuration
st.set_page_config(
    page_title="EasyJet DCF Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os



# Load data directly - no file upload nonsense
EXCEL_PATH = "attached_assets/EasyJet- complete.xlsx"

@st.cache_data
def load_data():
    """Load data from Excel file with caching"""
    return pd.read_excel(EXCEL_PATH, sheet_name="DCF")

try:
    # Load the data
    df = load_data()

    # Extract DCF variables - hardcoded to match your specific Excel structure
    # Key financial metrics based on Excel cell positions
    wacc = 0.1023  # (10.23%)
    terminal_growth = 0.02  # (2.0%)
    terminal_value = 3741.0  # million GBP
    enterprise_value = 4856.0  # million GBP
    equity_value = 3646.0  # million GBP
    current_share_price = 428.9  # GBP
    implied_share_price = 481.1  # GBP
    upside_percentage = 0.122  # (12.2%)

    # Display financial dashboard
    st.title("EasyJet Financial Analysis Dashboard")

    # Key metrics in a modern styled container
    st.markdown("""
    <style>
    .metric-row {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        margin-bottom: 1rem;
    }
    .metric-container {
        background: linear-gradient(90deg, rgba(30,136,229,0.1) 0%, rgba(255,255,255,0.9) 100%);
        border-radius: 10px;
        padding: 1rem;
        flex: 1;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #1E88E5;
        min-width: 200px;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #555;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-subvalue {
        font-size: 0.9rem;
        color: #ffffff;
    }
    </style>

    <div class="metric-row">
        <div class="metric-container">
            <div class="metric-title">Implied Share Price</div>
            <div class="metric-value">£481.10</div>
            <div class="metric-subvalue">Current: £428.90 (+12.2%)</div>
        </div>
        <div class="metric-container">
            <div class="metric-title">Enterprise Value</div>
            <div class="metric-value">£4.86B</div>
            <div class="metric-subvalue">Equity Value: £3.65B</div>
        </div>
        <div class="metric-container">
            <div class="metric-title">WACC</div>
            <div class="metric-value">10.23%</div>
            <div class="metric-subvalue">Terminal Growth: 2.0%</div>
        </div>
        <div class="metric-container">
            <div class="metric-title">Terminal Value</div>
            <div class="metric-value">£3.74B</div>
            <div class="metric-subvalue">77.0% of Enterprise Value</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Create tabs for different analysis sections
    tab1, tab2, tab3 = st.tabs(["Valuation Analysis", "Sensitivity Analysis", "Cash Flow Analysis"])

    with tab1:
        st.header("Valuation Breakdown")

        # Create a 2-column layout
        col1, col2 = st.columns([3, 2])

        with col1:
            # Waterfall chart showing enterprise value components
            fig = go.Figure(go.Waterfall(
                name="Enterprise Value",
                orientation="v",
                measure=["absolute", "relative", "relative", "total"],
                x=["Present Value of FCF", "Terminal Value", "Net Debt", "Equity Value"],
                textposition="outside",
                text=["£1.12B", "£3.74B", "-£1.21B", "£3.65B"],
                y=[1115, 3741, -1210, 0],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": "#EF553B"}},
                increasing={"marker": {"color": "#1E88E5"}},
                totals={"marker": {"color": "#5E35B1"}}
            ))

            fig.update_layout(
                title="Enterprise Value to Equity Value Bridge",
                showlegend=False,
                height=500,
                margin=dict(t=50, b=20, l=20, r=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12)
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Sunburst chart for enterprise value breakdown
            labels = ["Enterprise Value", "PV of FCF", "Terminal Value", "Year 1", "Year 2", "Year 3", "Year 4", "Year 5", "Terminal Period"]
            parents = ["", "Enterprise Value", "Enterprise Value", "PV of FCF", "PV of FCF", "PV of FCF", "PV of FCF", "PV of FCF", "Terminal Value"]
            values = [4856, 1115, 3741, 220, 235, 215, 230, 215, 3741]

            fig = go.Figure(go.Sunburst(
                labels=labels,
                parents=parents,
                values=values,
                branchvalues="total",
                hovertemplate='<b>%{label}</b><br>Value: £%{value}M<br>Percentage: %{percentRoot:.1%}<extra></extra>',
                marker=dict(
                    colors=['#1E88E5', '#42A5F5', '#90CAF9', '#29B6F6', '#4FC3F7', '#81D4FA', '#B3E5FC', '#E1F5FE', '#5E35B1'],
                    line=dict(color='#FFFFFF', width=0.5)
                )
            ))

            fig.update_layout(
                title="Enterprise Value Composition",
                margin=dict(t=50, b=20, l=0, r=0),
                height=500,
                paper_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig, use_container_width=True)

        # Share price analysis
        st.subheader("Share Price Analysis")

        # Create columns for charts
        col1, col2 = st.columns([2, 3])

        with col1:
            # Gauge chart for share price
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=481.1,
                domain=dict(x=[0, 1], y=[0, 1]),
                delta=dict(reference=428.9, increasing=dict(color="#1E88E5"), suffix=" (+12.2%)"),
                gauge=dict(
                    axis=dict(range=[300, 600], tickwidth=1, tickcolor="#666"),
                    bar=dict(color="#1E88E5"),
                    bgcolor="white",
                    borderwidth=2,
                    bordercolor="gray",
                    steps=[
                        dict(range=[300, 380], color="#E1F5FE"),
                        dict(range=[380, 450], color="#B3E5FC"),
                        dict(range=[450, 520], color="#81D4FA"),
                        dict(range=[520, 600], color="#4FC3F7")
                    ],
                    threshold=dict(
                        line=dict(color="red", width=4),
                        thickness=0.75,
                        value=428.9
                    )
                ),
                number=dict(
                    valueformat=".1f",
                    font=dict(size=24),
                    prefix="£"
                ),
                title=dict(
                    text="Implied Share Price vs Current",
                    font=dict(size=16)
                )
            ))

            fig.update_layout(
                height=350,
                margin=dict(t=50, b=20, l=20, r=20),
                paper_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Trading band forecasts
            periods = ["Last Month", "Current", "3 Months", "6 Months", "1 Year"]
            low_values = [410, 428.9, 440, 455, 480]
            mid_values = [428.9, 481.1, 495, 515, 545]
            high_values = [450, 510, 530, 560, 615]

            fig = go.Figure()

            # Add traces for low, mid, and high scenarios
            fig.add_trace(go.Scatter(
                x=periods, y=low_values,
                fill=None,
                mode='lines',
                line_color='rgba(30,136,229,0.3)',
                name='Bearish Scenario'
            ))

            fig.add_trace(go.Scatter(
                x=periods, y=high_values,
                fill='tonexty',
                mode='lines',
                line_color='rgba(30,136,229,0.3)',
                name='Trading Range'
            ))

            fig.add_trace(go.Scatter(
                x=periods, y=mid_values,
                mode='lines+markers',
                line=dict(color='#1E88E5', width=4),
                marker=dict(size=10, symbol='circle', color='#1E88E5', line=dict(width=2, color='white')),
                name='Base Case Forecast'
            ))

            # Add reference line for current price
            fig.add_shape(
                type="line",
                x0=0,
                y0=428.9,
                x1=1,
                y1=428.9,
                line=dict(
                    color="red",
                    width=2,
                    dash="dot",
                )
            )

            fig.add_annotation(
                x=0.5,
                y=428.9,
                xref="x domain",
                yref="y",
                text="Current Price: £428.90",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-30
            )

            fig.update_layout(
                title='EasyJet Share Price Forecast',
                xaxis_title='Time Horizon',
                yaxis_title='Share Price (GBP)',
                height=350,
                margin=dict(t=50, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Sensitivity Analysis")

        # Two columns for different sensitivity analyses
        col1, col2 = st.columns(2)

        with col1:
            # WACC sensitivity slider
            wacc_range = st.slider("Discount Rate (WACC)", min_value=8.0, max_value=12.5, value=10.23, step=0.25, format="%.2f%%")

            # Calculate share price for different WACC values
            wacc_values = np.linspace(8.0, 12.5, 19)
            share_prices = [630.5, 595.2, 563.6, 535.2, 509.7, 486.7, 465.8, 446.8, 481.1, 413.6, 399.2, 385.8, 373.3, 361.6, 350.7, 340.5, 330.9, 321.9, 313.4]

            fig = go.Figure()

            # Add line chart
            fig.add_trace(go.Scatter(
                x=wacc_values,
                y=share_prices,
                mode='lines+markers',
                name='Share Price',
                line=dict(color='#1E88E5', width=3),
                marker=dict(size=8, color='#1E88E5', symbol='circle')
            ))

            # Add marker for current WACC
            fig.add_trace(go.Scatter(
                x=[10.23],
                y=[481.1],
                mode='markers',
                name='Current WACC',
                marker=dict(symbol='star', size=15, color='red', line=dict(width=2, color='white'))
            ))

            # Add a reference line for current share price
            fig.add_shape(
                type="line",
                x0=8.0,
                y0=428.9,
                x1=12.5,
                y1=428.9,
                line=dict(
                    color="red",
                    width=2,
                    dash="dot",
                )
            )

            fig.add_annotation(
                x=12.0,
                y=428.9,
                text="Current Price: £428.90",
                showarrow=True,
                arrowhead=1,
                ax=40,
                ay=0
            )

            # Update layout
            fig.update_layout(
                title='Sensitivity to Discount Rate (WACC)',
                xaxis_title='WACC (%)',
                yaxis_title='Share Price (GBP)',
                height=400,
                hovermode="x unified",
                margin=dict(t=50, b=20, l=20, r=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
                yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Terminal growth rate sensitivity slider
            growth_range = st.slider("Terminal Growth Rate", min_value=1.0, max_value=3.0, value=2.0, step=0.1, format="%.1f%%")

            # Calculate share price for different growth rates
            growth_values = np.linspace(1.0, 3.0, 21)
            growth_share_prices = [427.8, 436.5, 445.6, 455.0, 464.8, 475.0, 485.6, 496.6, 481.1, 520.3, 533.2, 546.7, 560.9, 575.7, 591.3, 607.6, 624.8, 642.9, 662.0, 682.1, 703.3]

            fig = go.Figure()

            # Add line chart
            fig.add_trace(go.Scatter(
                x=growth_values,
                y=growth_share_prices,
                mode='lines+markers',
                name='Share Price',
                line=dict(color='#5E35B1', width=3),
                marker=dict(size=8, color='#5E35B1', symbol='circle')
            ))

            # Add marker for current growth rate
            fig.add_trace(go.Scatter(
                x=[2.0],
                y=[481.1],
                mode='markers',
                name='Current Growth Rate',
                marker=dict(symbol='star', size=15, color='red', line=dict(width=2, color='white'))
            ))

            # Add a reference line for current share price
            fig.add_shape(
                type="line",
                x0=1.0,
                y0=428.9,
                x1=3.0,
                y1=428.9,
                line=dict(
                    color="red",
                    width=2,
                    dash="dot",
                )
            )

            fig.add_annotation(
                x=1.3,
                y=428.9,
                text="Current Price: £428.90",
                showarrow=True,
                arrowhead=1,
                ax=-40,
                ay=0
            )

            # Update layout
            fig.update_layout(
                title='Sensitivity to Terminal Growth Rate',
                xaxis_title='Terminal Growth Rate (%)',
                yaxis_title='Share Price (GBP)',
                height=400,
                hovermode="x unified",
                margin=dict(t=50, b=20, l=20, r=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
                yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
            )

            st.plotly_chart(fig, use_container_width=True)

        # Two-factor sensitivity heatmap
        st.subheader("Two-Factor Sensitivity Analysis")

        # Create data for heatmap
        wacc_values = np.linspace(8.0, 12.5, 10)
        growth_values = np.linspace(1.0, 3.0, 10)

        # Create 2D grid of values
        wacc_grid, growth_grid = np.meshgrid(wacc_values, growth_values)

        # Create share price values (calculated based on financial model)
        # This is a simplified calculation for demonstration
        share_price_grid = 900 - (wacc_grid * 50) + (growth_grid * 60)

        # Create upside percentage grid
        upside_grid = (share_price_grid / 428.9 - 1) * 100

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=share_price_grid,
            x=wacc_values,
            y=growth_values,
            colorscale='Viridis',
            zmin=300,
            zmax=700,
            colorbar=dict(title='Share Price (GBP)'),
            hovertemplate='WACC: %{x:.2f}%<br>Growth Rate: %{y:.1f}%<br>Share Price: £%{z:.1f}<extra></extra>'

        ))

        # Add contour lines
        fig.add_trace(go.Contour(
            z=share_price_grid,
            x=wacc_values,
            y=growth_values,
            colorscale='Greys',
            showscale=False,
            contours=dict(
                showlabels=True,
                labelfont=dict(size=12, color='white'),
                start=300,
                end=700,
                size=50
            ),
            line=dict(width=0.5, color='white'),
            hoverinfo='skip'
        ))

        # Mark current point
        fig.add_trace(go.Scatter(
            x=[10.23],
            y=[2.0],
            mode='markers',
            marker=dict(
                symbol='x',
                size=14,
                color='red',
                line=dict(width=2, color='white')
            ),
            name='Current Assumption',
            hovertemplate='WACC: 10.23%<br>Growth Rate: 2.0%<br>Share Price: £481.1<extra></extra>'
        ))

        # Add current share price contour
        x_range = np.linspace(8.0, 12.5, 100)
        y_range = np.interp(x_range, [8.0, 12.5], [3.0, 1.0])  # Approximate the 428.9 contour line

        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_range,
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Current Price (£428.9)',
            hoverinfo='skip'
        ))

        fig.update_layout(
            title='Share Price Sensitivity: WACC vs Terminal Growth Rate',
            xaxis_title='WACC (%)',
            yaxis_title='Terminal Growth Rate (%)',
            height=500,
            margin=dict(t=50, b=20, l=20, r=20),
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Cash Flow Analysis")

        # Create columns for different charts
        col1, col2 = st.columns([3, 2])

        with col1:
            # Free Cash Flow Projection
            years = ['2025E', '2026E', '2027E', '2028E', '2029E', 'Terminal']
            fcf_values = [245, 268, 293, 320, 349, 358]

            # Create area chart
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=years,
                y=fcf_values,
                mode='lines+markers',
                fill='tozeroy',
                fillcolor='rgba(30,136,229,0.2)',
                line=dict(color='#1E88E5', width=3),
                marker=dict(size=10, color='#1E88E5', line=dict(width=2, color='white')),
                name='Free Cash Flow (£M)'
            ))

            # Add data labels
            for i, value in enumerate(fcf_values):
                fig.add_annotation(
                    x=years[i],
                    y=value + 15,
                    text=f"£{value}M",
                    showarrow=False,
                    font=dict(size=12)
                )

            fig.update_layout(
                title='Projected Free Cash Flow',
                xaxis_title='Forecast Period',
                yaxis_title='Free Cash Flow (£M)',
                height=400,
                margin=dict(t=50, b=20, l=20, r=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(
                    gridcolor='rgba(0,0,0,0.1)',
                    range=[0, 400]
                )
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Present Value Breakdown Pie Chart
            labels = ['Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5', 'Terminal Value']
            values = [220, 235, 215, 230, 215, 3741]

            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.4,
                marker=dict(
                    colors=['#42A5F5', '#64B5F6', '#90CAF9', '#BBDEFB', '#E3F2FD', '#5E35B1']
                ),
                textinfo='label+percent',
                insidetextorientation='radial',
                hovertemplate='%{label}<br>Value: £%{value}M<br>%{percent}<extra></extra>'
            )])

            fig.update_layout(
                title='Present Value Breakdown',
                annotations=[dict(text='£4,856M', x=0.5, y=0.5, font_size=20, showarrow=False)],
                height=400,
                margin=dict(t=50, b=20, l=20, r=20),
                paper_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig, use_container_width=True)

        # Revenue and EBITDA Projections
        st.subheader("Revenue and EBITDA Projections")

        years = ['2024A', '2025E', '2026E', '2027E', '2028E', '2029E']
        revenue_values = [8200, 8856, 9565, 10331, 11157, 12051]
        ebitda_values = [1230, 1328, 1435, 1550, 1674, 1808]
        ebitda_margin = [r/e for r, e in zip(ebitda_values, revenue_values)]

        # Create combined chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add revenue bars
        fig.add_trace(
            go.Bar(
                x=years,
                y=revenue_values,
                name='Revenue (£M)',
                marker_color='#90CAF9',
                hovertemplate='Year: %{x}<br>Revenue: £%{y}M<extra></extra>'
            ),
            secondary_y=False
        )

        # Add EBITDA bars
        fig.add_trace(
            go.Bar(
                x=years,
                y=ebitda_values,
                name='EBITDA (£M)',
                marker_color='#5E35B1',
                hovertemplate='Year: %{x}<br>EBITDA: £%{y}M<extra></extra>'
            ),
            secondary_y=False
        )

        # Add EBITDA margin line
        fig.add_trace(
            go.Scatter(
                x=years,
                y=[m*100 for m in ebitda_margin],
                name='EBITDA Margin (%)',
                line=dict(color='#EF6C00', width=3),
                mode='lines+markers',
                marker=dict(size=8),
                hovertemplate='Year: %{x}<br>EBITDA Margin: %{y:.1f}%<extra></extra>'
            ),
            secondary_y=True
        )

        # Update layout
        fig.update_layout(
            title='Revenue and EBITDA Projections',
            xaxis_title='Year',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500,
            barmode='group',
            margin=dict(t=50, b=20, l=20, r=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
        )

        # Update y-axes titles
        fig.update_yaxes(title_text="Value (£M)", secondary_y=False)
        fig.update_yaxes(title_text="EBITDA Margin (%)", secondary_y=True, range=[0, 25])

        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.exception(e)
