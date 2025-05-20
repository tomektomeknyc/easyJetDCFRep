import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def show_regression_surface():
    st.subheader("📊 EZJ Share Price Regression Surface (£) — Historical Fit")
    st.markdown("Use the sliders below to explore how changes in FTSE100 and Jet Fuel Swap Index impact EZJ share price based on historical regression.")

    # Load data from the Excel regression file
    EXCEL_PATH = "attached_assets/EZJFuelIndexReg.xlsx"

    # Manually input regression coefficients from Refinitiv
    intercept = 1364.9764
    coef_ftse = -0.007116
    coef_fuel = -5.3111

    # Embedded controls inside Regression tab
    with st.container():
        st.markdown("### Regression Simulation Controls")
        ftse_min = st.slider("FTSE100 Min", 6000, 7500, 6500)
        ftse_max = st.slider("FTSE100 Max", 7500, 8500, 8000)
        fuel_min = st.slider("Jet Fuel Index Min", 20, 80, 30)
        fuel_max = st.slider("Jet Fuel Index Max", 80, 130, 100)

    # Generate grid
    x_ftse = np.linspace(ftse_min, ftse_max, 50)
    y_fuel = np.linspace(fuel_min, fuel_max, 50)
    X, Y = np.meshgrid(x_ftse, y_fuel)

    # Predict EZJ price from regression equation
    Z = intercept + coef_ftse * X + coef_fuel * Y

    # Plot
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z,
        colorscale="Electric",
        colorbar=dict(title="EZJ Price £"),
        hovertemplate="FTSE100: %{x}<br>Fuel: %{y}<br>EZJ: £%{z:.2f}<extra></extra>"
    )])

    fig.update_layout(
        title="EZJ Regression Surface: FTSE100 vs Jet Fuel Impact",
        scene=dict(
            xaxis_title="FTSE100 Index",
            yaxis_title="Jet Fuel Swap Index",
            zaxis_title="EZJ Price (£)",
            xaxis=dict(backgroundcolor="black", gridcolor="gray", color="white"),
            yaxis=dict(backgroundcolor="black", gridcolor="gray", color="white"),
            zaxis=dict(backgroundcolor="black", gridcolor="gray", color="white")
        ),
        height=700,
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        margin=dict(l=10, r=10, t=50, b=10),
        scene_camera=dict(eye=dict(x=1.7, y=-2.2, z=1.0))
    )

    st.plotly_chart(fig, use_container_width=True)

    # Style for tables
    st.markdown("""
        <style>
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th {
                text-align: center;
                padding: 8px;
                background-color: #f2f2f2;
            }
            td {
                padding: 8px;
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)

    # Summary Statistics Table
    st.subheader("🔢 Regression Summary Statistics")
    reg_summary = pd.DataFrame({
        "Metric": [
            "No. of observations", "Degrees of freedom", "Adjusted R-squared",
            "F-statistic", "Residual standard deviation", "Durbin-Watson"
        ],
        "Value": [121, 118, 0.1647, 12.8276, 318.6935, 0.1632]
    })
    st.dataframe(reg_summary, use_container_width=True)

    # Coefficients Table
    st.subheader("🔗 Coefficients and Correlations")
    reg_coefs = pd.DataFrame({
    "Variable": ["Intercept", "FTSE100", "Jet Fuel"],
    "Coefficient": [1364.9764, -0.0176, -5.3110],
    "T-value": [3.9483, -0.3114, -3.6651],
    "Correlation": ["N/A", "-0.2917", "-0.4218"]  # ✅ all as strings
})

    st.dataframe(reg_coefs, use_container_width=True)

    # Interpretation
    st.subheader("🧠 Interpretation")
    st.markdown("""
        The regression model shows a **weak fit** with an adjusted R-squared of **0.1647**,
        meaning that about 16.5% of the variation in EZJ’s stock price can be explained by
        FTSE100 and Jet Fuel Swap prices. The impact of jet fuel appears more statistically
        significant, while FTSE100 shows a weak and possibly non-significant effect.
        This surface is for exploratory macroeconomic sensitivity analysis only.
    """)
