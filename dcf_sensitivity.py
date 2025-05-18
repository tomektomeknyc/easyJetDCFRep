import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def run_dcf_sensitivity(
    path: str,
    growth_rate: float,
    discount_rate: float,
    epsilon: float = 1e-6
):

    """
    Reads free cash flow data from an Excel file, computes a DCF sensitivity surface,
    and returns an interactive Plotly Figure.

    Parameters:
    - excel_path: Path to the EasyJet- complete.xlsx file.

    Returns:
    - A plotly.graph_objects.Figure containing the 3D surface.
    """
    # 1) Read the “DCF” sheet, cells E122:Q122
    df = pd.read_excel(
        path,
        sheet_name="DCF",
        header=None,
        usecols="E:Q",
        skiprows=121,
        nrows=1
    )
    if df.empty:
        raise ValueError("No data in E122:Q122 on sheet 'DCF'")
    base_fcf = float(df.iloc[0, 0])

    # 2) Build the grid of growth & discount rates
    growth = np.arange(0.0, 0.102, 0.002)
    discount = np.arange(0.04, 0.152, 0.002)
    G, R = np.meshgrid(growth, discount)

    # 3) Compute NPV surface
    num_years = 5
    years = np.arange(1, num_years + 1)
    NPV = np.zeros_like(G)
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            g = G[i, j]
            r = R[i, j]
            proj = base_fcf * (1 + g) ** years
            TV = proj[-1] * (1 + g) / (r - g)
            CFs = np.append(proj, TV)
            disc = (1 + r) ** np.append(years, num_years)
            NPV[i, j] = np.sum(CFs / disc)
    # 4) Scale into millions
            NPV_m = NPV / 1e6


    # 5) Create Plotly surface and flip it upright by negating Z
    fig = go.Figure(data=[
        go.Surface(
            x=G,
            y=R,
            z=-NPV_m,                  # flip the surface so positive values go up
            colorscale='Viridis',
            colorbar=dict(title="NPV (£m)")
        )
    ])
    fig.update_layout(
        title="DCF Sensitivity Surface (5 years)",
        scene=dict(
            xaxis_title="Growth Rate (g)",
            yaxis_title="Discount Rate (r)",
            zaxis_title="NPV (£m)",
            camera=dict(             # optional: adjust camera angle
                eye=dict(x=1.5, y=-1.5, z=0.5)
            )
        ),
        width=800,
        height=600
    )

    return fig
