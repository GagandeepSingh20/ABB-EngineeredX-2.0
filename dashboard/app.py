"""
Streamlit dashboard: operator-facing visualization of the digital twin.

Run with:
    pip install streamlit
    streamlit run dashboard/app.py

Displays:
  - Composite health index gauge
  - Real-time KPI tiles (hotspot, RUL, anomaly score)
  - RUL trajectory chart
  - DGA fault classification confidence
  - What-if overload simulator
  - Active alerts and maintenance schedule
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.data_generator import generate_transformer_data, generate_dga_training_set
from src.physics import ThermalModel, AgingModel, EquivalentCircuit
from src.ml_models import AnomalyDetector, DGAFaultClassifier, RULForecaster

# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="Transformer Digital Twin",
    page_icon="⚡",
    layout="wide",
)

NAVY = "#0F2A47"
RED = "#DC2626"
AMBER = "#D97706"
GREEN = "#16A34A"

# ============================================================
# Data + model loading (cached so it only runs once per session)
# ============================================================
@st.cache_data
def load_telemetry():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data",
                             "synthetic_transformer_telemetry.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return generate_transformer_data(days=180)

@st.cache_data
def load_dga_training():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data",
                             "dga_training_set.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return generate_dga_training_set(2000)

@st.cache_resource
def build_models(df_telemetry, df_dga):
    """Train all three ML models. Cached as a resource so they persist."""
    thermal = ThermalModel()
    df = df_telemetry.copy()
    df["hotspot_residual"] = thermal.residual(
        df["load_pu"], df["ambient_temp_c"], df["winding_hotspot_c"]
    )

    detector = AnomalyDetector(contamination=0.03).fit(df.iloc[:2000])
    classifier = DGAFaultClassifier().fit(df_dga)
    rul_model = RULForecaster(window_days=30).fit(df)

    return {
        "thermal": thermal,
        "aging": AgingModel(),
        "circuit": EquivalentCircuit(),
        "detector": detector,
        "classifier": classifier,
        "rul_model": rul_model,
        "df": df,
    }

# ============================================================
# Sidebar
# ============================================================
st.sidebar.title("⚡ Digital Twin")
st.sidebar.markdown("**Asset:** Transformer TR-04 / 132 kV")
st.sidebar.markdown("**Rating:** 25 MVA")
st.sidebar.markdown("**Type:** ONAF-cooled, two-winding")
st.sidebar.markdown("---")
st.sidebar.markdown("### ABB EngineeredX 2.0")
st.sidebar.markdown("Problem Statement 1: Digital Twin of Transformer / Electrical Equipment")
st.sidebar.markdown("---")
view = st.sidebar.radio("View", ["Operator dashboard", "Fault diagnostics", "What-if simulator", "About"])

# ============================================================
# Load data and models
# ============================================================
with st.spinner("Loading telemetry and training models..."):
    df_telemetry = load_telemetry()
    df_dga = load_dga_training()
    models = build_models(df_telemetry, df_dga)

df = models["df"]

# ============================================================
# Compute live state (last observation)
# ============================================================
df["anomaly_score"] = models["detector"].score(df)
df["rul_predicted_days"] = models["rul_model"].predict(df)
df["aging_factor"] = models["aging"].acceleration_factor(df["winding_hotspot_c"])

latest = df.iloc[-1]

# Composite health index: weighted combination of normalized indicators
def composite_health(row):
    score = 100
    score -= 30 * row["anomaly_score"]            # anomaly score drag
    score -= max(0, (row["winding_hotspot_c"] - 90) * 0.8)  # hot operation drag
    score -= max(0, (100 - min(row["rul_predicted_days"], 1000)) * 0.05)  # low RUL drag
    return max(0, min(100, score))

health = composite_health(latest)

# ============================================================
# View 1: Operator dashboard
# ============================================================
if view == "Operator dashboard":
    st.title("Transformer Health Dashboard")
    st.caption(f"Asset TR-04  ·  live  ·  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ---- KPI strip ----
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Composite health", f"{health:.0f} / 100",
                 delta=None if health > 70 else "below target", delta_color="inverse")
    col2.metric("Winding hotspot", f"{latest['winding_hotspot_c']:.1f}°C",
                 delta=f"{latest['hotspot_residual']:+.1f}° vs physics")
    col3.metric("RUL forecast", f"{latest['rul_predicted_days']:.0f} d")
    col4.metric("Anomaly score", f"{latest['anomaly_score']:.2f}",
                 delta="below threshold" if latest['anomaly_score'] < 0.5 else "ALERT",
                 delta_color="off" if latest['anomaly_score'] < 0.5 else "inverse")

    st.markdown("---")

    # ---- Charts ----
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.subheader("Health gauge")
        fig, ax = plt.subplots(figsize=(4, 3), subplot_kw={"projection": "polar"})

        # Arcs
        for s, e, c in [(0, 40, RED), (40, 70, AMBER), (70, 100, GREEN)]:
            t = np.linspace(np.pi - s/100*np.pi, np.pi - e/100*np.pi, 30)
            ax.plot(t, [1]*30, color=c, linewidth=18, solid_capstyle="butt", alpha=0.85)

        needle = np.pi - (health / 100) * np.pi
        ax.plot([needle, needle], [0, 0.92], color=NAVY, linewidth=3)
        ax.set_ylim(0, 1.2)
        ax.set_xlim(0, np.pi)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["polar"].set_visible(False)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(-1)
        fig.text(0.5, 0.28, f"{health:.0f}", fontsize=44, fontweight="bold",
                  color=NAVY, ha="center")
        st.pyplot(fig)

    with col_b:
        st.subheader("RUL trajectory (last 60 days)")
        recent = df.tail(60 * 24).reset_index(drop=True)
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        ax2.plot(recent.index, recent["rul_predicted_days"], color=GREEN, linewidth=1.2)
        ax2.fill_between(recent.index, 0, recent["rul_predicted_days"], color=GREEN, alpha=0.15)
        ax2.axhline(90, color=RED, linestyle="--", linewidth=1)
        ax2.set_xlabel("Hour")
        ax2.set_ylabel("RUL (days)")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        st.pyplot(fig2)

    st.markdown("---")

    st.subheader("Anomaly score (last 30 days)")
    last_month = df.tail(30 * 24).reset_index(drop=True)
    fig3, ax3 = plt.subplots(figsize=(12, 3))
    ax3.plot(last_month.index, last_month["anomaly_score"], color="#2563EB", linewidth=0.7)
    ax3.fill_between(last_month.index, 0, last_month["anomaly_score"], color="#2563EB", alpha=0.15)
    ax3.axhline(0.5, color=RED, linestyle="--", linewidth=1)
    ax3.set_xlabel("Hour")
    ax3.set_ylabel("Anomaly score")
    ax3.set_ylim(0, 1)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    st.pyplot(fig3)

    st.markdown("---")
    st.subheader("Active alerts & recommendations")

    alerts = []
    if latest["anomaly_score"] > 0.5:
        alerts.append(("🟠", f"Anomaly score {latest['anomaly_score']:.2f} exceeds threshold — investigate"))
    if latest["hotspot_residual"] > 5:
        alerts.append(("🟡", f"Hotspot running {latest['hotspot_residual']:.1f}°C above physics prediction — check cooling"))
    if latest["rul_predicted_days"] < 180:
        alerts.append(("🔴", f"RUL forecast {latest['rul_predicted_days']:.0f} d — schedule maintenance window"))
    if not alerts:
        alerts.append(("🟢", "All indicators within normal range — no action required"))

    for icon, msg in alerts:
        st.markdown(f"{icon} &nbsp; {msg}")

# ============================================================
# View 2: Fault diagnostics
# ============================================================
elif view == "Fault diagnostics":
    st.title("DGA Fault Diagnostics")
    st.caption("Enter dissolved gas concentrations to classify the fault type")

    col1, col2 = st.columns(2)
    with col1:
        h2 = st.number_input("H₂ (ppm)", 0.0, 5000.0, float(latest["h2_ppm"]), step=10.0)
        ch4 = st.number_input("CH₄ (ppm)", 0.0, 5000.0, float(latest["ch4_ppm"]), step=10.0)
        c2h2 = st.number_input("C₂H₂ (ppm)", 0.0, 5000.0, float(latest["c2h2_ppm"]), step=5.0)
        c2h4 = st.number_input("C₂H₄ (ppm)", 0.0, 5000.0, float(latest["c2h4_ppm"]), step=10.0)
    with col2:
        c2h6 = st.number_input("C₂H₆ (ppm)", 0.0, 5000.0, float(latest["c2h6_ppm"]), step=10.0)
        co = st.number_input("CO (ppm)", 0.0, 10000.0, float(latest["co_ppm"]), step=20.0)
        co2 = st.number_input("CO₂ (ppm)", 0.0, 20000.0, float(latest["co2_ppm"]), step=50.0)

    sample = pd.DataFrame([{
        "h2_ppm": h2, "ch4_ppm": ch4, "c2h2_ppm": c2h2, "c2h4_ppm": c2h4,
        "c2h6_ppm": c2h6, "co_ppm": co, "co2_ppm": co2
    }])

    probs = models["classifier"].predict_proba(sample)[0]
    pred = models["classifier"].predict(sample)[0]
    names = models["classifier"].CLASS_NAMES

    st.markdown(f"### Predicted fault: **{names[pred]}**  ·  confidence {probs[pred]:.1%}")

    # Bar chart of probabilities
    fig, ax = plt.subplots(figsize=(10, 3))
    colors = [AMBER if i == pred else "#E2E8F0" for i in range(len(names))]
    bars = ax.barh(names, probs * 100, color=colors, edgecolor=NAVY, linewidth=0.4)
    for i, (b, p) in enumerate(zip(bars, probs)):
        ax.text(p * 100 + 1, b.get_y() + b.get_height() / 2,
                 f"{p*100:.1f}%", va="center", fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, max(80, probs.max() * 110))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    st.pyplot(fig)

# ============================================================
# View 3: What-if simulator
# ============================================================
elif view == "What-if simulator":
    st.title("What-if Overload Simulator")
    st.caption("Simulate sustained overload and see projected hotspot + life consumption")

    col1, col2, col3 = st.columns(3)
    load = col1.slider("Load (p.u.)", 0.5, 1.5, 1.2, 0.05)
    duration = col2.slider("Duration (hours)", 0.5, 24.0, 4.0, 0.5)
    ambient = col3.slider("Ambient temp (°C)", 0, 50, 30)

    scenario = models["circuit"].simulate_overload(
        load_pu=load, duration_hours=duration, ambient_c=ambient,
        thermal=models["thermal"], aging=models["aging"],
    )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted hotspot", f"{scenario['predicted_hotspot_c']:.1f}°C",
                 delta="exceeds design limit" if scenario['exceeds_design_limit'] else "within limits",
                 delta_color="inverse" if scenario['exceeds_design_limit'] else "off")
    col2.metric("Aging factor F_AA", f"{scenario['aging_acceleration_factor']:.2f}x")
    col3.metric("Life hours consumed", f"{scenario['equivalent_life_hours_consumed']:.1f} h")
    col4.metric("Real hours", f"{scenario['duration_hours']:.1f} h")

    if scenario['exceeds_design_limit']:
        st.error("⚠️ This overload exceeds the IEEE design hotspot limit of 140°C. Operating in this regime risks accelerated insulation failure.")
    elif scenario['aging_acceleration_factor'] > 5:
        st.warning(f"This scenario consumes life at {scenario['aging_acceleration_factor']:.1f}× the reference rate. Use only for emergency loading.")
    else:
        st.success(f"Operation within normal parameters. {scenario['equivalent_life_hours_consumed']:.1f} hours of design life consumed.")

# ============================================================
# View 4: About
# ============================================================
else:
    st.title("About this digital twin")
    st.markdown("""
This is a proof-of-concept digital twin for transformer asset health monitoring,
built as part of an **ABB EngineeredX 2.0** submission (Problem Statement 1).

### Architecture

The system has four tiers:

1. **Physical asset layer** — transformer with IoT sensors (DGA, temperature, vibration, partial discharge)
2. **Edge gateway & data pipeline** — MQTT/OPC-UA → InfluxDB time-series store
3. **Digital twin core** — physics models (IEEE C57.91, Arrhenius) + ML layer (Isolation Forest, XGBoost, LSTM)
4. **Applications & dashboard** — this Streamlit interface

### The hybrid loop

The key idea: the physics model predicts what the transformer *should* be doing
under current loading; the ML model interprets the *deviation* between
expectation and reality. Pure physics misses unmodeled degradation; pure ML
demands prohibitive training data. Together they outperform either.

### Implementation status

- ✅ Synthetic data generator (180 days, 21 sensor channels)
- ✅ IEEE C57.91 thermal model
- ✅ Arrhenius aging model
- ✅ Anomaly detection (Isolation Forest + physics residuals)
- ✅ DGA fault classification (XGBoost, 6 classes)
- ✅ RUL forecasting (MLP regression — production design specifies LSTM)
- ✅ Operator dashboard (this app)

### Standards referenced

- IEEE C57.91 — Loading guide for mineral-oil-immersed transformers
- IEC 60599 — Interpretation of dissolved gas analysis
- Duval Triangle method — Graphical fault typing from DGA

### Caveats

This is a proof-of-concept on synthetic data. Real deployment would require
calibration against the specific asset's heat-run test data, historical
operating records, and training on real fault histories from an asset fleet.
""")
