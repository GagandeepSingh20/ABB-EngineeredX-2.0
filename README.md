# ABB-EngineeredX-2.0
# Transformer Digital Twin

> **Submission for ABB EngineeredX 2.0** · Problem Statement 1: Digital Twin of Transformer / Electrical Equipment

A working proof-of-concept digital twin that combines physics-based modeling
with machine learning to monitor power transformer health, detect anomalies,
classify dissolved-gas faults, and forecast remaining useful life.

---

## 🎯 The core idea — the hybrid loop

```
                ┌────────────────────────────────┐
                │     Digital twin core           │
                │                                 │
   Physical ──> │   Physics model ───┐            │
   sensors      │                    ├── ML layer │ ──> Operator
                │   (IEEE C57.91,    │            │     dashboard
                │   Arrhenius)       └── residual │
                │                                 │
                └────────────────────────────────┘
```

The physics model predicts what the transformer *should* be doing under current
loading and ambient conditions. The ML layer interprets the **residual** —
the gap between expected and observed — to flag faults and forecast life.

- Pure physics misses unmodeled degradation modes.
- Pure ML demands prohibitive amounts of training data.
- **The combination is how industrial-grade digital twins actually work.**

---

## 📁 Repository structure

```
transformer-digital-twin/
├── README.md                       ← you are here
├── requirements.txt
├── LICENSE
├── docs/
│   └── ABB_EngineeredX_PS1_DigitalTwin.pdf   ← the submission document
├── data/
│   ├── synthetic_transformer_telemetry.csv   ← 180 days, 21 channels, 4 320 rows
│   └── dga_training_set.csv                  ← 2 000 labeled DGA samples
├── src/
│   ├── data_generator.py           ← synthetic telemetry & DGA samples
│   ├── physics.py                  ← IEEE C57.91, Arrhenius, equivalent circuit
│   ├── ml_models.py                ← AnomalyDetector, DGAFaultClassifier, RULForecaster
│   └── visualization.py            ← reusable plotting helpers
├── notebooks/
│   ├── 01_data_generation.ipynb    ← walk through the synthetic data
│   ├── 02_physics_model.ipynb      ← thermal + aging models with validation
│   ├── 03_anomaly_detection.ipynb  ← Isolation Forest + hybrid-loop ablation
│   ├── 04_dga_fault_classifier.ipynb  ← XGBoost on Duval Triangle features
│   └── 05_rul_forecasting.ipynb    ← Remaining useful life predictions
└── dashboard/
    └── app.py                      ← Streamlit operator dashboard
```

---

## 🚀 Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/<YOUR_USERNAME>/transformer-digital-twin.git
cd transformer-digital-twin

# 2. Install dependencies (Python 3.10+ recommended)
pip install -r requirements.txt

# 3. Walk through the notebooks in order
jupyter notebook notebooks/

# 4. Launch the operator dashboard
streamlit run dashboard/app.py
```

The dashboard opens in your browser at `http://localhost:8501`.

---

## 🔬 What's implemented

### Physics layer (`src/physics.py`)

| Component | Standard | Inputs | Outputs |
|---|---|---|---|
| `ThermalModel` | IEEE C57.91 | load, ambient temp, cooling class | predicted hotspot, residual |
| `AgingModel` | Arrhenius / Montsinger | hotspot history | life consumed, F_AA |
| `EquivalentCircuit` | Standard T-model | V/I, impedance | losses, what-if simulations |

**Verified against textbook values:** F_AA at 117°C equals 2.02 (Montsinger's rule
says ageing should double every 7°C above the 110°C reference — ✓).

### ML layer (`src/ml_models.py`)

| Model | Algorithm | Input | Output |
|---|---|---|---|
| `AnomalyDetector` | Isolation Forest (200 trees) on sensors + physics residuals | 11-dim vector | anomaly score ∈ [0,1] |
| `DGAFaultClassifier` | XGBoost (6-class) on gas + Duval Triangle features | 13 features | fault type + confidence |
| `RULForecaster` | MLP regression on rolling window statistics | 18-feature window | days to maintenance |

**Production note:** The submission PDF specifies an LSTM with attention for
RUL forecasting. PyTorch didn't fit in the build environment used to assemble
this repo, so the demo uses scikit-learn's `MLPRegressor` on engineered window
features. The pipeline and interface are identical; swap `MLPRegressor` for
an LSTM in production:

```python
# Demo (this repo):
self.model = MLPRegressor(hidden_layer_sizes=(64, 32), ...)

# Production:
self.model = LSTMWithAttention(input_dim=18, hidden_dim=128, n_heads=4)
```

### Operator dashboard (`dashboard/app.py`)

Streamlit app with four views:

1. **Operator dashboard** — composite health gauge, KPI tiles, RUL trajectory, alerts
2. **Fault diagnostics** — interactive DGA fault classification with confidence
3. **What-if simulator** — overload scenario projection (hotspot + life consumed)
4. **About** — architecture and standards summary

---

## 📊 Validation results

Run notebook 03 to reproduce. Summary metrics on synthetic data:

| Metric | Value |
|---|---|
| Anomaly detector score during faults | 0.63 (vs 0.20 normal) |
| Anomaly detector precision @ 0.5 threshold | ~0.85 |
| DGA classifier test accuracy | 1.00 (synthetic) / typical 0.85–0.92 on real data |
| Thermal model fit RMSE (after recalibration) | < 1.5°C |

**Caveat:** Test accuracy of 1.00 reflects well-separated synthetic data, not
realistic real-world performance. On real DGA fault libraries (e.g. the IEC TC10
database), gradient-boosted classifiers typically achieve 0.85–0.92 accuracy.

---

## 🏗️ How this extends to production

This is a proof-of-concept. The path to a real industrial deployment looks like:

| Stage | Demo (this repo) | Production |
|---|---|---|
| **Data ingestion** | CSV file read | MQTT/OPC-UA streams via edge gateway, InfluxDB sink |
| **Physics parameters** | textbook defaults | calibrated against unit's heat-run test report |
| **Training data** | synthetic, 180 days | real fleet history, multi-year, with maintenance labels |
| **RUL model** | MLPRegressor | LSTM with attention or Temporal Fusion Transformer |
| **Deployment** | local Streamlit | Docker + Kubernetes on Azure Digital Twins (ABB Ability™ stack) |
| **Anomaly detection** | batch-trained | online, edge-deployed, with drift detection |

The architecture also **generalizes beyond transformers** — swap the physics
module to model a generator, exciter, or large motor. The data pipeline, ML
layer, and operator dashboard are asset-agnostic.

---

## 📚 Standards referenced

- **IEEE C57.91** — Guide for Loading Mineral-Oil-Immersed Transformers
- **IEC 60599** — Mineral oil-impregnated electrical equipment in service — Guidance on the interpretation of DGA
- **Duval Triangle Method** — Graphical fault classification using normalized %CH₄ / %C₂H₄ / %C₂H₂
- **Arrhenius / Montsinger's Rule** — Paper insulation aging acceleration

---

## ⚠️ Limitations & honest disclosures

- **Data is synthetic.** The generator mimics realistic relationships but is not
  validated against real transformer telemetry.
- **Physics models are simplified.** Real deployment would use finite-element
  thermal models and full IEEE C57.91 Annex G load-cycle integration.
- **RUL ground truth is synthetic.** Real RUL labels require time-to-failure
  data from a fleet of similar assets — not available for a hackathon submission.
- **The DGA classifier is trained on synthetic class profiles.** Real training
  needs the IEC TC10 fault library or equivalent.

These limitations are inherent to a proof-of-concept assembled in 24 hours.
The architecture and implementation pipeline are correct; calibration against
real data is the production extension.

---

## 👤 Submission details

**Candidate:** Gagandeep Singh Choudhary

**Institution:** JECRC College

**Batch:** B.Tech CSE · 2027

**Problem statement:** Digital twin of transformer or any other electrical equipment

For the full design document, see `docs/ABB_EngineeredX_PS1_DigitalTwin.pdf`.

---

*Submitted to ABB EngineeredX 2.0 · May 2026*
