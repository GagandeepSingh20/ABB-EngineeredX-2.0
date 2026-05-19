"""Reusable plotting functions for the digital twin notebooks and dashboard."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Consistent design palette
NAVY = "#0F2A47"
SLATE = "#334155"
MUTED = "#64748B"
LIGHT = "#E2E8F0"
RED = "#DC2626"
AMBER = "#D97706"
GREEN = "#16A34A"
BLUE = "#2563EB"


def setup_style():
    """Apply consistent rcParams to all charts."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": SLATE,
        "axes.labelcolor": SLATE,
        "axes.titlecolor": NAVY,
        "axes.titleweight": "bold",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 110,
    })


def plot_load_and_temperature(df: pd.DataFrame, hours: int = 24 * 30):
    """Show load profile and hotspot temperature for first N hours."""
    setup_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    sub = df.head(hours)

    ax1.plot(sub.index, sub["load_pu"], color=BLUE, linewidth=0.8)
    ax1.fill_between(sub.index, 0, sub["load_pu"], color=BLUE, alpha=0.15)
    ax1.set_ylabel("Load (p.u.)")
    ax1.set_title("Load profile and winding hotspot temperature", loc="left")
    ax1.set_ylim(0, 1.2)

    ax2.plot(sub.index, sub["winding_hotspot_c"], color=RED, linewidth=0.8, label="hotspot")
    ax2.plot(sub.index, sub["ambient_temp_c"], color=MUTED, linewidth=0.6, label="ambient")
    ax2.set_ylabel("Temperature (degC)")
    ax2.set_xlabel("Hour")
    ax2.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    return fig


def plot_residuals(measured, predicted, title="Physics residual"):
    """Plot measured vs predicted with residual subplot."""
    setup_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    idx = np.arange(len(measured))
    ax1.plot(idx, predicted, color=BLUE, linewidth=0.7, label="physics-predicted", alpha=0.9)
    ax1.plot(idx, measured, color=RED, linewidth=0.7, label="measured", alpha=0.7)
    ax1.set_ylabel("Hotspot (degC)")
    ax1.set_title(title, loc="left")
    ax1.legend(loc="upper right", frameon=False)

    residual = np.array(measured) - np.array(predicted)
    ax2.fill_between(idx, 0, residual, color=AMBER, alpha=0.4)
    ax2.axhline(0, color=SLATE, linewidth=0.5)
    ax2.set_ylabel("Residual (degC)")
    ax2.set_xlabel("Hour")
    plt.tight_layout()
    return fig


def plot_anomaly_score(scores, fault_label=None, threshold: float = 0.5):
    """Plot anomaly score time-series with optional fault overlay."""
    setup_style()
    fig, ax = plt.subplots(figsize=(11, 3.2))
    idx = np.arange(len(scores))
    ax.fill_between(idx, 0, scores, color=BLUE, alpha=0.15)
    ax.plot(idx, scores, color=BLUE, linewidth=0.7)
    ax.axhline(threshold, color=RED, linestyle="--", linewidth=1)
    ax.text(len(scores) * 0.98, threshold + 0.02, f"alert threshold ({threshold})",
            color=RED, fontsize=8, ha="right")

    # Highlight known fault periods
    if fault_label is not None:
        for fault_class in np.unique(fault_label):
            if fault_class == 0:
                continue
            mask = (fault_label == fault_class)
            ax.fill_between(idx, 0, 1, where=mask, color=RED, alpha=0.08,
                            transform=ax.get_xaxis_transform())

    ax.set_xlim(0, len(scores))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Anomaly score")
    ax.set_title("Multivariate anomaly score", loc="left")
    plt.tight_layout()
    return fig


def plot_classification_confidence(class_names, probabilities, true_label: int = None):
    """Horizontal bar chart of classifier confidence."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 3))
    y_pos = np.arange(len(class_names))
    colors = [AMBER if i == np.argmax(probabilities) else LIGHT for i in range(len(class_names))]
    bars = ax.barh(y_pos, probabilities * 100, color=colors, edgecolor=SLATE, linewidth=0.4)
    for i, (bar, p) in enumerate(zip(bars, probabilities)):
        ax.text(p * 100 + 1, bar.get_y() + bar.get_height() / 2, f"{p*100:.1f}%",
                va="center", fontsize=9, color=SLATE)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(80, probabilities.max() * 110))
    ax.set_xlabel("Classifier confidence (%)")
    ax.set_title("DGA fault classification", loc="left")
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(bottom=False)
    ax.set_xticks([])
    plt.tight_layout()
    return fig


def plot_rul_forecast(predicted_rul, dates=None, maintenance_threshold: int = 90):
    """Plot RUL trajectory over time."""
    setup_style()
    fig, ax = plt.subplots(figsize=(11, 3.2))
    idx = dates if dates is not None else np.arange(len(predicted_rul))
    ax.fill_between(idx, 0, predicted_rul, color=GREEN, alpha=0.15)
    ax.plot(idx, predicted_rul, color=GREEN, linewidth=1.2)
    ax.axhline(maintenance_threshold, color=RED, linestyle="--", linewidth=1)
    ax.text(len(predicted_rul) * 0.98 if dates is None else dates[-1],
            maintenance_threshold + 5,
            f"maintenance threshold ({maintenance_threshold} d)",
            color=RED, fontsize=8, ha="right")
    ax.set_ylabel("RUL (days)")
    ax.set_xlabel("Time")
    ax.set_title("Predicted remaining useful life", loc="left")
    plt.tight_layout()
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 10):
    """Horizontal bar chart of top N feature importances."""
    setup_style()
    top = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 0.35 * top_n + 0.8))
    ax.barh(top["feature"], top["importance"], color=NAVY, alpha=0.85)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Top feature importances (XGBoost)", loc="left")
    plt.tight_layout()
    return fig
