"""
Synthetic transformer telemetry generator.

Produces hourly sensor data for a power transformer over a configurable time window.
Models realistic relationships between load, ambient temperature, hotspot temperature,
DGA gases, and degradation, with occasional fault events injected.

This is for demonstration only - real deployments would ingest data via MQTT/OPC-UA
from physical sensors.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_transformer_data(
    days: int = 180,
    samples_per_hour: int = 1,
    rated_kva: float = 25000,
    rated_voltage_hv: float = 132,
    rated_voltage_lv: float = 33,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic transformer telemetry.

    Args:
        days: number of days to simulate
        samples_per_hour: temporal resolution
        rated_kva: nameplate rating
        rated_voltage_hv: HV side rated voltage (kV)
        rated_voltage_lv: LV side rated voltage (kV)
        seed: random seed for reproducibility

    Returns:
        DataFrame with columns:
            timestamp, load_pu, ambient_temp_c, oil_temp_top_c, oil_temp_bottom_c,
            winding_hotspot_c, voltage_hv_kv, voltage_lv_kv, current_hv_a, current_lv_a,
            h2_ppm, ch4_ppm, c2h2_ppm, c2h4_ppm, c2h6_ppm, co_ppm, co2_ppm,
            vibration_rms_mm_s, partial_discharge_pc, oltc_position,
            fault_label  (ground truth - only used for training)
    """
    rng = np.random.default_rng(seed)
    n = days * 24 * samples_per_hour

    # Time index
    start = datetime(2025, 11, 1, 0, 0, 0)
    timestamps = [start + timedelta(hours=i / samples_per_hour) for i in range(n)]
    hours_of_day = np.array([t.hour + t.minute / 60 for t in timestamps])
    days_of_year = np.array([t.timetuple().tm_yday for t in timestamps])

    # ===== Load profile - daily cycle + weekday/weekend + seasonal =====
    # Industrial/urban load: peak in morning + evening, lower at night
    daily_load = 0.55 + 0.25 * np.sin(2 * np.pi * (hours_of_day - 7) / 24) \
                 + 0.15 * np.sin(2 * np.pi * (hours_of_day - 19) / 24)
    seasonal = 0.05 * np.sin(2 * np.pi * days_of_year / 365)
    weekend_factor = np.array([0.85 if (t.weekday() >= 5) else 1.0 for t in timestamps])
    load_pu = (daily_load + seasonal) * weekend_factor + rng.normal(0, 0.03, n)
    load_pu = np.clip(load_pu, 0.15, 1.15)

    # ===== Ambient temperature - daily cycle + seasonal =====
    seasonal_temp = 20 + 12 * np.sin(2 * np.pi * (days_of_year - 80) / 365)  # peak in summer
    daily_temp = 5 * np.sin(2 * np.pi * (hours_of_day - 14) / 24)  # peak at 2pm
    ambient_temp = seasonal_temp + daily_temp + rng.normal(0, 1.0, n)

    # ===== Oil and winding temperatures (simplified IEEE C57.91-style model) =====
    # top oil = ambient + delta_top_oil where delta scales with load^1.6
    delta_top_oil_rated = 38  # K at rated load
    delta_top_oil = delta_top_oil_rated * (load_pu ** 1.6)
    oil_temp_top = ambient_temp + delta_top_oil

    # bottom oil is cooler
    oil_temp_bottom = ambient_temp + 0.6 * delta_top_oil

    # winding hotspot = top oil + hotspot rise (scales with load^2)
    delta_hotspot_rated = 17  # K at rated load
    delta_hotspot = delta_hotspot_rated * (load_pu ** 2)
    hotspot = oil_temp_top + delta_hotspot + rng.normal(0, 0.8, n)

    # ===== Voltages and currents =====
    voltage_hv = rated_voltage_hv * (1.0 + rng.normal(0, 0.005, n))
    voltage_lv = rated_voltage_lv * (1.0 + rng.normal(0, 0.008, n))
    # I = S / (sqrt(3) * V) ; S = load_pu * rated_kva
    s_kva = load_pu * rated_kva
    current_hv = s_kva / (np.sqrt(3) * voltage_hv)
    current_lv = s_kva / (np.sqrt(3) * voltage_lv)

    # ===== DGA gases (ppm) - baseline + temperature-driven slow build =====
    # Arrhenius-like accumulation with hotspot temperature
    aging_factor = np.exp((hotspot - 80) / 25)  # higher at high temps
    aging_factor_cumulative = np.cumsum(aging_factor) / 1000

    h2 = 30 + 0.4 * aging_factor_cumulative + rng.normal(0, 3, n)
    ch4 = 20 + 0.3 * aging_factor_cumulative + rng.normal(0, 2, n)
    c2h2 = 1 + 0.02 * aging_factor_cumulative + rng.normal(0, 0.3, n)
    c2h4 = 15 + 0.2 * aging_factor_cumulative + rng.normal(0, 2, n)
    c2h6 = 12 + 0.15 * aging_factor_cumulative + rng.normal(0, 1.5, n)
    co = 400 + 1.2 * aging_factor_cumulative + rng.normal(0, 15, n)
    co2 = 3500 + 8 * aging_factor_cumulative + rng.normal(0, 80, n)

    # ===== Vibration and partial discharge =====
    vibration = 1.2 + 0.5 * load_pu + rng.normal(0, 0.15, n)
    partial_discharge = np.clip(rng.exponential(50, n), 0, 5000)

    # ===== OLTC tap position (discrete) =====
    oltc_position = np.full(n, 9, dtype=int)
    # Small drift over time as grid voltage shifts
    for i in range(50, n):
        if rng.random() < 0.001:
            oltc_position[i:] = np.clip(oltc_position[i] + rng.choice([-1, 1]), 1, 17)

    # ===== Fault label (ground truth) =====
    # 0 = normal, 1 = partial discharge, 2 = low-energy discharge, 3 = arcing,
    # 4 = thermal fault <300C, 5 = thermal fault >700C
    fault_label = np.zeros(n, dtype=int)

    # Inject a thermal fault <300C around days 90-95 (gradual onset, then cleared)
    fault1_start = int(n * 0.50)
    fault1_end = int(n * 0.53)
    ramp = np.linspace(0, 1, fault1_end - fault1_start)
    ch4[fault1_start:fault1_end] += 80 * ramp
    c2h4[fault1_start:fault1_end] += 60 * ramp
    c2h6[fault1_start:fault1_end] += 30 * ramp
    hotspot[fault1_start:fault1_end] += 8 * ramp
    fault_label[fault1_start:fault1_end] = 4

    # Inject a partial discharge event around days 140-142
    fault2_start = int(n * 0.78)
    fault2_end = int(n * 0.79)
    ramp2 = np.linspace(0, 1, fault2_end - fault2_start)
    h2[fault2_start:fault2_end] += 150 * ramp2
    ch4[fault2_start:fault2_end] += 10 * ramp2
    partial_discharge[fault2_start:fault2_end] += 800 * ramp2
    fault_label[fault2_start:fault2_end] = 1

    # Sensor noise / dropouts
    sensor_glitch = rng.random(n) < 0.003
    hotspot = np.where(sensor_glitch, hotspot + rng.normal(0, 8, n), hotspot)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "load_pu": np.round(load_pu, 4),
        "ambient_temp_c": np.round(ambient_temp, 2),
        "oil_temp_top_c": np.round(oil_temp_top, 2),
        "oil_temp_bottom_c": np.round(oil_temp_bottom, 2),
        "winding_hotspot_c": np.round(hotspot, 2),
        "voltage_hv_kv": np.round(voltage_hv, 3),
        "voltage_lv_kv": np.round(voltage_lv, 3),
        "current_hv_a": np.round(current_hv, 2),
        "current_lv_a": np.round(current_lv, 2),
        "h2_ppm": np.round(np.clip(h2, 0, None), 2),
        "ch4_ppm": np.round(np.clip(ch4, 0, None), 2),
        "c2h2_ppm": np.round(np.clip(c2h2, 0, None), 2),
        "c2h4_ppm": np.round(np.clip(c2h4, 0, None), 2),
        "c2h6_ppm": np.round(np.clip(c2h6, 0, None), 2),
        "co_ppm": np.round(np.clip(co, 0, None), 2),
        "co2_ppm": np.round(np.clip(co2, 0, None), 2),
        "vibration_rms_mm_s": np.round(np.clip(vibration, 0, None), 3),
        "partial_discharge_pc": np.round(partial_discharge, 1),
        "oltc_position": oltc_position,
        "fault_label": fault_label,
    })

    return df


def generate_dga_training_set(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a DGA training set with known fault types based on IEC 60599 / Duval Triangle ranges.

    Returns DataFrame with 7 gas concentrations + fault_type label.
    Fault types:
        0 = normal ageing
        1 = partial discharge
        2 = low-energy discharge
        3 = arcing
        4 = thermal fault <300C
        5 = thermal fault >700C
    """
    rng = np.random.default_rng(seed)
    samples = []

    # Base concentrations and typical fault signatures (ppm)
    # Profile: (H2, CH4, C2H2, C2H4, C2H6, CO, CO2) means
    profiles = {
        0: ([35, 25, 1, 18, 14, 450, 3600], 0.15),   # normal
        1: ([400, 35, 2, 20, 15, 480, 3800], 0.30),  # PD: high H2
        2: ([280, 60, 35, 40, 12, 460, 3700], 0.25),  # low-energy: C2H2 + H2
        3: ([320, 100, 280, 250, 18, 500, 3900], 0.35),  # arcing: C2H2 dominant
        4: ([45, 180, 1, 200, 90, 700, 4500], 0.20),  # thermal <300C
        5: ([60, 320, 5, 480, 75, 900, 5200], 0.25),  # thermal >700C
    }

    per_class = n_samples // 6
    for label, (means, noise_frac) in profiles.items():
        for _ in range(per_class):
            sample = [max(0, m * (1 + rng.normal(0, noise_frac))) for m in means]
            samples.append(sample + [label])

    df = pd.DataFrame(samples, columns=[
        "h2_ppm", "ch4_ppm", "c2h2_ppm", "c2h4_ppm", "c2h6_ppm",
        "co_ppm", "co2_ppm", "fault_type"
    ])
    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    print("Generating 180-day telemetry...")
    df = generate_transformer_data(days=180)
    print(f"  shape: {df.shape}")
    print(f"  fault distribution: {df.fault_label.value_counts().to_dict()}")
    print(f"  hotspot range: {df.winding_hotspot_c.min():.1f} - {df.winding_hotspot_c.max():.1f} C")

    print("\nGenerating DGA training set...")
    dga = generate_dga_training_set(2000)
    print(f"  shape: {dga.shape}")
    print(f"  fault distribution: {dga.fault_type.value_counts().to_dict()}")
