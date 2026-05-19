"""
Physics-based modeling for transformer digital twin.

Implements simplified versions of:
  - IEEE C57.91 hotspot temperature model
  - Arrhenius-based paper insulation aging
  - Standard T-equivalent circuit losses

These give the "expected behavior" baseline against which the ML layer
detects deviations. Production deployments would use higher-fidelity
finite-element thermal models and detailed loss-of-life integration.
"""

import numpy as np
import pandas as pd

# ============================================================
# Thermal model (IEEE C57.91 style)
# ============================================================

class ThermalModel:
    """
    Simplified IEEE C57.91 top-oil / hotspot temperature model.

    The standard exponential rise equation is:
        theta_top  = theta_ambient + delta_top_rated * (load_pu^1.6)
        theta_hs   = theta_top + delta_hotspot_rated * (load_pu^2.0)

    For real assets these exponents and rise values come from heat-run tests.
    """

    def __init__(
        self,
        delta_top_oil_rated: float = 38.0,   # K at rated load
        delta_hotspot_rated: float = 17.0,   # K at rated load
        n_oil: float = 1.6,
        n_winding: float = 2.0,
    ):
        self.delta_top_oil_rated = delta_top_oil_rated
        self.delta_hotspot_rated = delta_hotspot_rated
        self.n_oil = n_oil
        self.n_winding = n_winding

    def predict_top_oil(self, load_pu, ambient_temp_c):
        """Predict top oil temperature in degC."""
        load_pu = np.asarray(load_pu)
        ambient = np.asarray(ambient_temp_c)
        return ambient + self.delta_top_oil_rated * (load_pu ** self.n_oil)

    def predict_hotspot(self, load_pu, ambient_temp_c):
        """Predict winding hotspot temperature in degC."""
        load_pu = np.asarray(load_pu)
        ambient = np.asarray(ambient_temp_c)
        top_oil = self.predict_top_oil(load_pu, ambient)
        delta_hs = self.delta_hotspot_rated * (load_pu ** self.n_winding)
        return top_oil + delta_hs

    def residual(self, load_pu, ambient_temp_c, measured_hotspot_c):
        """
        Compute residual = measured - predicted.
        Persistent positive residual means cooling degradation or insulation
        problem; persistent negative might mean a sensor miscalibration.
        """
        predicted = self.predict_hotspot(load_pu, ambient_temp_c)
        return np.asarray(measured_hotspot_c) - predicted


# ============================================================
# Arrhenius aging model
# ============================================================

class AgingModel:
    """
    Paper insulation aging via Arrhenius equation.

    Per IEEE C57.91, normalized aging acceleration factor:
        F_AA = exp( 15000 / 383 - 15000 / (hotspot_C + 273) )

    At reference hotspot 110 C (=383 K), F_AA = 1.0.
    F_AA doubles roughly every 7 deg C increase (Montsinger's rule).
    """

    def __init__(self, ref_hotspot_c: float = 110.0, activation_constant: float = 15000.0):
        self.ref_hotspot_c = ref_hotspot_c
        self.activation_constant = activation_constant
        self.ref_kelvin = ref_hotspot_c + 273.0

    def acceleration_factor(self, hotspot_c):
        """Aging acceleration factor F_AA (dimensionless, 1.0 at reference)."""
        hotspot_c = np.asarray(hotspot_c)
        return np.exp(
            self.activation_constant / self.ref_kelvin
            - self.activation_constant / (hotspot_c + 273.0)
        )

    def life_consumed_hours(self, hotspot_history_c, dt_hours: float = 1.0):
        """
        Cumulative equivalent ageing hours: integrating F_AA over time.
        For example 1 hour at hotspot 117 C with F_AA=2 consumes 2 hours of life.
        """
        f_aa = self.acceleration_factor(hotspot_history_c)
        return np.cumsum(f_aa) * dt_hours

    def life_remaining_pct(self, hotspot_history_c, design_life_hours: float = 180_000):
        """Returns remaining life as percentage of design life."""
        consumed = self.life_consumed_hours(hotspot_history_c)
        return np.clip(100.0 * (1.0 - consumed / design_life_hours), 0, 100)


# ============================================================
# Equivalent circuit losses
# ============================================================

class EquivalentCircuit:
    """
    Simplified T-equivalent circuit for a two-winding transformer.

    Used to compute per-period electrical losses (copper + iron) for
    efficiency monitoring and to expose a 'what-if' fault simulation API.
    """

    def __init__(
        self,
        rated_kva: float = 25000,
        copper_loss_rated_kw: float = 120.0,   # P_cu at rated load
        iron_loss_kw: float = 25.0,            # P_fe constant
    ):
        self.rated_kva = rated_kva
        self.copper_loss_rated_kw = copper_loss_rated_kw
        self.iron_loss_kw = iron_loss_kw

    def total_losses_kw(self, load_pu):
        """Total losses = copper (scales with load^2) + iron (constant)."""
        load_pu = np.asarray(load_pu)
        copper = self.copper_loss_rated_kw * (load_pu ** 2)
        iron = self.iron_loss_kw
        return copper + iron

    def efficiency(self, load_pu, power_factor: float = 0.9):
        """Transformer efficiency (0-1)."""
        load_pu = np.asarray(load_pu)
        output_kw = load_pu * self.rated_kva * power_factor
        losses = self.total_losses_kw(load_pu)
        return np.where(output_kw > 0, output_kw / (output_kw + losses), 0)

    def simulate_overload(self, load_pu: float, duration_hours: float,
                          ambient_c: float = 30,
                          thermal: ThermalModel = None,
                          aging: AgingModel = None) -> dict:
        """
        What-if: simulate sustained overload and return projected hotspot
        and equivalent ageing hours consumed.
        """
        thermal = thermal or ThermalModel()
        aging = aging or AgingModel()
        hotspot = thermal.predict_hotspot(load_pu, ambient_c)
        f_aa = aging.acceleration_factor(hotspot)
        life_hours = f_aa * duration_hours
        return {
            "load_pu": load_pu,
            "duration_hours": duration_hours,
            "predicted_hotspot_c": float(hotspot),
            "aging_acceleration_factor": float(f_aa),
            "equivalent_life_hours_consumed": float(life_hours),
            "exceeds_design_limit": bool(hotspot > 140),  # IEEE limit
        }


# ============================================================
# Validation utility: re-fit thermal parameters from history
# ============================================================

def fit_thermal_parameters(df: pd.DataFrame) -> dict:
    """
    Recalibrate thermal model rated-rise parameters from observed history.
    Uses least-squares on the IEEE C57.91 functional form.

    Args:
        df: must contain columns load_pu, ambient_temp_c, winding_hotspot_c

    Returns:
        dict with fitted delta_top_oil_rated and delta_hotspot_rated.
    """
    from scipy.optimize import minimize

    load = df["load_pu"].values
    ambient = df["ambient_temp_c"].values
    measured = df["winding_hotspot_c"].values

    def loss(params):
        d_top, d_hs = params
        predicted = ambient + d_top * (load ** 1.6) + d_hs * (load ** 2.0)
        return np.mean((measured - predicted) ** 2)

    result = minimize(loss, x0=[40.0, 18.0], method="Nelder-Mead")
    return {
        "delta_top_oil_rated": float(result.x[0]),
        "delta_hotspot_rated": float(result.x[1]),
        "rmse_c": float(np.sqrt(result.fun)),
    }


if __name__ == "__main__":
    # Quick self-test
    print("=== Thermal model ===")
    tm = ThermalModel()
    print(f"hotspot at 0.8pu load, 25C ambient: {tm.predict_hotspot(0.8, 25):.1f} C")
    print(f"hotspot at 1.2pu load, 35C ambient: {tm.predict_hotspot(1.2, 35):.1f} C")

    print("\n=== Aging model ===")
    am = AgingModel()
    print(f"F_AA at 110C (reference): {am.acceleration_factor(110):.3f}")
    print(f"F_AA at 117C: {am.acceleration_factor(117):.3f}  (expect ~2.0)")
    print(f"F_AA at 130C: {am.acceleration_factor(130):.3f}")

    print("\n=== Equivalent circuit ===")
    ec = EquivalentCircuit()
    print(f"Losses at 1.0pu: {ec.total_losses_kw(1.0):.1f} kW")
    print(f"Efficiency at 0.8pu, pf 0.9: {ec.efficiency(0.8) * 100:.2f}%")

    print("\n=== What-if simulation: 120% load for 4 hours, 35C ambient ===")
    sim = ec.simulate_overload(1.20, 4.0, 35)
    for k, v in sim.items():
        print(f"  {k}: {v}")
