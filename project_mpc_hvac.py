#!/usr/bin/env python3
"""Three-week teaching project: single-zone HVAC model predictive control.

Run with: python project_mpc_hvac.py
Dependencies: numpy, scipy, matplotlib. All powers are kW electrical unless
explicitly labelled thermal; the simplified zone model is intentionally gray-box.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --------------------------- Project configuration ---------------------------
DT_H = 0.25                         # 15-minute sample period
N = 96                              # 24 hours / 15 minutes
TIMES = np.arange(N) * DT_H         # interval-start times, hours after midnight
STATE_TIMES = np.arange(N + 1) * DT_H
C_J_K = 2.0e6                       # effective zone thermal capacitance
R_K_W = 0.020                       # envelope resistance (UA = 50 W/K)
COP = 3.0
P_MAX_KW = 8.0 / COP                 # electrical input; gives 8 kW thermal cooling capacity
T_INITIAL_C = 26.0
COMFORT_LOW_C, COMFORT_HIGH_C = 21.0, 21.0
OCC_START_H, OCC_END_H = 8.0, 18.0


def make_profiles():
    """Create transparent hot-summer weather, gains, and TOU price profiles."""
    # Piecewise-linear outdoor profile: hot afternoon, cooler pre-dawn/night.
    knots_h = np.array([0, 4, 8, 12, 16, 20, 24], dtype=float)
    knots_c = np.array([27, 26, 28, 34, 38, 34, 29], dtype=float)
    tout = np.interp(TIMES, knots_h, knots_c)
    # Smooth daylight solar gain (up to 1.6 kW sensible heat into this zone).
    solar = 1600.0 * np.maximum(0.0, np.sin(np.pi * (TIMES - 6.0) / 12.0))
    occupied = (TIMES >= OCC_START_H) & (TIMES < OCC_END_H)
    # People, lighting, and plug loads combined as sensible internal gain.
    q_occ = np.where(occupied, 650.0, 0.0)
    # TOU tariff: inexpensive overnight, normal daytime, expensive 17:00-21:00.
    price = np.where((TIMES >= 17) & (TIMES < 21), 0.52,
             np.where((TIMES < 6), 0.12, 0.24))  # currency units per kWh
    return tout, solar, q_occ, occupied, price


def simulate_zone(u_kw, tout, solar, q_occ):
    """Forward-Euler 1R1C model; u is electrical cooling power in kW."""
    t = np.empty(N + 1, dtype=float)
    t[0] = T_INITIAL_C
    dt_s = DT_H * 3600.0
    for k in range(N):
        q_envelope = (tout[k] - t[k]) / R_K_W
        q_hvac = -COP * u_kw[k] * 1000.0
        t[k + 1] = t[k] + dt_s / C_J_K * (q_envelope + solar[k] + q_occ[k] + q_hvac)
    return t


def baseline_controller(tout, solar, q_occ):
    """Bang-bang thermostat requested in the brief (22.5 on / 21.5 off)."""
    u = np.zeros(N)
    t = np.empty(N + 1)
    t[0] = T_INITIAL_C
    dt_s = DT_H * 3600.0
    for k in range(N):
        if t[k] > 22.5:
            u[k] = P_MAX_KW
        elif t[k] < 21.5:
            u[k] = 0.0
        else:
            # Retain the previous actuator state in the dead band.
            u[k] = u[k - 1] if k else 0.0
        q_env = (tout[k] - t[k]) / R_K_W
        t[k + 1] = t[k] + dt_s / C_J_K * (
            q_env + solar[k] + q_occ[k] - COP * u[k] * 1000.0)
    return u, t


def mpc_slsqp(tout, solar, q_occ, price, verbose=False):
    """Full-day convex program solved by SLSQP with analytic objective/Jacobian.

    The linear state sensitivities are built once. Comfort inequalities constrain
    predicted states at occupied interval boundaries (including 08:00 and 18:00).
    The small tracking term selects a comfortable trajectory near 22 C among
    minimum-cost feasible solutions. In practice this is re-solved each sample.
    """
    a = 1.0 - (DT_H * 3600.0) / (C_J_K * R_K_W)
    b = -(DT_H * 3600.0 / C_J_K) * COP * 1000.0  # degC per kW per step
    dt_s = DT_H * 3600.0
    # No-HVAC affine trajectory, plus the lower-triangular input sensitivity.
    free = np.empty(N + 1)
    free[0] = T_INITIAL_C
    for k in range(N):
        free[k + 1] = a * free[k] + (1 - a) * tout[k] + dt_s / C_J_K * (solar[k] + q_occ[k])
    sensitivity = np.zeros((N + 1, N))
    for state_idx in range(1, N + 1):
        sensitivity[state_idx, :state_idx] = b * a ** np.arange(state_idx - 1, -1, -1)

    # Require comfort for all occupied predicted states; 08:00 boundary is included.
    occ_state = np.where((STATE_TIMES >= OCC_START_H) & (STATE_TIMES <= OCC_END_H))[0]
    A_occ = sensitivity[occ_state, :]
    low = np.full(len(occ_state), COMFORT_LOW_C)
    high = np.full(len(occ_state), COMFORT_HIGH_C)
    # Electricity bill is price * kWh. A modest comfort-tracking penalty is added.
    tracking_weight = 0.08
    target = 22.0
    def objective(u):
        temps = free[1:] + sensitivity[1:] @ u
        return float(np.dot(price * DT_H, u) + tracking_weight * np.sum((temps - target) ** 2))
    def objective_jac(u):
        temps = free[1:] + sensitivity[1:] @ u
        return price * DT_H + 2 * tracking_weight * sensitivity[1:].T @ (temps - target)
    def comfort_fun(u):
        tc = free[occ_state] + A_occ @ u
        return np.r_[tc - low, high - tc]
    def comfort_jac(u):
        return np.vstack([A_occ, -A_occ])

    # A warm-start near full cooling during occupied hours helps SLSQP converge.
    initial = np.where((TIMES >= 7) & (TIMES < 18), 2.0, 0.0)
    result = minimize(objective, initial, method="SLSQP", jac=objective_jac,
                      bounds=[(0.0, P_MAX_KW)] * N,
                      constraints=[{"type": "ineq", "fun": comfort_fun, "jac": comfort_jac}],
                      options={"maxiter": 1000, "ftol": 1e-10, "disp": verbose})
    if not result.success:
        raise RuntimeError(f"SLSQP failed: {result.message}")
    u = np.clip(result.x, 0.0, P_MAX_KW)
    return u, simulate_zone(u, tout, solar, q_occ), result


def mpc_cvxpy_alternative(tout, solar, q_occ, price):
    """Equivalent convex formulation, optional when cvxpy is installed.

    This function is deliberately separate so the project runs with only NumPy,
    SciPy, and Matplotlib. Install cvxpy to use this alternative solver.
    """
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError("Optional solver unavailable; install cvxpy to use this function") from exc
    a = 1.0 - (DT_H * 3600.0) / (C_J_K * R_K_W)
    b = -(DT_H * 3600.0 / C_J_K) * COP * 1000.0
    dt_s = DT_H * 3600.0
    u = cp.Variable(N)
    temp = cp.Variable(N + 1)
    cons = [temp[0] == T_INITIAL_C, u >= 0, u <= P_MAX_KW]
    for k in range(N):
        cons += [temp[k + 1] == a * temp[k] + (1-a) * tout[k]
                 + dt_s / C_J_K * (solar[k] + q_occ[k]) + b * u[k]]
    occ_state = np.where((STATE_TIMES >= OCC_START_H) & (STATE_TIMES <= OCC_END_H))[0]
    cons += [temp[occ_state] >= COMFORT_LOW_C, temp[occ_state] <= COMFORT_HIGH_C]
    objective = cp.Minimize(price @ (u * DT_H) + 0.08 * cp.sum_squares(temp[1:] - 22.0))
    problem = cp.Problem(objective, cons)
    problem.solve()
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"CVXPY solve failed: {problem.status}")
    return np.asarray(u.value).ravel(), np.asarray(temp.value).ravel()


def main():
    outdir = Path(__file__).resolve().parent
    tout, solar, q_occ, occupied, price = make_profiles()
    u_base, t_base = baseline_controller(tout, solar, q_occ)
    u_mpc, t_mpc, result = mpc_slsqp(tout, solar, q_occ, price)

    # Summary metrics use the same 15-minute energy and tariff accounting.
    peak = (TIMES >= 17) & (TIMES < 21)
    def metrics(u):
        energy = float(np.sum(u * DT_H))
        cost = float(np.sum(u * DT_H * price))
        peak_kwh = float(np.sum(u[peak] * DT_H))
        return energy, cost, peak_kwh
    eb, cb, pb = metrics(u_base)
    em, cm, pm = metrics(u_mpc)
    occ_temp = t_mpc[1:][occupied]
    occ_min, occ_max = float(np.min(occ_temp)), float(np.max(occ_temp))
    comfort_violation = max(0.0, COMFORT_LOW_C - occ_min, occ_max - COMFORT_HIGH_C)
    assert comfort_violation <= 0.05, f"MPC comfort violation {comfort_violation:.4f} C"
    assert cm < cb, f"MPC cost {cm:.3f} is not below baseline cost {cb:.3f}"

    rows = [
        {"controller": "Baseline bang-bang", "energy_kWh": eb, "cost_currency": cb,
         "peak_17_21_kWh": pb, "occupied_min_C": float(np.min(t_base[1:][occupied])),
         "occupied_max_C": float(np.max(t_base[1:][occupied]))},
        {"controller": "Full-day MPC (SLSQP)", "energy_kWh": em, "cost_currency": cm,
         "peak_17_21_kWh": pm, "occupied_min_C": occ_min, "occupied_max_C": occ_max},
    ]
    csv_path = outdir / "results_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    # Figure 1: indoor temperature, outdoor temperature, and occupied comfort band.
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.4), constrained_layout=True)
    ax.fill_between(STATE_TIMES, COMFORT_LOW_C, COMFORT_HIGH_C,
                    where=(STATE_TIMES >= OCC_START_H) & (STATE_TIMES <= OCC_END_H),
                    color="#8ecae6", alpha=0.30, label="Occupied comfort band (21–24 °C)")
    ax.plot(STATE_TIMES, t_base, color="#d1495b", lw=2, label="Baseline zone temperature")
    ax.plot(STATE_TIMES, t_mpc, color="#0077b6", lw=2.4, label="MPC zone temperature")
    ax.plot(TIMES, tout, color="#555555", lw=1.7, ls="--", label="Outdoor temperature")
    ax.axvspan(OCC_START_H, OCC_END_H, color="#8ecae6", alpha=0.07)
    ax.set(xlim=(0, 24), xlabel="Hour of day", ylabel="Temperature (°C)",
           title="Single-zone office temperature: baseline vs. MPC")
    ax.set_xticks(np.arange(0, 25, 2)); ax.legend(loc="best", frameon=True)
    fig.savefig(outdir / "hvac_mpc_temperature.png", dpi=150)
    plt.close(fig)

    # Figure 2: electrical cooling power and TOU electricity tariff.
    fig, ax = plt.subplots(figsize=(10, 5.4), constrained_layout=True)
    ax.step(TIMES, u_base, where="post", color="#d1495b", lw=1.8, label="Baseline cooling power")
    ax.step(TIMES, u_mpc, where="post", color="#0077b6", lw=2.1, label="MPC cooling power")
    ax.set(xlim=(0, 24), xlabel="Hour of day", ylabel="Electrical cooling power (kW)",
           title="Cooling power and time-of-use electricity price")
    ax.set_xticks(np.arange(0, 25, 2)); ax.set_ylim(bottom=0)
    ax2 = ax.twinx()
    ax2.step(TIMES, price, where="post", color="#f4a261", lw=1.7, ls="--", label="Electricity price")
    ax2.set_ylabel("Price (currency units/kWh)", color="#b05d00")
    ax2.tick_params(axis="y", labelcolor="#b05d00")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc="upper left", frameon=True)
    fig.savefig(outdir / "hvac_mpc_power_price.png", dpi=150)
    plt.close(fig)

    print("Model: 1R1C, C=2.0e6 J/K, R=0.020 K/W, COP=3, Pmax=2.67 kW electric (8 kW thermal); dt=15 min")
    print(f"SLSQP: success={result.success}, iterations={result.nit}, objective={result.fun:.4f}")
    print(f"MPC occupied temperature range: {occ_min:.3f} to {occ_max:.3f} C; max comfort violation={comfort_violation:.4f} C")
    print(f"Cost check: MPC {cm:.4f} < baseline {cb:.4f}: {cm < cb}")
    print("controller,energy_kWh,cost_currency,peak_17_21_kWh,occupied_min_C,occupied_max_C")
    for r in rows:
        print(",".join(str(r[k]) for k in r))
    print(f"Wrote: {outdir / 'hvac_mpc_temperature.png'}")
    print(f"Wrote: {outdir / 'hvac_mpc_power_price.png'}")
    print(f"Wrote: {csv_path}")

if __name__ == "__main__":
    main()
