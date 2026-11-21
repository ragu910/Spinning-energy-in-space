"""
The Spinning Energy in Space -- mission-scale energy-storage model
Author: Ragu Nandan JS (student exploration; revised edition)

Every number marked ASSUMPTION is an invented practice value, not measured data.
Run:  python3 sim.py
"""
import json
import math

import numpy as np

# ------------------------------------------------------------------ assumptions
ORBIT_S = 92 * 60                  # ASSUMPTION: low Earth orbit period
ECL_S = 35 * 60                    # ASSUMPTION: eclipse duration per orbit
SUN_S = ORBIT_S - ECL_S
LOAD_W = 500.0                     # ASSUMPTION: constant electrical load
MISSION_YEARS = 15                 # ASSUMPTION
ORBITS = MISSION_YEARS * 365.25 * 86400 / ORBIT_S

BAT = dict(eta_c=0.90, eta_d=0.95,  # ASSUMPTION: charge / discharge efficiency
           wh_per_kg=120.0,         # ASSUMPTION: pack-level specific energy
           n_ref=5000, dod_ref=0.80, k=1.6)  # ASSUMPTION: cycle-life law

FLY_BASE = dict(eta_c=0.80, eta_g=0.80,   # ASSUMPTION: motor / generator paths
                v_tip=500.0,               # ASSUMPTION: rim speed, m/s
                K=0.8,                     # ASSUMPTION: share of thin-rim ideal
                rotor_frac=0.5,            # ASSUMPTION: rotor = half of system mass
                window=0.75,               # speed range w_max .. w_max/2
                standby_per_h=0.005)       # ASSUMPTION: idle loss, fraction/hour
FLY_ADV = dict(FLY_BASE, eta_c=0.92, eta_g=0.92, v_tip=700.0)

ARRAY_W_PER_KG = 80.0              # ASSUMPTION: solar array specific power
RW_MASS_KG = 6.0                   # ASSUMPTION: four reaction wheels replaced by IPACS
R_ROTOR_M = 0.20                   # ASSUMPTION: wheel radius
I_SAT = 100.0                      # ASSUMPTION: satellite inertia, kg m^2


def wh(j):
    return j / 3600.0


# ------------------------------------------------------------------ battery
def battery_design(load_w=LOAD_W, p=BAT):
    e_out = load_w * ECL_S / p["eta_d"]                      # J leaving cells
    dod_max = p["dod_ref"] * (p["n_ref"] / ORBITS) ** (1 / p["k"])
    cap_j = e_out / dod_max
    mass = wh(cap_j) / p["wh_per_kg"]
    charge_w = e_out / p["eta_c"] / SUN_S                    # average charging power
    return dict(e_out_j=e_out, dod_max=dod_max, capacity_wh=wh(cap_j),
                mass_kg=mass, charge_w=charge_w,
                roundtrip=p["eta_c"] * p["eta_d"])


# ------------------------------------------------------------------ flywheel (time-stepped)
def fly_eclipse(emax, load_w, p, dt=1.0, trace=False):
    s = p["standby_per_h"] / 3600.0
    e = emax
    out = [e]
    for _ in range(int(ECL_S / dt)):
        e -= (load_w / p["eta_g"] + s * e) * dt
        out.append(e)
    return (e, out) if trace else e


def fly_sun(e0, pc, p, dt=1.0, trace=False):
    s = p["standby_per_h"] / 3600.0
    e = e0
    out = [e]
    for _ in range(int(SUN_S / dt)):
        e += (p["eta_c"] * pc - s * e) * dt
        out.append(e)
    return (e, out) if trace else e


def bisect(f, lo, hi, n=80):
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def flywheel_design(load_w=LOAD_W, p=FLY_BASE):
    emin_frac = 1 - p["window"]
    emax = bisect(lambda x: fly_eclipse(x, load_w, p) - emin_frac * x, 1e4, 1e8)
    pc = bisect(lambda x: fly_sun(emin_frac * emax, x, p) - emax, 1.0, 1e5)
    rotor_mass = emax / (p["K"] * 0.5 * p["v_tip"] ** 2)
    sys_mass = rotor_mass / p["rotor_frac"]
    omega_max = p["v_tip"] / R_ROTOR_M
    # four wheels in two counter-rotating pairs
    m_wheel = rotor_mass / 4
    L_wheel = m_wheel * R_ROTOR_M ** 2 * omega_max            # thin rim I = m r^2
    return dict(emax_j=emax, emax_wh=wh(emax), charge_w=pc, rotor_kg=rotor_mass,
                mass_kg=sys_mass, rpm_max=omega_max * 60 / (2 * math.pi),
                omega_max=omega_max, L_wheel=L_wheel,
                spin_rate_deg_s=math.degrees(L_wheel / I_SAT),
                spec_energy_sys_wh_kg=wh(emax) / sys_mass,
                roundtrip=p["eta_c"] * p["eta_g"])


def totals(load_w=LOAD_W):
    b = battery_design(load_w)
    out = {"battery": b}
    for name, p in (("fly_base", FLY_BASE), ("fly_adv", FLY_ADV)):
        out[name] = flywheel_design(load_w, p)
    return out


def scenario_table(d):
    rows = {}
    b = d["battery"]
    b_arr = b["charge_w"] / ARRAY_W_PER_KG
    for key, label in (("fly_base", "Baseline assumptions"),
                       ("fly_adv", "Advanced flywheel")):
        f = d[key]
        f_arr = f["charge_w"] / ARRAY_W_PER_KG
        rows[label] = {
            "battery_only": {"storage": b["mass_kg"], "array": b_arr, "wheels": 0.0,
                             "total": b["mass_kg"] + b_arr},
            "flywheel_only": {"storage": f["mass_kg"], "array": f_arr, "wheels": 0.0,
                              "total": f["mass_kg"] + f_arr},
            "battery_plus_reaction_wheels": {"storage": b["mass_kg"], "array": b_arr,
                                             "wheels": RW_MASS_KG,
                                             "total": b["mass_kg"] + b_arr + RW_MASS_KG},
        }
    return rows


def hybrid_sweep(d):
    """Mass-only blend: fraction f of eclipse energy from the flywheel."""
    b = d["battery"]
    b_tot = b["mass_kg"] + b["charge_w"] / ARRAY_W_PER_KG
    res = {}
    for key in ("fly_base", "fly_adv"):
        f = d[key]
        f_tot = f["mass_kg"] + f["charge_w"] / ARRAY_W_PER_KG
        res[key] = [(x, (1 - x) * b_tot + x * f_tot) for x in (0, 0.25, 0.5, 0.75, 1.0)]
    return res


def book_checks():
    """Re-compute every worked number printed in the book."""
    out = {}
    m, r, rpm = 2.0, 0.10, 600
    i = 0.5 * m * r ** 2
    w = 2 * math.pi * rpm / 60
    out["disk_I"] = i
    out["disk_omega"] = w
    out["disk_E_J"] = 0.5 * i * w ** 2
    out["disk_L"] = i * w
    out["after_30d_fraction"] = 0.99 ** 30
    out["after_30d_J"] = out["disk_E_J"] * 0.99 ** 30
    out["solar_100W_1h_J"] = 100 * 3600
    out["stored_at_80pct_J"] = 100 * 3600 * 0.8
    out["fill_100kJ_s"] = 100_000 / 80
    out["extract_mech_W"] = 20 / 0.8
    out["removed_1h_J"] = 25 * 3600
    # 100 kJ rim rotor
    i2 = 5 * 0.15 ** 2
    w2 = math.sqrt(2 * 100_000 / i2)
    out["rim_I"] = i2
    out["rim_omega"] = w2
    out["rim_rpm"] = w2 * 60 / (2 * math.pi)
    out["rim_v"] = w2 * 0.15
    out["rim_wh_per_kg"] = 100_000 / 5 / 3600
    # 20 W eclipse comparison
    need = 20 * ECL_S
    out["ecl_need_J"] = need
    out["bat_from_storage"] = need / 0.95
    out["bat_solar"] = need / 0.95 / 0.90
    out["fly_from_storage"] = need / 0.80
    out["fly_solar"] = need / 0.80 / 0.80
    out["extra_solar_ratio"] = out["fly_solar"] / out["bat_solar"]
    out["orbits_per_year"] = 365.25 * 86400 / ORBIT_S
    out["orbits_15y"] = ORBITS
    # momentum example
    out["sat_turn_deg_s"] = math.degrees(out["disk_L"] / 50)
    out["usable_fraction_half_speed"] = 1 - 0.5 ** 2
    out["top_quarter_energy_share"] = 1 - 0.75 ** 2
    return out


if __name__ == "__main__":
    chk = book_checks()
    print("== Book worked-number check ==")
    for k, v in chk.items():
        print(f"{k:28s} {v:,.4f}")

    d = totals()
    print("\n== Mission-scale design (500 W, 35 min eclipse, 15 years) ==")
    print(json.dumps(d, indent=2))
    tab = scenario_table(d)
    print("\n== Mass scenarios (kg) ==")
    print(json.dumps(tab, indent=2))
    print("\n== Hybrid blend ==")
    print(json.dumps(hybrid_sweep(d), indent=2))

    json.dump(dict(checks=chk, design=d, scenarios=tab, hybrid=hybrid_sweep(d)),
              open("results.json", "w"), indent=2)
