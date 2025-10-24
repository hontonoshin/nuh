#!/usr/bin/env python3
"""
Nuclear-Enabled Hydrogen – Single-Page App (2025-10-24)
-------------------------------------------------------
Inputs + Run + Plot + Conclusions + Problem Solver on ONE screen.

Requirements: Python 3.10+, numpy, matplotlib (TkAgg).
"""

import csv
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -------------------------
# Domain Models & Constants
# -------------------------
LHV_H2_MJ_PER_KG = 120.0
SEC_PER_HOUR = 3600


@dataclass
class Reactor:
    name: str
    thermal_power_MWt: float
    electric_efficiency: float
    outlet_temp_C: float
    capacity_factor: float
    def electric_power_MWe(self) -> float:
        return self.thermal_power_MWt * self.electric_efficiency


@dataclass
class Pathway:
    code: str
    kind: str   # 'electrolysis' or 'thermochemical'
    display: str


def alkaline_kwh_per_kg(temp_C: float) -> float:
    base = 51.0 - 0.01 * (temp_C - 60.0)
    return float(np.clip(base, 48.0, 55.0))


def pem_kwh_per_kg(temp_C: float) -> float:
    base = 50.0 - 0.008 * (temp_C - 60.0)
    return float(np.clip(base, 47.0, 54.0))


def soec_split_energy(temp_C: float) -> Tuple[float, float]:
    T = np.clip(temp_C, 650, 900)
    e = 38.0 - 5.0 * (T - 650) / 250.0   # 38 -> 33
    q = 15.0 + 10.0 * (T - 650) / 250.0  # 15 -> 25
    return float(e), float(q)


def si_cycle_efficiency(temp_C: float) -> float:
    T = np.clip(temp_C, 750, 1000)
    eta = 0.35 + (0.47 - 0.35) * (T - 800) / 150.0
    return float(np.clip(eta, 0.30, 0.50))


PATHWAYS: Dict[str, Pathway] = {
    "ALK": Pathway("ALK", "electrolysis", "Alkaline Electrolysis"),
    "PEM": Pathway("PEM", "electrolysis", "PEM Electrolysis"),
    "SOEC": Pathway("SOEC", "electrolysis", "High-Temperature SOEC"),
    "SI":   Pathway("SI",   "thermochemical", "Thermochemical S–I Cycle"),
}

PRESET_REACTORS: Dict[str, Reactor] = {
    "PWR":     Reactor("PWR",     3400, 0.33, 320, 0.92),
    "SMR-PWR": Reactor("SMR-PWR",  450, 0.31, 300, 0.92),
    "HTGR":    Reactor("HTGR",     600, 0.40, 750, 0.90),
    "VHTR":    Reactor("VHTR",     600, 0.44, 900, 0.90),
}

# -------------------------
# Core Calculations
# -------------------------
def h2_from_electricity(power_MWe: float, kwh_per_kg: float) -> float:
    kW = max(power_MWe, 0.0) * 1e3
    return 0.0 if kwh_per_kg <= 0 else kW / kwh_per_kg


def h2_from_thermal(power_MWt: float, eta_LHV: float) -> float:
    MJ_per_h = max(power_MWt, 0.0) * 1e6 * SEC_PER_HOUR / 1e6
    return (max(eta_LHV, 0.0) * MJ_per_h) / LHV_H2_MJ_PER_KG


def lcohUSD_per_kg(capex_USD_per_kW: float,
                   opex_frac_per_year: float,
                   electricity_USD_per_MWh: float,
                   heat_USD_per_MWhth: float,
                   kwh_electric_per_kg: float,
                   MJ_heat_per_kg: float,
                   stack_kWe: float,
                   cf: float,
                   lifetime_years: int,
                   discount_rate: float) -> float:
    r = max(discount_rate, 0.0)
    n = max(int(lifetime_years), 1)
    crf = (r * (1 + r) ** n) / ((1 + r) ** n - 1) if r > 0 else 1 / n

    annual_kWh = max(stack_kWe, 1.0) * 8760 * np.clip(cf, 0.0, 1.0)
    annual_kg  = annual_kWh / max(kwh_electric_per_kg, 1e-9)

    annual_capex = max(capex_USD_per_kW, 0.0) * max(stack_kWe, 1.0) * crf
    annual_opex  = max(opex_frac_per_year, 0.0) * max(capex_USD_per_kW, 0.0) * max(stack_kWe, 1.0)

    energy_cost = (max(electricity_USD_per_MWh, 0.0) * (kwh_electric_per_kg / 1000.0)
                   + max(heat_USD_per_MWhth, 0.0) * (MJ_heat_per_kg / 3600.0))

    fixed_per_kg = (annual_capex + annual_opex) / max(annual_kg, 1e-9)
    return float(fixed_per_kg + energy_cost)


def simulate(reac: Reactor,
             pathway_code: str,
             split_to_electric: float,
             electrolyzer_temp_C: float,
             price_elec_USD_per_MWh: float,
             price_heat_USD_per_MWhth: float,
             capex_USD_per_kW: float,
             opex_frac: float,
             lifetime_years: int,
             discount_rate: float):
    pathway = PATHWAYS[pathway_code]
    f = float(np.clip(split_to_electric, 0.0, 1.0))
    Pth_total = float(reac.thermal_power_MWt)
    Pe = f * Pth_total * float(reac.electric_efficiency)
    Q  = (1.0 - f) * Pth_total

    if pathway.kind == "electrolysis":
        if pathway_code == "ALK":
            kwh, qMJ = alkaline_kwh_per_kg(electrolyzer_temp_C), 0.0
        elif pathway_code == "PEM":
            kwh, qMJ = pem_kwh_per_kg(electrolyzer_temp_C), 0.0
        elif pathway_code == "SOEC":
            kwh, qMJ = soec_split_energy(electrolyzer_temp_C)
        else:
            raise ValueError("Unknown electrolysis pathway")

        kg_h_elec = h2_from_electricity(Pe, kwh)
        if qMJ > 0:
            MJ_h = Q * 1e6 * SEC_PER_HOUR / 1e6
            kg_h_heat_cap = MJ_h / qMJ
        else:
            kg_h_heat_cap = np.inf
        h2_kg_h = float(min(kg_h_elec, kg_h_heat_cap))

        H2_MJ_h = h2_kg_h * LHV_H2_MJ_PER_KG
        eta_sys = H2_MJ_h / (Pth_total * 1e6 * SEC_PER_HOUR / 1e6) if Pth_total > 0 else 0.0

        lcoh = lcohUSD_per_kg(
            capex_USD_per_kW=capex_USD_per_kW,
            opex_frac_per_year=opex_frac,
            electricity_USD_per_MWh=price_elec_USD_per_MWh,
            heat_USD_per_MWhth=price_heat_USD_per_MWhth,
            kwh_electric_per_kg=kwh,
            MJ_heat_per_kg=qMJ,
            stack_kWe=max(Pe * 1e3, 1.0),
            cf=reac.capacity_factor,
            lifetime_years=lifetime_years,
            discount_rate=discount_rate,
        )

    else:  # thermochemical (SI)
        eta = si_cycle_efficiency(min(reac.outlet_temp_C, electrolyzer_temp_C))
        h2_kg_h = h2_from_thermal(Q, eta)
        H2_MJ_h = h2_kg_h * LHV_H2_MJ_PER_KG
        eta_sys = H2_MJ_h / (Pth_total * 1e6 * SEC_PER_HOUR / 1e6) if Pth_total > 0 else 0.0

        eq_kwh_per_kg = 40.0
        eq_kWe = max(h2_kg_h * eq_kwh_per_kg, 1.0)
        lcoh = lcohUSD_per_kg(
            capex_USD_per_kW=capex_USD_per_kW * 1.2,
            opex_frac_per_year=opex_frac,
            electricity_USD_per_MWh=price_elec_USD_per_MWh * 0.1,
            heat_USD_per_MWhth=price_heat_USD_per_MWhth,
            kwh_electric_per_kg=2.0,
            MJ_heat_per_kg=(LHV_H2_MJ_PER_KG / max(eta, 1e-9)),
            stack_kWe=eq_kWe,
            cf=reac.capacity_factor,
            lifetime_years=lifetime_years,
            discount_rate=discount_rate,
        )

    return {
        "h2_kg_per_h": h2_kg_h,
        "eta_sys": eta_sys,
        "lcoh_USD_per_kg": lcoh,
        "Pe_MWe": Pe,
        "Q_MWt": Q,
    }

# -------------------
# Tkinter UI (Single Page)
# -------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Nuclear-Enabled Hydrogen – Single Page")
        self.geometry("1350x850")

        # Default economics/ops
        self.price_elec = tk.DoubleVar(value=35.0)
        self.price_heat = tk.DoubleVar(value=10.0)
        self.capex      = tk.DoubleVar(value=900.0)
        self.opex_frac  = tk.DoubleVar(value=0.04)
        self.lifetime   = tk.IntVar(value=20)
        self.discount   = tk.DoubleVar(value=0.07)

        self.split      = tk.DoubleVar(value=0.6)
        self.proc_temp  = tk.DoubleVar(value=70.0)

        # Reactor fields (preset fills)
        self.reac_name_var = tk.StringVar(value="SMR-PWR")
        self.th_var = tk.DoubleVar(value=PRESET_REACTORS["SMR-PWR"].thermal_power_MWt)
        self.eta_var = tk.DoubleVar(value=PRESET_REACTORS["SMR-PWR"].electric_efficiency)
        self.tout_var = tk.DoubleVar(value=PRESET_REACTORS["SMR-PWR"].outlet_temp_C)
        self.cf_var   = tk.DoubleVar(value=PRESET_REACTORS["SMR-PWR"].capacity_factor)

        self.path_var = tk.StringVar(value="PEM")

        # Results state
        self._last_res = None

        # Layout: Left column (inputs + solver), Right column (results + conclusions + plot)
        self._build_layout()

    # ---------- helpers ----------
    def _reactor_from_fields(self) -> Reactor:
        return Reactor(
            name=self.reac_name_var.get().strip() or "Custom",
            thermal_power_MWt=float(self.th_var.get()),
            electric_efficiency=float(self.eta_var.get()),
            outlet_temp_C=float(self.tout_var.get()),
            capacity_factor=float(self.cf_var.get()),
        )

    def _run_simulation(self):
        reac = self._reactor_from_fields()
        res = simulate(
            reac,
            self.path_var.get(),
            float(self.split.get()),
            float(self.proc_temp.get()),
            float(self.price_elec.get()),
            float(self.price_heat.get()),
            float(self.capex.get()),
            float(self.opex_frac.get()),
            int(self.lifetime.get()),
            float(self.discount.get()),
        )
        self._last_res = res
        self._display_results(res)
        return res

    def _display_results(self, res: dict):
        self.var_h2.set(f"{res['h2_kg_per_h']:,.1f} kg/h")
        self.var_eta.set(f"{res['eta_sys']*100:,.1f} %")
        self.var_lcoh.set(f"${res['lcoh_USD_per_kg']:,.2f} /kg")
        self.var_pe.set(f"{res['Pe_MWe']:,.2f} MWe")
        self.var_q.set(f"{res['Q_MWt']:,.2f} MWt")

    def _gen_conclusions_from_current(self):
        reac = self._reactor_from_fields()
        code = self.path_var.get()
        f = float(self.split.get())
        t = float(self.proc_temp.get())
        res = simulate(
            reac, code, f, t,
            float(self.price_elec.get()), float(self.price_heat.get()),
            float(self.capex.get()), float(self.opex_frac.get()),
            int(self.lifetime.get()), float(self.discount.get())
        )
        self._last_res = res
        p = PATHWAYS[code].display
        lines = [
            f"Pathway: {p}, split f={f:.2f}, process temperature ≈ {t:.0f} °C",
            f"Hydrogen production: {res['h2_kg_per_h']:,.1f} kg/h",
            f"LCOH: ${res['lcoh_USD_per_kg']:,.2f}/kg",
            f"System LHV efficiency: {res['eta_sys']*100:,.1f} %",
        ]
        if code == "SOEC":
            lines.append("SOEC benefits from higher temperature and adequate process heat.")
        if code in ("ALK", "PEM"):
            lines.append("Low-temp electrolysis prefers higher electric split and low electricity price.")
        if code == "SI":
            lines.append("S–I needs very high temperatures; outlet ≥850–900 °C improves performance.")
        if res['lcoh_USD_per_kg'] < 2.0:
            lines.append("Very competitive (<$2/kg) under optimistic assumptions.")
        elif res['lcoh_USD_per_kg'] < 4.0:
            lines.append("Potentially competitive vs low-carbon benchmarks.")
        else:
            lines.append("Costs high; reduce CAPEX, increase CF, or improve efficiency.")
        self._set_text(self.txt_conclusion, "\n".join(lines))
        self._display_results(res)

    # ---------- UI ----------
    def _build_layout(self):
        root = ttk.Frame(self)
        root.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        root.grid_columnconfigure(0, weight=0)
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)

        # LEFT: Inputs + Solver
        left = ttk.Frame(root)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        # Reactor
        lf_reac = ttk.Labelframe(left, text="Reactor (entries are live)")
        lf_reac.pack(fill=tk.X, pady=6)
        # preset
        self.preset_var = tk.StringVar(value="SMR-PWR")
        ttk.Label(lf_reac, text="Preset").grid(row=0, column=0, sticky="w")
        cb = ttk.Combobox(lf_reac, textvariable=self.preset_var,
                          values=list(PRESET_REACTORS.keys()),
                          width=12, state="readonly")
        cb.grid(row=0, column=1, sticky="ew", pady=2)
        def on_preset(*_):
            r = PRESET_REACTORS[self.preset_var.get()]
            self.reac_name_var.set(r.name)
            self.th_var.set(r.thermal_power_MWt)
            self.eta_var.set(r.electric_efficiency)
            self.tout_var.set(r.outlet_temp_C)
            self.cf_var.set(r.capacity_factor)
        self.preset_var.trace_add("write", on_preset)

        rows = [
            ("Name", self.reac_name_var),
            ("Thermal Power (MWt)", self.th_var),
            ("Electric Efficiency (0–1)", self.eta_var),
            ("Outlet Temp (°C)", self.tout_var),
            ("Capacity Factor (0–1)", self.cf_var),
        ]
        for i, (lbl, var) in enumerate(rows, start=1):
            ttk.Label(lf_reac, text=lbl).grid(row=i, column=0, sticky="w")
            ttk.Entry(lf_reac, textvariable=var, width=12).grid(row=i, column=1, sticky="ew", pady=2)
        lf_reac.grid_columnconfigure(1, weight=1)

        # Pathway & Ops
        lf_ops = ttk.Labelframe(left, text="Pathway & Operation")
        lf_ops.pack(fill=tk.X, pady=6)
        ttk.Label(lf_ops, text="Pathway").grid(row=0, column=0, sticky="w")
        ttk.Combobox(lf_ops, textvariable=self.path_var,
                     values=list(PATHWAYS.keys()),
                     width=10, state="readonly").grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Label(lf_ops, text="Process Temp (°C)").grid(row=1, column=0, sticky="w")
        ttk.Entry(lf_ops, textvariable=self.proc_temp, width=12).grid(row=1, column=1, sticky="ew", pady=2)
        ttk.Label(lf_ops, text="Split to Electricity f").grid(row=2, column=0, sticky="w")
        ttk.Scale(lf_ops, from_=0.0, to=1.0, variable=self.split,
                  orient=tk.HORIZONTAL).grid(row=2, column=1, sticky="ew", pady=4)
        lf_ops.grid_columnconfigure(1, weight=1)

        # Economics
        lf_econ = ttk.Labelframe(left, text="Economics")
        lf_econ.pack(fill=tk.X, pady=6)
        econ = [
            ("Electricity ($/MWh)", self.price_elec),
            ("Heat ($/MWhth)", self.price_heat),
            ("CAPEX ($/kW)", self.capex),
            ("OPEX (frac/yr)", self.opex_frac),
            ("Lifetime (yr)", self.lifetime),
            ("Discount (frac)", self.discount),
        ]
        for i, (lbl, var) in enumerate(econ):
            ttk.Label(lf_econ, text=lbl).grid(row=i, column=0, sticky="w")
            ttk.Entry(lf_econ, textvariable=var, width=12).grid(row=i, column=1, sticky="ew", pady=2)
        lf_econ.grid_columnconfigure(1, weight=1)

        # Actions
        lf_act = ttk.Labelframe(left, text="Actions")
        lf_act.pack(fill=tk.X, pady=6)
        self.var_plot_update = tk.BooleanVar(value=True)
        self.var_auto_conc   = tk.BooleanVar(value=True)
        ttk.Checkbutton(lf_act, text="Update Plot on Run", variable=self.var_plot_update).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(lf_act, text="Refresh Conclusions on Run", variable=self.var_auto_conc).grid(row=1, column=0, sticky="w")
        ttk.Button(lf_act, text="Run Simulation", command=self.on_run).grid(row=2, column=0, sticky="ew", pady=4)

        # Solver
        lf_solv = ttk.Labelframe(left, text="Problem Solver (meet target at min LCOH)")
        lf_solv.pack(fill=tk.X, pady=10)
        self.target_h2 = tk.DoubleVar(value=10000.0)
        ttk.Label(lf_solv, text="Target H₂ (kg/h)").grid(row=0, column=0, sticky="w")
        ttk.Entry(lf_solv, textvariable=self.target_h2, width=12).grid(row=0, column=1, sticky="ew")
        self.var_apply_best = tk.BooleanVar(value=True)
        ttk.Checkbutton(lf_solv, text="Apply best to inputs & refresh", variable=self.var_apply_best).grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Button(lf_solv, text="Solve", command=self.solve_for_target).grid(row=2, column=0, columnspan=2, sticky="ew", pady=4)
        self.txt_solver = tk.Text(lf_solv, height=10, wrap=tk.WORD)
        self.txt_solver.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4,0))
        for c in (0,1):
            lf_solv.grid_columnconfigure(c, weight=1)

        # RIGHT: Results + Plot + Conclusions
        right = ttk.Frame(root)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        # Results
        lf_res = ttk.Labelframe(right, text="Results")
        lf_res.grid(row=0, column=0, sticky="ew", pady=(0,6))
        self.var_h2  = tk.StringVar(value="-")
        self.var_eta = tk.StringVar(value="-")
        self.var_lcoh= tk.StringVar(value="-")
        self.var_pe  = tk.StringVar(value="-")
        self.var_q   = tk.StringVar(value="-")
        items = [("Hydrogen Rate", self.var_h2),
                 ("System Efficiency (LHV)", self.var_eta),
                 ("LCOH", self.var_lcoh),
                 ("Electric Power", self.var_pe),
                 ("Thermal to Process", self.var_q)]
        for i,(lbl,var) in enumerate(items):
            ttk.Label(lf_res, text=lbl+":").grid(row=i, column=0, sticky="w", padx=4, pady=2)
            ttk.Label(lf_res, textvariable=var, font=("TkDefaultFont", 11, "bold")).grid(row=i, column=1, sticky="w")
        ttk.Button(lf_res, text="Reset Outputs", command=self.reset_outputs).grid(row=0, column=2, rowspan=2, padx=6)
        ttk.Button(lf_res, text="Export Report", command=self.export_report).grid(row=2, column=2, rowspan=2, padx=6)

        # Plot
        lf_plot = ttk.Labelframe(right, text="Plot: H₂ (solid) and LCOH (dashed) vs. split")
        lf_plot.grid(row=1, column=0, sticky="nsew")
        self.fig = Figure(figsize=(7.8, 4.6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=lf_plot)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        plot_ctrl = ttk.Frame(lf_plot)
        plot_ctrl.pack(side=tk.RIGHT, fill=tk.Y, padx=6)
        ttk.Button(plot_ctrl, text="Scan Current Pathway", command=self.scan_current).pack(fill=tk.X, pady=4)
        ttk.Button(plot_ctrl, text="Compare All Pathways", command=self.compare_all).pack(fill=tk.X, pady=4)
        ttk.Button(plot_ctrl, text="Clear Plot", command=self.clear_plot).pack(fill=tk.X, pady=4)
        ttk.Button(plot_ctrl, text="Save PNG", command=self.save_plot).pack(fill=tk.X, pady=4)

        # Conclusions
        lf_conc = ttk.Labelframe(right, text="Conclusions")
        lf_conc.grid(row=2, column=0, sticky="ew", pady=(6,0))
        self.txt_conclusion = tk.Text(lf_conc, height=7, wrap=tk.WORD)
        self.txt_conclusion.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        conc_btns = ttk.Frame(lf_conc)
        conc_btns.pack(fill=tk.X)
        ttk.Button(conc_btns, text="Generate from current inputs", command=self._gen_conclusions_from_current).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(conc_btns, text="Clear", command=lambda: self._set_text(self.txt_conclusion, "")).pack(side=tk.LEFT, padx=4, pady=4)

    # ---------- actions ----------
    def on_run(self):
        try:
            res = self._run_simulation()
            if self.var_plot_update.get():
                self._plot_scan(self.path_var.get())
            if self.var_auto_conc.get():
                self._gen_conclusions_from_current()
        except Exception as e:
            messagebox.showerror("Run error", str(e))

    def reset_outputs(self):
        for v in (self.var_h2, self.var_eta, self.var_lcoh, self.var_pe, self.var_q):
            v.set("-")
        self._last_res = None

    # ---------- plotting ----------
    def _plot_scan(self, code: str):
        fs = np.linspace(0, 1, 41)
        h2, lcoh = [], []
        reac = self._reactor_from_fields()
        for f in fs:
            r = simulate(
                reac, code, float(f), float(self.proc_temp.get()),
                float(self.price_elec.get()), float(self.price_heat.get()),
                float(self.capex.get()), float(self.opex_frac.get()),
                int(self.lifetime.get()), float(self.discount.get()))
            h2.append(r["h2_kg_per_h"])
            lcoh.append(r["lcoh_USD_per_kg"])
        self.ax.clear()
        self.ax.plot(fs, h2, label=f"H2 – {code}")
        self.ax.plot(fs, lcoh, linestyle="--", label=f"LCOH – {code}")
        self.ax.set_xlabel("Split to Electricity f")
        self.ax.set_ylabel("kg H₂ / h (solid) and $/kg (dashed)")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def scan_current(self):
        self._plot_scan(self.path_var.get())

    def compare_all(self):
        fs = np.linspace(0, 1, 41)
        self.ax.clear()
        reac = self._reactor_from_fields()
        for code in PATHWAYS.keys():
            h2, lcoh = [], []
            for f in fs:
                r = simulate(
                    reac, code, float(f), float(self.proc_temp.get()),
                    float(self.price_elec.get()), float(self.price_heat.get()),
                    float(self.capex.get()), float(self.opex_frac.get()),
                    int(self.lifetime.get()), float(self.discount.get()))
                h2.append(r["h2_kg_per_h"])
                lcoh.append(r["lcoh_USD_per_kg"])
            self.ax.plot(fs, h2, label=f"H2 – {code}")
            self.ax.plot(fs, lcoh, linestyle="--", label=f"LCOH – {code}")
        self.ax.set_xlabel("Split to Electricity f")
        self.ax.set_ylabel("kg H₂ / h (solid) and $/kg (dashed)")
        self.ax.legend(ncol=2)
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def clear_plot(self):
        self.ax.clear()
        self.canvas.draw()

    def save_plot(self):
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[["PNG", "*.png"]])
        if path:
            self.fig.savefig(path, bbox_inches="tight")
            messagebox.showinfo("Saved", f"Plot saved to {path}")

    # ---------- solver ----------
    def solve_for_target(self):
        try:
            target = float(self.target_h2.get())
        except Exception:
            messagebox.showerror("Solver error", "Invalid target H₂.")
            return
        self._set_text(self.txt_solver, "Solving...\n")
        best = None
        fs = np.linspace(0.0, 1.0, 41)
        reac = self._reactor_from_fields()

        for code in PATHWAYS.keys():
            for f in fs:
                res = simulate(
                    reac, code, float(f), float(self.proc_temp.get()),
                    float(self.price_elec.get()), float(self.price_heat.get()),
                    float(self.capex.get()), float(self.opex_frac.get()),
                    int(self.lifetime.get()), float(self.discount.get())
                )
                if res["h2_kg_per_h"] >= target:
                    score = res["lcoh_USD_per_kg"]
                    if (best is None) or (score < best["lcoh_USD_per_kg"]):
                        best = {"pathway": code, "split": f, **res}

        if best is None:
            self._set_text(self.txt_solver,
                           "No configuration met the target.\n"
                           "Increase reactor power, relax target, or pick SOEC/S–I.")
            return

        text = (
            "Best configuration meeting target (min LCOH):\n"
            f"• Pathway: {PATHWAYS[best['pathway']].display} ({best['pathway']})\n"
            f"• Split f: {best['split']:.2f}\n"
            f"• H₂: {best['h2_kg_per_h']:,.1f} kg/h\n"
            f"• LCOH: ${best['lcoh_USD_per_kg']:,.2f}/kg\n"
            f"• η(LHV): {best['eta_sys']*100:,.1f}%\n"
            f"• Electric: {best['Pe_MWe']:,.2f} MWe, Heat: {best['Q_MWt']:,.2f} MWt\n"
        )
        self._set_text(self.txt_solver, text)

        # Optionally apply best to inputs and refresh results + conclusions
        if self.var_apply_best.get():
            self.path_var.set(best["pathway"])
            self.split.set(float(best["split"]))
            # Re-run main sim and refresh plot/conclusions for immediate feedback
            self.on_run()

    # ---------- utilities ----------
    @staticmethod
    def _set_text(widget: tk.Text, text: str):
        widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.config(state=tk.NORMAL)

    def export_report(self):
        if self._last_res is None:
            messagebox.showwarning("Export", "Run a simulation first.")
            return
        # Pick base path (CSV); also save PNG if plot exists
        csv_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                               filetypes=[["CSV", "*.csv"]],
                                               title="Save report (CSV)")
        if not csv_path:
            return
        # Gather inputs
        inputs = {
            "reactor_name": self.reac_name_var.get(),
            "thermal_power_MWt": self.th_var.get(),
            "electric_efficiency": self.eta_var.get(),
            "outlet_temp_C": self.tout_var.get(),
            "capacity_factor": self.cf_var.get(),
            "pathway": self.path_var.get(),
            "process_temp_C": self.proc_temp.get(),
            "split_f": self.split.get(),
            "price_elec_USD_MWh": self.price_elec.get(),
            "price_heat_USD_MWhth": self.price_heat.get(),
            "capex_USD_kW": self.capex.get(),
            "opex_frac_per_year": self.opex_frac.get(),
            "lifetime_years": self.lifetime.get(),
            "discount_rate": self.discount.get(),
        }
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["key", "value"])
            w.writerow(["--- inputs ---",""])
            for k,v in inputs.items():
                w.writerow([k, v])
            w.writerow(["--- results ---",""])
            for k,v in self._last_res.items():
                w.writerow([k, v])

        # Also offer to save plot
        try:
            png_path = csv_path.rsplit(".",1)[0] + ".png"
            self.fig.savefig(png_path, bbox_inches="tight")
        except Exception:
            pass
        messagebox.showinfo("Export", f"Report saved to {csv_path} (plot PNG alongside if available).")

# --------------
# Entrypoint
# --------------
if __name__ == "__main__":
    App().mainloop()
