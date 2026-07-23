"""
run_pipeline.py
End to end reproducible pipeline for the MR-MOU-D-Tree framework.

Stages
  1  Sample baseline coverage scenarios via k means centroids
  2  Run the stochastic transmission model (stochastic intervention effects)
  3  Derive multiobjective cost effectiveness outcomes (per-draw, joint)
  4  Train the uncertainty aware deep ensemble metamodel with a scenario
     independent train/test split, an out-of-distribution hold-out, split
     conformal recalibration, and a suite of baseline metamodels
  5  Build district populations from real division data; compute risk adjusted
     objectives under JOINT and INDEPENDENT outcome sampling
  6  Fit the deterministic and uncertainty aware (CVaR) decision trees
  7  Bootstrap and NESTED bootstrap decision boundary stability
  8  Deterministic vs uncertainty aware comparison, flip typology, equity with
     intervals and alternative objectives, mechanistic CHW analysis
  9  Decision-rule sensitivity sweep (delta, CVaR alpha, immunity target, WTP,
     budget) and ICER tail / CVaR-alpha safeguards

Outputs land in ../figures, ../tables, ../data and ../results.
"""

import json
import os
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

import config as C
import epi_model as EPI
import epi_model_age as AGE

warnings.filterwarnings("ignore")
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
rcParams["axes.titlesize"] = 11
rcParams["axes.labelsize"] = 10
rcParams["font.size"] = 10
rcParams["figure.dpi"] = 150

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures")
TAB = os.path.join(ROOT, "tables")
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
for d in (FIG, TAB, DATA, RES):
    os.makedirs(d, exist_ok=True)

rng = np.random.default_rng(C.GLOBAL_SEED)
SUMMARY = {}
TARGETS = ["measles_averted", "daly_averted", "cost_total"]
Z90 = 1.6449


def discounted_years(L, r=C.DISCOUNT):
    if r <= 0:
        return L
    return (1.0 - (1.0 + r) ** (-L)) / r


def ens_predict(entry, Xs):
    ys = entry["yscaler"]
    stack = np.array([m.predict(Xs) for m in entry["ens"]])
    return stack * ys[1] + ys[0]


def predict_total_sd(entry, Xs, epi_sd):
    """Total predictive standard deviation of Equation 23 at ARBITRARY x.

    Equation 23 is sigma^2 = sigma^2_ep + sigma^2_al. The epistemic term is the
    ensemble spread. The aleatoric term is the replication (Monte-Carlo)
    standard error of the scenario mean, which is only observed at the training
    scenarios; `entry["al_model"]` regresses that quantity on x so it is
    available at any new district, which is what the decision stage requires.
    """
    al = entry["al_model"].predict(Xs) / np.sqrt(C.N_DRAWS)
    al = np.maximum(al, 0.0)
    return np.sqrt(epi_sd ** 2 + al ** 2)


def immunity_target(r0_adj, name=None):
    """Programmatic immunity target as a function of local R0 multiplier."""
    name = name or C.DEFAULT_TARGET
    base, slope = C.TARGET_GRID[name]
    return np.clip(base + slope * (r0_adj - 0.9), 0.86, 0.97)


# ======================================================================
# STAGE 1  Scenario sampling
# ======================================================================
def stage1_scenarios():
    print("Stage 1: sampling coverage scenarios")
    vecs = np.column_stack([
        rng.uniform(*C.MR1_RANGE, C.N_SAMPLE_VECTORS),
        rng.uniform(*C.MR2_RANGE, C.N_SAMPLE_VECTORS),
        rng.uniform(*C.SIA_RANGE, C.N_SAMPLE_VECTORS),
    ])
    ranges = np.array([C.MR1_RANGE[1] - C.MR1_RANGE[0],
                       C.MR2_RANGE[1] - C.MR2_RANGE[0],
                       C.SIA_RANGE[1] - C.SIA_RANGE[0]])
    norm = vecs / ranges

    grid_rows = []
    for k in C.CLUSTER_GRID:
        km = KMeans(n_clusters=k, random_state=C.GLOBAL_SEED, n_init=4).fit(norm)
        cen = km.cluster_centers_
        d = np.sqrt(((cen[:, None, :] - cen[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        grid_rows.append({"n_clusters": k, "min_norm_distance": float(d.min())})
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(os.path.join(TAB, "table_S1_cluster_grid.csv"), index=False)

    km = KMeans(n_clusters=C.N_CENTROIDS, random_state=C.GLOBAL_SEED, n_init=6).fit(norm)
    centroids = km.cluster_centers_ * ranges
    centroids = np.clip(centroids,
                        [C.MR1_RANGE[0], C.MR2_RANGE[0], C.SIA_RANGE[0]],
                        [C.MR1_RANGE[1], C.MR2_RANGE[1], C.SIA_RANGE[1]])
    scen = pd.DataFrame(centroids, columns=["mr1_before", "mr2_before", "sia_before"])
    scen.to_csv(os.path.join(DATA, "scenarios_centroids.csv"), index=False)

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(grid["n_clusters"], grid["min_norm_distance"], "-o", color="#c0392b")
    ax.axhline(C.DIST_THRESHOLD, ls="--", color="gray", lw=1)
    ax.text(C.CLUSTER_GRID[1], C.DIST_THRESHOLD + 0.001, "target 0.025", fontsize=8, color="gray")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Minimum normalized distance")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig01_cluster_distance.png"), bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.2))
    names = ["MR1 before", "MR2 before", "SIA before"]
    for i, nm in enumerate(names):
        r, c = divmod(i, 2)
        axs[r, c].hist(centroids[:, i], bins=25, color="#5d6d7e", edgecolor="white")
        axs[r, c].set_title(nm)
        axs[r, c].set_xlabel("coverage")
    axs[1, 1].scatter(centroids[:, 0], centroids[:, 2], s=8, alpha=0.5, color="#c0392b")
    axs[1, 1].set_title("MR1 before vs SIA before")
    axs[1, 1].set_xlabel("MR1 before")
    axs[1, 1].set_ylabel("SIA before")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig02_centroid_distributions.png"), bbox_inches="tight")
    plt.close(fig)

    SUMMARY["n_centroids"] = int(C.N_CENTROIDS)
    SUMMARY["min_distance_selected"] = float(grid.loc[grid.n_clusters == C.N_CENTROIDS,
                                                      "min_norm_distance"].values[0]) \
        if C.N_CENTROIDS in C.CLUSTER_GRID else float(grid["min_norm_distance"].min())
    return scen, grid


# ======================================================================
# STAGE 2 and 3  Simulation and multiobjective outcomes (joint aware)
# ======================================================================
def stage2_simulate(scen, si0=0, si1=None, acc=None):
    """Run the age structured simulations for scenarios [si0, si1).

    The age structured model is roughly three times the cost of the single cohort
    model it replaces and there is now a fourth intervention, so this stage is
    chunked and resumable; `acc` carries the accumulated rows between chunks.
    """
    print("Stage 2: running transmission simulations (stochastic effects)")
    L_death = discounted_years(C.LIFE_EXPECTANCY - C.MEAN_AGE_MEASLES_DEATH)
    dur_crs_disc = discounted_years(C.DUR_CRS)

    if acc is None:
        acc = {"rows": [], "age_rows": [], "draw_records": {},
               "corr_records": {iv: [] for iv in C.INTERVENTIONS}}
    rows = acc["rows"]; age_rows = acc["age_rows"]
    draw_records = acc["draw_records"]; corr_records = acc["corr_records"]
    si1 = len(scen) if si1 is None else min(si1, len(scen))

    for si in range(si0, si1):
        mr1b = scen.mr1_before.iloc[si]
        mr2b = scen.mr2_before.iloc[si]
        siab = scen.sia_before.iloc[si]
        params = EPI.draw_parameters(rng, C.N_DRAWS)
        eff = EPI.draw_effect_sizes(rng, C.N_DRAWS)      # STOCHASTIC effectiveness

        m_base, dose_base, base_age = AGE.simulate_age(
            rng, np.full(C.N_DRAWS, mr1b), np.full(C.N_DRAWS, mr2b),
            np.full(C.N_DRAWS, siab), params, "measles", return_age_incidence=True)
        r_base, _ = AGE.simulate_age(rng, np.full(C.N_DRAWS, mr1b),
                                     np.full(C.N_DRAWS, mr2b),
                                     np.full(C.N_DRAWS, siab), params, "rubella")
        # age distribution of the baseline epidemic, retained for validation
        u9, u2, u5 = AGE.age_distribution_of_cases(base_age)
        age_rows.append({"scenario": si, "mr1_before": mr1b,
                         "share_under9mo": float(u9.mean()),
                         "share_under2y": float(u2.mean()),
                         "share_under5y": float(u5.mean())})

        for iv in C.INTERVENTIONS:
            mr1d, siad = EPI.effectiveness(iv, mr1b, siab, eff)
            mr1d = np.full(C.N_DRAWS, float(mr1d)) if np.ndim(mr1d) == 0 else mr1d
            siad = np.full(C.N_DRAWS, float(siad)) if np.ndim(siad) == 0 else siad
            # The age lowered schedule acts through the simulator's policy, not
            # through the coverage vector; every other intervention acts through
            # coverage and leaves the schedule alone.
            pol = "six_month" if iv == "mr_six_month" else "routine"
            m_iv, dose_iv = AGE.simulate_age(rng, mr1d, np.full(C.N_DRAWS, mr2b), siad,
                                             params, "measles", policy=pol)
            r_iv, _ = AGE.simulate_age(rng, mr1d, np.full(C.N_DRAWS, mr2b), siad,
                                       params, "rubella", policy=pol)

            measles_averted = np.maximum(m_base - m_iv, 0.0)
            rubella_averted = np.maximum(r_base - r_iv, 0.0)
            crs_averted = rubella_averted * C.FRAC_RUBELLA_IN_WCBA_FIRST_TRI * C.CRS_PER_RUBELLA_WCBA
            extra_doses = np.maximum(dose_iv - dose_base, 0.0)

            deaths_averted = measles_averted * (0.8 * C.CFR_MEASLES_U5 + 0.2 * C.CFR_MEASLES_5PLUS)
            yll = deaths_averted * L_death
            yld_measles = measles_averted * C.DW_MEASLES * C.DUR_MEASLES
            daly_crs = crs_averted * C.DW_CRS * dur_crs_disc
            daly_averted = yll + yld_measles + daly_crs

            delivery = C.COST_DELIVERY_SIA if iv == "sia_campaign" else C.COST_DELIVERY_ROUTINE
            cost_total = (C.C_FIXED[iv] * C.POP_UNIT
                          + extra_doses * (C.COST_MR_DOSE + delivery))

            draw_records[(si, iv)] = {
                "measles_averted": measles_averted,
                "crs_averted": crs_averted,
                "daly_averted": daly_averted,
                "cost_total": cost_total,
                "extra_doses": extra_doses,
            }
            if daly_averted.std() > 1e-9 and cost_total.std() > 1e-9:
                corr_records[iv].append(float(np.corrcoef(cost_total, daly_averted)[0, 1]))

            rows.append({
                "scenario": si, "mr1_before": mr1b, "mr2_before": mr2b,
                "sia_before": siab, "intervention": iv,
                "measles_averted_mean": measles_averted.mean(),
                "measles_averted_sd": measles_averted.std(),
                "crs_averted_mean": crs_averted.mean(),
                "daly_averted_mean": daly_averted.mean(),
                "daly_averted_sd": daly_averted.std(),
                "cost_total_mean": cost_total.mean(),
                "cost_total_sd": cost_total.std(),
                "cost_daly_corr": (float(np.corrcoef(cost_total, daly_averted)[0, 1])
                                   if daly_averted.std() > 1e-9 and cost_total.std() > 1e-9 else 0.0),
                "cost_per_daly_mean": (cost_total / np.maximum(daly_averted, 1e-6)).mean(),
                "cost_per_measles_mean": (cost_total / np.maximum(measles_averted, 1e-6)).mean(),
            })
        if (si + 1) % 50 == 0:
            print(f"   simulated {si + 1}/{len(scen)} scenarios", flush=True)

    if si1 < len(scen):
        return acc, False          # more chunks to come

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "simulation_outcomes.csv"), index=False)
    pd.DataFrame(age_rows).to_csv(os.path.join(DATA, "age_distribution_by_scenario.csv"),
                                  index=False)

    corr_summary = {iv: (float(np.mean(v)), float(np.std(v))) for iv, v in corr_records.items()}
    SUMMARY["cost_daly_corr"] = {C.INTERVENTION_LABELS[iv]: round(corr_summary[iv][0], 3)
                                 for iv in C.INTERVENTIONS}
    SUMMARY["_stage2_complete"] = True

    # Figures 3-5: effectiveness curves with UNCERTAINTY BANDS (stochastic effects)
    grid_b = np.linspace(0.40, 0.97, 60)
    ve1, ve2, vesia = C.VE1_MEAN, C.VE2_MEAN, C.VE_SIA_MEAN
    fig, axs = plt.subplots(1, len(C.INTERVENTIONS), figsize=(3.5 * len(C.INTERVENTIONS), 3.2))
    for ax, iv, col in zip(axs, C.INTERVENTIONS, ["#2980b9", "#27ae60", "#c0392b", "#8e44ad"]):
        base_imm = EPI.immune_fraction(grid_b, 0.75 * grid_b, np.full_like(grid_b, 0.10),
                                       ve1, ve2, vesia)
        post_lo, post_mid, post_hi = [], [], []
        for b in grid_b:
            reps = 200
            eff = EPI.draw_effect_sizes(rng, reps)
            m1d, siad = EPI.effectiveness(iv, b, 0.10, eff)
            m1d = np.full(reps, float(m1d)) if np.ndim(m1d) == 0 else m1d
            siad = np.full(reps, float(siad)) if np.ndim(siad) == 0 else siad
            imm = EPI.immune_fraction(m1d, np.full(reps, 0.75 * b), siad, ve1, ve2, vesia)
            post_lo.append(np.percentile(imm, 5))
            post_mid.append(np.mean(imm))
            post_hi.append(np.percentile(imm, 95))
        post_lo, post_mid, post_hi = map(np.array, (post_lo, post_mid, post_hi))
        ax.plot(grid_b, base_imm, ls="--", color="gray", lw=1, label="no intervention")
        ax.plot(grid_b, post_mid, color=col, lw=2, label="with intervention")
        ax.fill_between(grid_b, post_lo, post_hi, color=col, alpha=0.18)
        ax.fill_between(grid_b, base_imm, post_mid, color=col, alpha=0.06)
        ax.set_xlabel("MR1 before")
        ax.set_ylabel("Effective immunity")
        ax.set_ylim(0.3, 1.0)
        ax.set_title(C.INTERVENTION_LABELS[iv])
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig03_05_effectiveness_curves.png"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for cov, col, lab in [(0.55, "#c0392b", "MR1 = 0.55 (low)"),
                          (0.80, "#e67e22", "MR1 = 0.80"),
                          (0.95, "#27ae60", "MR1 = 0.95 (high)")]:
        params = EPI.draw_parameters(rng, C.N_DRAWS)
        traj = _trajectory(rng, cov, 0.7 * cov, 0.1, params)
        ax.plot(np.arange(len(traj)) / 52.0, traj, color=col, lw=2, label=lab)
    ax.set_xlabel("Year")
    ax.set_ylabel("Weekly measles incidence per 100k")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig06_transmission_dynamics.png"), bbox_inches="tight")
    plt.close(fig)

    return df, draw_records, corr_summary


def _sim_incidence_per_million(rho, urban_frac, mr1, mr2, weeks, n=None, seed=0):
    """Simulated measles incidence per million over the observation window for a
    division with transmission scaling rho and urbanicity urban_frac."""
    n = n or max(8, C.N_DRAWS // 4)
    r = np.random.default_rng(seed)
    params = EPI.draw_parameters(r, n)
    inf, _ = AGE.simulate_age(r, np.full(n, mr1), np.full(n, mr2), np.zeros(n),
                              params, "measles", r0_mult=rho, urban_frac=urban_frac,
                              horizon_weeks=weeks)
    return float(np.mean(inf) / C.POP_UNIT * 1e6)


def stage3_calibrate():
    """Calibrate the model to the 2026 outbreak, then validate it on data that
    were not used to fit it.

    Two calibrations happen here and they answer different questions.

    First, the division transmission scaling. The earlier version of this work
    ASSUMED a transmission multiplier from urbanicity and incidence and then
    reported the assumption as if it were data. Here each division's scaling is
    FITTED so the model reproduces that division's observed measles incidence per
    million over 15 March to 14 April 2026. The incidence data therefore enter as
    a calibration target, which is what they can legitimately be, rather than as a
    covariate standing in for transmissibility.

    Second, the age structure. The natural infection hazard and the maternal
    antibody waning rate are calibrated to two reported quantities, the share of
    cases under nine months and the share under five years. The share under two
    years is deliberately NOT fitted and is reported as a held out check.
    """
    print("Stage 3: calibrating to the 2026 outbreak and validating")
    names = list(C.DIVISIONS_REAL.keys())
    obs_all = np.array([C.DIVISIONS_REAL[d][3] for d in names], float)
    urb_all = np.array([C.DIVISIONS_REAL[d][1] for d in names], float)
    pop_all = np.array([C.DIVISIONS_REAL[d][0] for d in names], float)

    # What is identifiable here, and what is not. The reported figures are
    # laboratory confirmed incidence, and only a fraction of true infections is
    # confirmed, so the ABSOLUTE level of observed incidence cannot be matched
    # without a reporting fraction that these data do not identify. The model
    # does not need it: the division multiplier is normalized to a population
    # weighted mean of one, so only the CROSS DIVISION CONTRAST enters. We
    # therefore map the observed pattern onto the model's own incidence scale and
    # calibrate the contrast, and we do not claim to estimate a reporting rate.
    sim_at_one = np.array([_sim_incidence_per_million(1.0, u, C.MR1_NATIONAL, C.MR2_NATIONAL,
                                                      C.CALIB_WINDOW_WEEKS, seed=99)
                           for u in urb_all])
    scale_map = float(np.average(sim_at_one, weights=pop_all) /
                      np.average(obs_all, weights=pop_all))
    SUMMARY["calibration_scale_factor"] = round(scale_map, 3)
    targets = obs_all * scale_map

    rows = []
    rho_fit = {}
    for i, d in enumerate(names):
        lo, hi = 0.05, 6.0
        for _ in range(22):
            mid = 0.5 * (lo + hi)
            sim = _sim_incidence_per_million(mid, urb_all[i], C.MR1_NATIONAL, C.MR2_NATIONAL,
                                             C.CALIB_WINDOW_WEEKS, seed=100 + i)
            if sim < targets[i]:
                lo = mid
            else:
                hi = mid
        rho = 0.5 * (lo + hi)
        sim = _sim_incidence_per_million(rho, urb_all[i], C.MR1_NATIONAL, C.MR2_NATIONAL,
                                         C.CALIB_WINDOW_WEEKS, seed=100 + i)
        rho_fit[d] = float(rho)
        rows.append({"division": d, "urban_frac": urb_all[i],
                     "observed_incidence_per_million": obs_all[i],
                     "target_on_model_scale": round(float(targets[i]), 1),
                     "simulated_incidence_per_million": round(sim, 1),
                     "calibrated_r0_mult": round(rho, 3)})
    cal = pd.DataFrame(rows)
    cal.to_csv(os.path.join(TAB, "table_calibration_divisions.csv"), index=False)
    # publish the fitted scalings so DIVISIONS is rebuilt from calibration rather
    # than from the urbanicity/incidence blend
    C.CALIBRATED_R0_MULT = rho_fit
    with open(os.path.join(DATA, "calibrated_r0_mult.json"), "w") as f:
        json.dump(rho_fit, f, indent=1)
    C.DIVISIONS = C._division_derived("calibrated")
    obs_v = cal["target_on_model_scale"].values
    sim_v = cal["simulated_incidence_per_million"].values
    SUMMARY["calibration_division_r2"] = float(r2_score(obs_v, sim_v))
    SUMMARY["calibration_division_max_abs_err"] = float(np.max(np.abs(obs_v - sim_v)))
    SUMMARY["calibrated_r0_mult"] = {d: round(v, 3) for d, v in rho_fit.items()}

    # ---- age distribution validation ----
    n = max(16, C.N_DRAWS)
    r = np.random.default_rng(4242)
    params = EPI.draw_parameters(r, n)
    _, _, agei = AGE.simulate_age(r, np.full(n, C.MR1_NATIONAL), np.full(n, C.MR2_NATIONAL),
                                  np.zeros(n), params, "measles",
                                  return_age_incidence=True, horizon_weeks=52)
    u9, u2, u5 = AGE.age_distribution_of_cases(agei)
    val = pd.DataFrame([
        {"quantity": "Share of cases under 9 months", "observed": C.OBS_AGE_SHARE_UNDER9MO,
         "simulated": round(float(u9.mean()), 3), "role": "calibration target"},
        {"quantity": "Share of cases under 2 years", "observed": C.OBS_AGE_SHARE_UNDER2Y,
         "simulated": round(float(u2.mean()), 3), "role": "held out"},
        {"quantity": "Share of cases under 5 years", "observed": C.OBS_AGE_SHARE_UNDER5Y,
         "simulated": round(float(u5.mean()), 3), "role": "calibration target"},
    ])
    val.to_csv(os.path.join(TAB, "table_age_validation.csv"), index=False)
    SUMMARY["age_validation"] = {
        "under9mo_sim": round(float(u9.mean()), 3), "under9mo_obs": C.OBS_AGE_SHARE_UNDER9MO,
        "under2y_sim": round(float(u2.mean()), 3), "under2y_obs": C.OBS_AGE_SHARE_UNDER2Y,
        "under5y_sim": round(float(u5.mean()), 3), "under5y_obs": C.OBS_AGE_SHARE_UNDER5Y,
        "under2y_held_out_error": round(float(u2.mean()) - C.OBS_AGE_SHARE_UNDER2Y, 3)}

    # ---- calibration figure ----
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.4))
    ax = axs[0]
    ax.scatter(obs_v, sim_v, s=42, color="#2980b9", zorder=3)
    lim = [0, max(obs_v.max(), sim_v.max()) * 1.15]
    ax.plot(lim, lim, "k--", lw=1)
    for i, d in enumerate(names):
        ax.annotate(d, (obs_v[i], sim_v[i]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Observed incidence, mapped to the model scale")
    ax.set_ylabel("Simulated incidence per million")
    ax.set_title("Division calibration (contrast)", fontsize=9)
    ax = axs[1]
    q = ["<9mo", "<2y", "<5y"]
    obs_a = [C.OBS_AGE_SHARE_UNDER9MO, C.OBS_AGE_SHARE_UNDER2Y, C.OBS_AGE_SHARE_UNDER5Y]
    sim_a = [float(u9.mean()), float(u2.mean()), float(u5.mean())]
    x = np.arange(3)
    ax.bar(x - 0.18, obs_a, 0.36, label="observed (WHO)", color="#34495e")
    ax.bar(x + 0.18, sim_a, 0.36, label="simulated", color="#e67e22")
    ax.set_xticks(x); ax.set_xticklabels(q)
    ax.set_ylabel("Share of cases")
    ax.set_title("Age distribution (centre bar is held out)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig19_calibration.png"), dpi=200)
    plt.close(fig)
    print(f"   division calibration R2 = {SUMMARY['calibration_division_r2']:.3f}")
    print(f"   age validation: u9mo {u9.mean():.3f}/{C.OBS_AGE_SHARE_UNDER9MO} "
          f"u2y(held out) {u2.mean():.3f}/{C.OBS_AGE_SHARE_UNDER2Y} "
          f"u5y {u5.mean():.3f}/{C.OBS_AGE_SHARE_UNDER5Y}")
    return rho_fit, cal, val


def stage3b_clustering_diagnostic():
    """Ask whether within district clustering of unvaccinated children can
    reconcile the model with BOTH quantities the surveillance reports.

    The main model matches the reported age distribution but implies that about
    53 percent of cases are zero dose, against a reported 72 percent. That gap is
    real and has an obvious candidate explanation: the model spreads unvaccinated
    children evenly across a district, whereas they cluster in the same
    neighbourhoods, and clustering raises their attack rate above their
    population share. This stage implements that mechanism, splitting each
    district into a reached and an under-reached stratum with assortative mixing,
    and reports what it does to all four reported quantities at once.
    """
    print("Stage 3b: clustering diagnostic")
    n = max(16, C.N_DRAWS)
    rows = []
    configs = [("No clustering (main model)", False, 0.0),
               ("Clustering", True, 0.0),
               ("Clustering and cohort coverage collapse", True, 0.35)]
    saved = (C.CLUSTER_ENABLED, C.COHORT_COLLAPSE)
    for tag, enabled, collapse in configs:
        C.CLUSTER_ENABLED, C.COHORT_COLLAPSE = enabled, collapse
        r = np.random.default_rng(4242)
        params = EPI.draw_parameters(r, n)
        out = AGE.simulate_age(r, np.full(n, C.MR1_NATIONAL), np.full(n, C.MR2_NATIONAL),
                               np.zeros(n), params, "measles", return_age_incidence=True,
                               return_dose_status=True, horizon_weeks=52)
        agei, zd = out[2], out[3]
        u9, u2, u5 = AGE.age_distribution_of_cases(agei)
        rows.append({"model": tag,
                     "zero_dose_share_of_cases": round(float(zd.mean()), 3),
                     "share_under9mo": round(float(u9.mean()), 3),
                     "share_under2y": round(float(u2.mean()), 3),
                     "share_under5y": round(float(u5.mean()), 3)})
    C.CLUSTER_ENABLED, C.COHORT_COLLAPSE = saved
    rows.append({"model": "Observed (WHO; Kamrujiaman et al. 2026)",
                 "zero_dose_share_of_cases": C.OBS_ZERO_DOSE_SHARE_OF_CASES,
                 "share_under9mo": C.OBS_AGE_SHARE_UNDER9MO,
                 "share_under2y": C.OBS_AGE_SHARE_UNDER2Y,
                 "share_under5y": C.OBS_AGE_SHARE_UNDER5Y})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "table_clustering_diagnostic.csv"), index=False)
    SUMMARY["clustering_diagnostic"] = {r["model"]: {k: v for k, v in r.items() if k != "model"}
                                        for r in rows}
    print(df.to_string(index=False))
    return df


def _trajectory(rng, mr1, mr2, sia, params):
    n = C.N_DRAWS
    N = C.POP_UNIT
    imm = EPI.immune_fraction(np.full(n, mr1), np.full(n, mr2), np.full(n, sia),
                              params["ve1"], params["ve2"], params["vesia"])
    S = np.maximum(N * (1 - imm), 1.0)
    E = np.zeros(n); I = np.full(n, float(C.SEED_INFECTIVES)); R = N - S - I
    beta = params["R0m"] / C.INFECTIOUS_WEEKS
    sigma = 1 / C.LATENT_WEEKS; gamma = 1 / C.INFECTIOUS_WEEKS
    inc = []
    for _ in range(C.HORIZON_WEEKS):
        lam = beta * I / N
        nE = rng.binomial(np.maximum(S, 0).astype(int), np.clip(1 - np.exp(-lam), 0, 1))
        nI = rng.binomial(np.maximum(E, 0).astype(int), 1 - np.exp(-sigma))
        nR = rng.binomial(np.maximum(I, 0).astype(int), 1 - np.exp(-gamma))
        S = np.maximum(S - nE, 0); E = E + nE - nI; I = I + nI - nR; R = R + nR
        inc.append(nI.mean())
    return np.array(inc)


# ======================================================================
# STAGE 4  Deep ensemble metamodel + train/test/OOD + conformal + baselines
# ======================================================================
def _train_ensemble(Xtr, ytr_scaled, yscaler, n_members, sd_tr=None):
    """Train the deep ensemble. `sd_tr` is the per-scenario replication standard
    deviation of the target on the SAME training scenarios; a random forest is
    fitted to it so that the aleatoric term of Equation 23 can be evaluated at
    districts that were never simulated."""
    ens = []
    for k in range(n_members):
        idx = rng.integers(0, len(Xtr), len(Xtr))
        mlp = MLPRegressor(hidden_layer_sizes=(48, 32), activation="relu",
                           max_iter=1200, random_state=k, alpha=1e-3,
                           early_stopping=False)
        mlp.fit(Xtr[idx], ytr_scaled[idx])
        ens.append(mlp)
    al_model = RandomForestRegressor(n_estimators=120, min_samples_leaf=3,
                                     random_state=0, n_jobs=-1)
    if sd_tr is None:
        sd_tr = np.zeros(len(Xtr))
    al_model.fit(Xtr, sd_tr)
    return {"ens": ens, "yscaler": yscaler, "al_model": al_model}


def stage4_metamodel(scen, draw_records):
    print("Stage 4: training deep ensemble + baselines with train/test/OOD split")
    X = scen[["mr1_before", "mr2_before", "sia_before"]].values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # Scenario-independent split: an OOD hold-out (high MR1 extrapolation region)
    # plus random calibration and test splits of the remaining in-distribution
    # scenarios. The calibration split is used only to fit split-conformal
    # factors; PICP before/after recalibration is evaluated on the disjoint
    # in-distribution test split, and OOD coverage is reported separately.
    # Two DISTINCT splits, because the shipped design conflated two different
    # questions and answered the wrong one for the deployed model.
    #
    #   DEPLOYMENT split  - a random train/calibration/test partition over the
    #       FULL coverage design. This trains and calibrates the metamodel that
    #       the decision stage actually uses. The scenario design already spans
    #       MR1 in [0.40, 0.98] and the high coverage scenarios were simulated,
    #       so withholding them from the deployed model made it extrapolate into
    #       a region where roughly a third of the sampled districts live. There
    #       is no reason to pay that cost: the data exist.
    #
    #   EXTRAPOLATION diagnostic - trained only on MR1 < cutoff and evaluated at
    #       MR1 >= cutoff. This is a genuine and informative probe of how the
    #       metamodel behaves outside its training range, so it is retained and
    #       reported, but it is a DIAGNOSTIC and no longer the model that makes
    #       recommendations.
    split_rng = np.random.default_rng(C.SPLIT_SEED)
    ood_mask = scen["mr1_before"].values >= C.OOD_MR1_CUTOFF
    in_idx = np.where(~ood_mask)[0]
    ood_idx = np.where(ood_mask)[0]

    perm_all = split_rng.permutation(len(scen))
    n_test = int(round(C.TEST_FRACTION * len(perm_all)))
    n_calib = int(round(C.CALIB_FRACTION * len(perm_all)))
    test_idx = perm_all[:n_test]
    calib_idx = perm_all[n_test:n_test + n_calib]
    train_idx = perm_all[n_test + n_calib:]

    # diagnostic split: train strictly below the cutoff
    perm_in = split_rng.permutation(in_idx)
    diag_train_idx = perm_in[int(round(0.25 * len(perm_in))):]

    SUMMARY["split"] = {"n_train": int(len(train_idx)), "n_calib": int(len(calib_idx)),
                        "n_test": int(len(test_idx)),
                        "design": "deployment split covers the full MR1 design range",
                        "n_extrapolation_probe": int(len(ood_idx)),
                        "extrapolation_rule": f"diagnostic model trained on MR1 < {C.OOD_MR1_CUTOFF}, "
                                              f"evaluated at MR1 >= {C.OOD_MR1_CUTOFF}",
                        "n_diag_train": int(len(diag_train_idx)),
                        "split_seed": C.SPLIT_SEED, "global_seed": C.GLOBAL_SEED,
                        "n_ensemble": C.N_ENSEMBLE, "n_draws": C.N_DRAWS}

    models = {}
    aleatoric = {}
    acc_rows = []
    calib_rows = []
    conformal_q = {}     # (iv,tgt,nominal) -> multiplicative conformal factor
    baseline_rows = []

    for iv in C.INTERVENTIONS:
        for tgt in TARGETS:
            y = np.array([draw_records[(si, iv)][tgt].mean() for si in range(len(scen))])
            y_sd = np.array([draw_records[(si, iv)][tgt].std() for si in range(len(scen))])
            aleatoric[(iv, tgt)] = y_sd
            ymean, ystd = y[train_idx].mean(), y[train_idx].std() + 1e-9
            y_scaled = (y - ymean) / ystd

            entry = _train_ensemble(Xs[train_idx], y_scaled[train_idx], (ymean, ystd),
                                    C.N_ENSEMBLE, sd_tr=y_sd[train_idx])
            models[(iv, tgt)] = entry

            pred_all = ens_predict(entry, Xs)
            mean_all = pred_all.mean(0)
            epi_all = pred_all.std(0)

            r2_tr = r2_score(y[train_idx], mean_all[train_idx])
            r2_te = r2_score(y[test_idx], mean_all[test_idx])
            rmse_te = np.sqrt(mean_squared_error(y[test_idx], mean_all[test_idx]))

            # Extrapolation diagnostic: a SEPARATE ensemble that never sees the
            # high coverage scenarios, scored where it must extrapolate. The
            # deployed ensemble above is trained across the full design, so this
            # number now measures extrapolation rather than describing the model
            # that issues recommendations.
            diag = _train_ensemble(Xs[diag_train_idx], y_scaled[diag_train_idx],
                                   (ymean, ystd), max(4, C.N_ENSEMBLE // 3),
                                   sd_tr=y_sd[diag_train_idx])
            diag_pred = ens_predict(diag, Xs[ood_idx]).mean(0)
            r2_ood = r2_score(y[ood_idx], diag_pred) if len(ood_idx) > 2 else float("nan")
            # in-region accuracy of the DEPLOYED model at the same high coverage
            # scenarios, which is what actually governs the recommendations there
            hi_te = np.intersect1d(test_idx, ood_idx)
            r2_hi_deploy = (r2_score(y[hi_te], mean_all[hi_te]) if len(hi_te) > 2 else float("nan"))
            # bootstrap CI for the test R2: the test split has only ~87 scenarios,
            # so a point estimate alone overstates the precision of the comparison
            bs = []
            for _ in range(400):
                bi = rng.integers(0, len(test_idx), len(test_idx))
                if np.std(y[test_idx][bi]) < 1e-9:
                    continue
                bs.append(r2_score(y[test_idx][bi], mean_all[test_idx][bi]))
            r2_lo, r2_hi = (np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan))
            acc_rows.append({"intervention": C.INTERVENTION_LABELS[iv], "outcome": tgt,
                             "R2_train": round(r2_tr, 3), "R2_test": round(r2_te, 3),
                             "R2_test_lo": round(float(r2_lo), 3), "R2_test_hi": round(float(r2_hi), 3),
                             "R2_test_highcov_deployed": (round(r2_hi_deploy, 3)
                                                          if r2_hi_deploy == r2_hi_deploy else "NA"),
                             "R2_extrapolation_diag": round(r2_ood, 3) if r2_ood == r2_ood else "NA",
                             "RMSE_test": round(rmse_te, 2), "mean_target": round(y.mean(), 2),
                             # Programme cost is dominated by a fixed per-child component and varies
                             # by only about one percent across scenarios, so R2, a
                             # variance-explained metric, is uninformative for it. Relative RMSE is
                             # reported alongside so accuracy can be read on a scale that does not
                             # collapse when the target is nearly constant.
                             "rel_RMSE_test": round(float(rmse_te / max(abs(y.mean()), 1e-9)), 4)})

            # Total predictive sd EXACTLY as in Equation 23: epistemic ensemble
            # spread plus the aleatoric replication standard error of the
            # scenario mean. No ad-hoc variance floor is added; any residual
            # miscalibration is corrected by the split-conformal factor below,
            # which is the component that carries the coverage guarantee.
            tot_sd = np.sqrt(epi_all ** 2 + (y_sd / np.sqrt(C.N_DRAWS)) ** 2)

            # Split-conformal recalibration: fit the multiplicative factor on the
            # dedicated CALIBRATION split, then report empirical coverage before
            # and after recalibration on the disjoint in-distribution TEST split.
            # Out-of-distribution coverage is reported separately (raw only).
            n_cal = len(calib_idx)
            for nominal, z in [(0.80, 1.2816), (0.90, 1.6449), (0.95, 1.9600)]:
                # raw (uncalibrated) coverage on the in-distribution test split
                lo = mean_all[test_idx] - z * tot_sd[test_idx]
                hi = mean_all[test_idx] + z * tot_sd[test_idx]
                picp_raw = float(np.mean((y[test_idx] >= lo) & (y[test_idx] <= hi)))
                mpiw_raw = float(np.mean(hi - lo))

                # split-conformal factor from calibration nonconformity scores,
                # with finite-sample correction ceil((n+1)(1-alpha))/n
                nc = np.abs(y[calib_idx] - mean_all[calib_idx]) / (tot_sd[calib_idx] + 1e-9)
                level = min(1.0, np.ceil((n_cal + 1) * nominal) / n_cal)
                q = float(np.quantile(nc, level))
                conformal_q[(iv, tgt, nominal)] = q
                lo_c = mean_all[test_idx] - q * tot_sd[test_idx]
                hi_c = mean_all[test_idx] + q * tot_sd[test_idx]
                picp_cal = float(np.mean((y[test_idx] >= lo_c) & (y[test_idx] <= hi_c)))
                mpiw_cal = float(np.mean(hi_c - lo_c))

                # Interval coverage of the DEPLOYED model restricted to the high
                # coverage test scenarios. This is the number that matters for
                # the roughly one third of districts that sit above the cutoff;
                # the previous column applied the conformal factor to scenarios
                # the model had been blinded to, which measured extrapolation
                # failure rather than the intervals in use.
                if len(hi_te) > 2:
                    lo_o = mean_all[hi_te] - q * tot_sd[hi_te]
                    hi_o = mean_all[hi_te] + q * tot_sd[hi_te]
                    picp_hi = float(np.mean((y[hi_te] >= lo_o) & (y[hi_te] <= hi_o)))
                else:
                    picp_hi = float("nan")

                calib_rows.append({"intervention": C.INTERVENTION_LABELS[iv], "outcome": tgt,
                                   "nominal": nominal,
                                   "PICP_raw": round(picp_raw, 3), "MPIW_raw": round(mpiw_raw, 2),
                                   "PICP_conformal": round(picp_cal, 3), "MPIW_conformal": round(mpiw_cal, 2),
                                   "PICP_highcov_test": (round(picp_hi, 3) if picp_hi == picp_hi else "NA")})

            # ---- Baseline metamodels on the same split (reviewer #10) ----
            for bname, model in _baseline_models():
                model.fit(Xs[train_idx], y[train_idx])
                if bname == "GP_kriging":
                    mu_te, sd_te = model.predict(Xs[test_idx], return_std=True)
                    lo = mu_te - Z90 * sd_te; hi = mu_te + Z90 * sd_te
                    picp = float(np.mean((y[test_idx] >= lo) & (y[test_idx] <= hi)))
                elif bname == "QuantileRF":
                    mu_te = model.predict(Xs[test_idx])
                    # per-tree predictions give a predictive distribution
                    tree_preds = np.array([t.predict(Xs[test_idx]) for t in model.estimators_])
                    lo = np.percentile(tree_preds, 5, axis=0)
                    hi = np.percentile(tree_preds, 95, axis=0)
                    picp = float(np.mean((y[test_idx] >= lo) & (y[test_idx] <= hi)))
                else:
                    mu_te = model.predict(Xs[test_idx])
                    picp = float("nan")
                r2b = r2_score(y[test_idx], mu_te)
                rmseb = np.sqrt(mean_squared_error(y[test_idx], mu_te))
                baseline_rows.append({"intervention": C.INTERVENTION_LABELS[iv], "outcome": tgt,
                                      "model": bname, "R2_test": round(r2b, 3),
                                      "RMSE_test": round(rmseb, 2),
                                      "PICP90": (round(picp, 3) if picp == picp else "NA")})

    # add the deep ensemble to the baseline comparison for a like-for-like row
    for iv in C.INTERVENTIONS:
        for tgt in TARGETS:
            entry = models[(iv, tgt)]
            pred_all = ens_predict(entry, Xs)
            mean_all = pred_all.mean(0); epi_all = pred_all.std(0)
            y = np.array([draw_records[(si, iv)][tgt].mean() for si in range(len(scen))])
            y_sd = np.array([draw_records[(si, iv)][tgt].std() for si in range(len(scen))])
            tot_sd = np.sqrt(epi_all ** 2 + (y_sd / np.sqrt(C.N_DRAWS)) ** 2)
            q = conformal_q[(iv, tgt, 0.90)]
            lo = mean_all[test_idx] - q * tot_sd[test_idx]; hi = mean_all[test_idx] + q * tot_sd[test_idx]
            picp = float(np.mean((y[test_idx] >= lo) & (y[test_idx] <= hi)))
            r2b = r2_score(y[test_idx], mean_all[test_idx])
            rmseb = np.sqrt(mean_squared_error(y[test_idx], mean_all[test_idx]))
            baseline_rows.append({"intervention": C.INTERVENTION_LABELS[iv], "outcome": tgt,
                                  "model": "DeepEnsemble", "R2_test": round(r2b, 3),
                                  "RMSE_test": round(rmseb, 2), "PICP90": round(picp, 3)})

    acc = pd.DataFrame(acc_rows)
    calib = pd.DataFrame(calib_rows)
    baseline = pd.DataFrame(baseline_rows)
    acc.to_csv(os.path.join(TAB, "table_metamodel_accuracy.csv"), index=False)
    calib.to_csv(os.path.join(TAB, "table_calibration.csv"), index=False)
    baseline.to_csv(os.path.join(TAB, "table_baseline_comparison.csv"), index=False)

    # store conformal factors for downstream propagation
    for k, v in conformal_q.items():
        models.setdefault("_conformal", {})[k] = v

    # Figure 7: predicted vs simulated (chw), test points highlighted
    iv0 = "chw_outreach"
    y = np.array([draw_records[(si, iv0)]["measles_averted"].mean() for si in range(len(scen))])
    pr = ens_predict(models[(iv0, "measles_averted")], Xs).mean(0)
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.scatter(y[train_idx], pr[train_idx], s=9, alpha=0.4, color="#95a5a6", label="train")
    ax.scatter(y[test_idx], pr[test_idx], s=16, alpha=0.7, color="#27ae60", label="test")
    if len(ood_idx):
        ax.scatter(y[ood_idx], pr[ood_idx], s=22, alpha=0.9, color="#c0392b", marker="^",
                   label="high coverage (MR1>=%.2f)" % C.OOD_MR1_CUTOFF)
    lim = [0, max(y.max(), pr.max()) * 1.05]
    ax.plot(lim, lim, "--", color="gray")
    ax.set_xlabel("Simulated measles cases averted per 100k")
    ax.set_ylabel("Metamodel prediction")
    ax.set_title(f"CHW outreach  test R2 = {r2_score(y[test_idx], pr[test_idx]):.3f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig07_pred_vs_sim.png"), bbox_inches="tight")
    plt.close(fig)

    # Figure 8: calibration before/after conformal (DALYs averted, all interventions)
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)
    for ax, col_pref, ttl in zip(axs, ["PICP_raw", "PICP_conformal"],
                                 ["Before recalibration", "After conformal recalibration"]):
        for iv in C.INTERVENTIONS:
            sub = calib[(calib.intervention == C.INTERVENTION_LABELS[iv]) &
                        (calib.outcome == "daly_averted")]
            ax.plot(sub["nominal"], sub[col_pref], "-o", label=C.INTERVENTION_LABELS[iv])
        ax.plot([0.8, 0.95], [0.8, 0.95], "--", color="gray")
        ax.set_xlabel("Nominal coverage")
        ax.set_title(ttl)
        ax.legend(fontsize=8)
    axs[0].set_ylabel("Empirical coverage (PICP)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig08_calibration.png"), bbox_inches="tight")
    plt.close(fig)

    # Figure 16: baseline metamodel comparison (test R2 and PICP90)
    piv = baseline[baseline.outcome == "daly_averted"].groupby("model").agg(
        R2_test=("R2_test", "mean")).reset_index()
    order = ["DeepEnsemble", "GP_kriging", "RandomForest", "GradientBoosting", "QuantileRF"]
    piv["model"] = pd.Categorical(piv["model"], order)
    piv = piv.sort_values("model")
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(piv["model"].astype(str), piv["R2_test"], color="#34495e")
    ax.set_ylabel("Mean test R-squared (DALYs averted)")
    ax.set_ylim(0.9, 1.0)
    plt.xticks(rotation=20, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig16_baseline_comparison.png"), bbox_inches="tight")
    plt.close(fig)

    # cost-uncertainty band figure (Fig 9) with conformal intervals
    grid = np.linspace(0.45, 0.95, 40)
    Xg = scaler.transform(np.column_stack([grid, 0.75 * grid, np.full_like(grid, 0.2)]))
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    band = {}
    for iv, col in zip(C.INTERVENTIONS, ["#2980b9", "#27ae60", "#c0392b"]):
        cost = ens_predict(models[(iv, "cost_total")], Xg).mean(0)
        daly_stack = ens_predict(models[(iv, "daly_averted")], Xg)
        daly = daly_stack.mean(0)
        daly_sd = daly_stack.std(0) + 1e-6
        cpd = cost / np.maximum(daly, C.ICER_DALY_FLOOR)
        cpd_hi = cost / np.maximum(daly - Z90 * daly_sd, C.ICER_DALY_FLOOR)
        cpd_lo = cost / np.maximum(daly + Z90 * daly_sd, C.ICER_DALY_FLOOR)
        ax.plot(grid, cpd, color=col, lw=2, label=C.INTERVENTION_LABELS[iv])
        ax.fill_between(grid, cpd_lo, np.minimum(cpd_hi, np.nanpercentile(cpd_hi, 95)),
                        color=col, alpha=0.15)
        band[iv] = cpd
    ax.set_xlabel("MR1 before")
    ax.set_ylabel("Cost per DALY averted (USD)")
    ax.set_ylim(0, np.nanpercentile(np.concatenate(list(band.values())), 97))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig09_cost_uncertainty_bands.png"), bbox_inches="tight")
    plt.close(fig)

    _r2 = pd.to_numeric(acc["R2_test"], errors="coerce")
    _r2 = _r2[np.isfinite(_r2) & (_r2 > -1.0)]
    SUMMARY["metamodel_mean_R2_test"] = float(_r2.mean())
    _h = acc[acc["outcome"] != "cost_total"]
    SUMMARY["metamodel_mean_R2_test_health"] = float(pd.to_numeric(_h["R2_test"], errors="coerce").mean())
    SUMMARY["metamodel_mean_R2_test_cost"] = float(
        pd.to_numeric(acc[acc["outcome"] == "cost_total"]["R2_test"], errors="coerce").mean())
    SUMMARY["metamodel_max_rel_RMSE_test"] = float(pd.to_numeric(acc["rel_RMSE_test"], errors="coerce").max())
    SUMMARY["metamodel_min_R2_test"] = float(_r2.min())
    SUMMARY["metamodel_mean_R2_extrapolation_diag"] = float(pd.to_numeric(acc["R2_extrapolation_diag"], errors="coerce").mean())
    SUMMARY["metamodel_mean_R2_highcov_deployed"] = float(pd.to_numeric(acc["R2_test_highcov_deployed"], errors="coerce").mean())
    # calibration headline: worst-case DALY PICP at nominal 0.90 before/after
    d90 = calib[(calib.outcome == "daly_averted") & (calib.nominal == 0.90)]
    SUMMARY["daly_picp90_highcov_range"] = [float(pd.to_numeric(d90["PICP_highcov_test"], errors="coerce").min()),
                                            float(pd.to_numeric(d90["PICP_highcov_test"], errors="coerce").max())]
    SUMMARY["daly_picp90_raw_range"] = [float(d90["PICP_raw"].min()), float(d90["PICP_raw"].max())]
    SUMMARY["daly_picp90_conformal_range"] = [float(d90["PICP_conformal"].min()), float(d90["PICP_conformal"].max())]
    return models, scaler, aleatoric, acc, calib, baseline, (train_idx, test_idx, ood_idx)


def _baseline_models():
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(1e-2)
    return [
        ("GP_kriging", GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                                n_restarts_optimizer=1, random_state=0)),
        ("RandomForest", RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)),
        ("GradientBoosting", GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                                       learning_rate=0.05, random_state=0)),
        ("QuantileRF", RandomForestRegressor(n_estimators=300, random_state=1, n_jobs=-1)),
    ]


# ======================================================================
# STAGE 5  District populations (real division data) and objectives
# ======================================================================
def build_populations():
    div_names = list(C.DIVISIONS.keys())
    div_u5 = np.array([C.DIVISION_U5_SHARE[d] for d in div_names])   # real cohort weights
    div_urban = np.array([C.DIVISIONS[d][1] for d in div_names])
    div_r0 = np.array([C.DIVISIONS[d][2] for d in div_names])
    div_cfr = np.array([C.DIVISIONS[d][3] for d in div_names])
    div_pop = np.array([C.DIVISIONS_REAL[d][0] for d in div_names], float)

    rows = []
    for _ in range(C.M_POPULATIONS):
        # centre district coverage on the real national anchors (CES 2023)
        mr1 = float(np.clip(rng.normal(C.MR1_NATIONAL, 0.12), *C.MR1_RANGE))
        mr2 = float(np.clip(rng.normal(min(mr1, C.MR2_NATIONAL), 0.10),
                            C.MR2_RANGE[0], min(mr1, C.MR2_RANGE[1])))
        sia = float(rng.uniform(*C.SIA_RANGE))
        # division mix drawn from the real census population shares
        dw = rng.dirichlet((div_pop / div_pop.sum()) * 60)
        aw = rng.dirichlet(C.AGE_WEIGHTS_CENSUS * 40)
        female = float(np.clip(rng.normal(0.49, 0.02), 0.44, 0.54))
        urban = float((dw * div_urban).sum())
        mean_cfr = float((dw * div_cfr).sum())
        mean_r0 = float((dw * div_r0).sum())
        rows.append({
            "mr1_before": mr1, "mr2_before": mr2, "sia_before": sia,
            "urban_frac": urban, "female_frac": female,
            "sylhet_share": dw[div_names.index("Sylhet")],
            "rangpur_share": dw[div_names.index("Rangpur")],
            "dhaka_share": dw[div_names.index("Dhaka")],
            "mean_cfr_mult": mean_cfr, "mean_r0_mult": mean_r0,
            "young_infant_share": aw[0],
            **{f"div_{d}": dw[i] for i, d in enumerate(div_names)},
        })
    return pd.DataFrame(rows), div_names, div_u5, div_r0, div_cfr, div_pop


def _propagate_objectives(pops, models, scaler, target_name, cvar_alpha, delta,
                          wtp, budget, joint=True, S_MC=None, dmin=None):
    if S_MC is None:
        S_MC = C.S_MC
    if dmin is None:
        dmin = C.ICER_DALY_FLOOR
    """Propagate ensemble uncertainty to per-population decision quantities.
    Returns dict of arrays keyed by intervention plus feasibility and labels."""
    feats = pops[["mr1_before", "mr2_before", "sia_before"]].values
    Xs = scaler.transform(feats)
    P = len(pops)
    cfr_adj = pops["mean_cfr_mult"].values
    r0_adj = pops["mean_r0_mult"].values
    T = immunity_target(r0_adj, target_name)

    ve1s = np.clip(rng.normal(C.VE1_MEAN, C.VE1_SD, S_MC), 0.75, 0.93)
    ve2s = np.clip(rng.normal(C.VE2_MEAN, C.VE2_SD, S_MC), 0.93, 0.99)
    vesias = np.clip(rng.normal(C.VE_SIA_MEAN, C.VE_SIA_SD, S_MC), 0.75, 0.92)
    effs = EPI.draw_effect_sizes(rng, S_MC)

    corr = SUMMARY.get("cost_daly_corr", {})
    conformal = models.get("_conformal", {})

    det_cpd, cvar_cpd, det_daly, robust_daly = {}, {}, {}, {}
    det_feas, robust_feas, cost_mean, measles_mean = {}, {}, {}, {}
    budget_ok = {}

    mr1b = pops["mr1_before"].values
    mr2b = pops["mr2_before"].values
    siab = pops["sia_before"].values

    for iv in C.INTERVENTIONS:
        m_stack = ens_predict(models[(iv, "measles_averted")], Xs)
        d_stack = ens_predict(models[(iv, "daly_averted")], Xs)
        c_stack = ens_predict(models[(iv, "cost_total")], Xs)
        m_mu = m_stack.mean(0)
        d_mu = d_stack.mean(0)
        c_mu = c_stack.mean(0)
        # Equation 23: epistemic ensemble spread PLUS the aleatoric replication
        # term, evaluated at these districts through the fitted aleatoric model.
        # Using the ensemble spread alone here would discard simulation noise at
        # exactly the point where the decision is made.
        d_sd = predict_total_sd(models[(iv, "daly_averted")], Xs, d_stack.std(0)) + 1e-6
        c_sd = predict_total_sd(models[(iv, "cost_total")], Xs, c_stack.std(0)) + 1e-6
        # conformal scaling of the predictive sd (90% factor) so intervals are calibrated
        qd = conformal.get((iv, "daly_averted", 0.90), Z90) / Z90
        qc = conformal.get((iv, "cost_total", 0.90), Z90) / Z90
        d_sd = d_sd * qd
        c_sd = c_sd * qc

        # stochastic post-intervention coverage with literature effect sizes
        if iv == "sms_reminder":
            gain = effs["sms_share"][None, :] * (1 - mr1b)[:, None]
            mr1d = np.clip(mr1b[:, None] + gain, 0, 0.99); siad = np.tile(siab[:, None], (1, S_MC))
        elif iv == "chw_outreach":
            gain = (effs["chw_base"][None, :] - effs["chw_slope"][None, :] * mr1b[:, None]) * (1 - mr1b)[:, None]
            mr1d = np.clip(mr1b[:, None] + gain, 0, 0.99); siad = np.tile(siab[:, None], (1, S_MC))
        else:
            mr1d = np.tile(mr1b[:, None], (1, S_MC)); siad = np.tile(effs["sia_reach"][None, :], (P, 1))

        imm_det = EPI.immune_fraction(
            np.clip(mr1b + (0.30 * (1 - mr1b) if iv == "sms_reminder"
                    else (0.62 - 0.15 * mr1b) * (1 - mr1b) if iv == "chw_outreach" else 0.0), 0, 0.99),
            np.minimum(mr2b, mr1b),
            (np.full(P, C.EFF_SIA_MEAN) if iv == "sia_campaign" else siab),
            C.VE1_MEAN, C.VE2_MEAN, C.VE_SIA_MEAN)
        imm_samp = EPI.immune_fraction(mr1d, np.minimum(mr2b[:, None], mr1d), siad,
                                       ve1s[None, :], ve2s[None, :], vesias[None, :])
        det_feas[iv] = imm_det >= T
        robust_feas[iv] = (imm_samp >= T[:, None]).mean(1) >= (1 - delta)

        scale = (0.6 + 0.4 * cfr_adj) * (0.9 + 0.1 * r0_adj)
        d_mu_s = d_mu * scale
        det_daly[iv] = d_mu_s
        robust_daly[iv] = (d_mu - 1.2816 * d_sd) * scale
        cost_mean[iv] = c_mu
        measles_mean[iv] = m_mu * (0.9 + 0.1 * r0_adj)
        budget_ok[iv] = c_mu <= budget

        rho = np.clip(corr.get(C.INTERVENTION_LABELS[iv], 0.0), -0.95, 0.95) if joint else 0.0
        z1 = rng.standard_normal((P, S_MC))
        z2 = rng.standard_normal((P, S_MC))
        zc = z1
        zd = rho * z1 + np.sqrt(max(1e-6, 1 - rho ** 2)) * z2
        # Vectorized over districts. This draws the same standard normals in the
        # same order as the per-district loop it replaces and performs the same
        # arithmetic, so results are unchanged; it is only faster, which matters
        # because the nested bootstrap re-executes this propagation on every
        # replicate.
        dsamp = np.maximum(d_mu_s[:, None] + (d_sd * scale)[:, None] * zd, dmin)
        csamp = np.maximum(c_mu[:, None] + c_sd[:, None] * zc, 1.0)
        cpd = csamp / dsamp
        det_cpd[iv] = cpd.mean(1)
        qv = np.quantile(cpd, cvar_alpha, axis=1)
        tail = cpd >= qv[:, None]
        cvar_cpd[iv] = (cpd * tail).sum(1) / np.maximum(tail.sum(1), 1)

    IVS = C.INTERVENTIONS
    BIG = 1e15

    def choose(feas_dict, cpd_dict, daly_dict, budget_dict=None):
        """Vectorized form of the constrained rule. Identical semantics to the
        per-district loop it replaces: feasible interventions are screened by
        willingness to pay, the screen is relaxed if it would empty the pool, the
        lowest risk adjusted cost per DALY wins, and if nothing is feasible the
        rule falls back to the greatest lower bound on DALYs averted."""
        F = np.column_stack([np.asarray(feas_dict[iv], bool) &
                             (np.asarray(budget_dict[iv], bool) if budget_dict
                              else np.ones(P, bool))
                             for iv in IVS])
        CPD = np.column_stack([cpd_dict[iv] for iv in IVS])
        GAIN = np.column_stack([daly_dict[iv] for iv in IVS])

        accept = F & (CPD <= wtp)
        pool = np.where(accept.any(1)[:, None], accept, F)
        cost = np.where(pool, CPD, BIG)
        lab = np.argmin(cost, axis=1)
        none_feasible = ~F.any(1)
        if none_feasible.any():
            lab[none_feasible] = np.argmax(GAIN[none_feasible], axis=1)
        return lab.astype(int)

    det_label = choose(det_feas, det_cpd, det_daly)
    cvar_label = choose(robust_feas, cvar_cpd, robust_daly, budget_ok if np.isfinite(budget) else None)

    return {"det_label": det_label, "cvar_label": cvar_label,
            "det_cpd": det_cpd, "cvar_cpd": cvar_cpd, "cost_mean": cost_mean,
            "measles_mean": measles_mean, "robust_feas": robust_feas, "det_feas": det_feas}


def stage5_objectives(pops, models, scaler):
    print("Stage 5: risk adjusted objectives (joint vs independent)")
    base = _propagate_objectives(pops, models, scaler, C.DEFAULT_TARGET, C.CVAR_ALPHA,
                                 C.DELTA_ROBUST, C.WTP_THRESHOLD, float("inf"), joint=True)
    indep = _propagate_objectives(pops, models, scaler, C.DEFAULT_TARGET, C.CVAR_ALPHA,
                                  C.DELTA_ROBUST, C.WTP_THRESHOLD, float("inf"), joint=False)

    pops = pops.copy()
    pops["det_label"] = base["det_label"]
    pops["cvar_label"] = base["cvar_label"]
    pops["cvar_label_indep"] = indep["cvar_label"]
    for iv in C.INTERVENTIONS:
        pops[f"cpd_mean_{iv}"] = base["det_cpd"][iv]
        pops[f"cpd_cvar_{iv}"] = base["cvar_cpd"][iv]
        pops[f"cost_{iv}"] = base["cost_mean"][iv]
        pops[f"measles_{iv}"] = base["measles_mean"][iv]
        pops[f"robust_feas_{iv}"] = base["robust_feas"][iv].astype(int)
    pops.to_csv(os.path.join(DATA, "population_objectives.csv"), index=False)

    # joint vs independent comparison (reviewer #7)
    joint_shares = _shares(base["cvar_label"])
    indep_shares = _shares(indep["cvar_label"])
    flip_ji = float((base["cvar_label"] != indep["cvar_label"]).mean())
    ji_rows = [{"sampling": "joint", **joint_shares},
               {"sampling": "independent", **indep_shares}]
    pd.DataFrame(ji_rows).to_csv(os.path.join(TAB, "table_joint_vs_independent.csv"), index=False)
    SUMMARY["joint_vs_independent_flip"] = flip_ji

    # Pareto frontier membership. The two objectives are MINIMIZE programme cost
    # and MAXIMIZE measles cases averted, so dominance must be evaluated with the
    # averted axis negated; comparing both axes in the minimize direction would
    # treat averting fewer cases as an improvement and hand the frontier to the
    # cheapest intervention by construction.
    cost_mat = np.column_stack([base["cost_mean"][iv] for iv in C.INTERVENTIONS])
    meas_mat = np.column_stack([base["measles_mean"][iv] for iv in C.INTERVENTIONS])
    pareto_count = np.zeros(len(C.INTERVENTIONS))
    ratio_count = np.zeros(len(C.INTERVENTIONS))
    for j in range(len(pops)):
        pts = np.column_stack([cost_mat[j], -meas_mat[j]])
        for a in range(len(C.INTERVENTIONS)):
            dominated = False
            for b in range(len(C.INTERVENTIONS)):
                if b == a:
                    continue
                if (pts[b] <= pts[a]).all() and (pts[b] < pts[a]).any():
                    dominated = True
                    break
            if not dominated:
                pareto_count[a] += 1
        # frequency with which each intervention minimizes the SCALAR cost per
        # case averted, which is the objective the original framework optimizes
        ratio = cost_mat[j] / np.maximum(meas_mat[j], 1e-6)
        ratio_count[int(np.argmin(ratio))] += 1

    j0 = int(np.argmin(pops["mr1_before"].values))
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for a, (iv, col) in enumerate(zip(C.INTERVENTIONS, ["#2980b9", "#27ae60", "#c0392b"])):
        ax.scatter(cost_mat[j0, a], meas_mat[j0, a], s=90, color=col,
                   label=C.INTERVENTION_LABELS[iv], zorder=3)
    ax.set_xlabel("Programme cost per 100k (USD)")
    ax.set_ylabel("Residual measles cases per 100k")
    ax.set_title("Objective space (lowest coverage district)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig10_pareto.png"), bbox_inches="tight")
    plt.close(fig)

    pareto_df = pd.DataFrame({"intervention": [C.INTERVENTION_LABELS[i] for i in C.INTERVENTIONS],
                              "pareto_frequency": np.round(pareto_count / len(pops), 3),
                              "scalar_ratio_minimizer_frequency": np.round(ratio_count / len(pops), 3)})
    pareto_df.to_csv(os.path.join(TAB, "table_pareto_frequency.csv"), index=False)
    SUMMARY["pareto_frequency"] = {C.INTERVENTION_LABELS[iv]: float(round(pareto_count[i] / len(pops), 3))
                                   for i, iv in enumerate(C.INTERVENTIONS)}
    SUMMARY["scalar_ratio_minimizer_frequency"] = {
        C.INTERVENTION_LABELS[iv]: float(round(ratio_count[i] / len(pops), 3))
        for i, iv in enumerate(C.INTERVENTIONS)}
    return pops


def _shares(labels):
    vals, counts = np.unique(labels, return_counts=True)
    out = {C.INTERVENTION_LABELS[iv]: 0.0 for iv in C.INTERVENTIONS}
    for v, c in zip(vals, counts):
        out[C.INTERVENTION_LABELS[C.INTERVENTIONS[int(v)]]] = round(c / len(labels), 3)
    return out


# ======================================================================
# STAGE 6  Decision trees
# ======================================================================
TREE_FEATURES = ["mr1_before", "mr2_before", "sia_before", "urban_frac",
                 "female_frac", "sylhet_share", "rangpur_share", "mean_cfr_mult"]


def fit_tree(pops, label_col, depth=4):
    X = pops[TREE_FEATURES].values
    y = pops[label_col].values
    clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=25,
                                 random_state=C.GLOBAL_SEED)
    clf.fit(X, y)
    return clf


def tree_splits_table(clf, name):
    t = clf.tree_
    rows = []
    for i in range(t.node_count):
        if t.children_left[i] != t.children_right[i]:
            rows.append({"tree": name, "node": i,
                         "feature": TREE_FEATURES[t.feature[i]],
                         "threshold": round(float(t.threshold[i]), 4),
                         "samples": int(t.n_node_samples[i])})
    return pd.DataFrame(rows)


def plot_one_tree(clf, fname, title):
    fig, ax = plt.subplots(figsize=(11, 6.2))
    plot_tree(clf, feature_names=TREE_FEATURES,
              class_names=[C.INTERVENTION_LABELS[i] for i in C.INTERVENTIONS],
              filled=True, rounded=True, fontsize=7, ax=ax, impurity=False, proportion=True)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), bbox_inches="tight")
    plt.close(fig)


def stage6_trees(pops):
    print("Stage 6: fitting decision trees")
    det = fit_tree(pops, "det_label")
    cvar = fit_tree(pops, "cvar_label")
    plot_one_tree(det, "fig11_tree_deterministic.png",
                  "Deterministic decision tree (minimize mean cost per DALY averted)")
    plot_one_tree(cvar, "fig12_tree_cvar.png",
                  "Uncertainty aware decision tree (minimize CVaR of cost per DALY averted)")
    st = pd.concat([tree_splits_table(det, "deterministic"),
                    tree_splits_table(cvar, "uncertainty_aware")], ignore_index=True)
    st.to_csv(os.path.join(TAB, "table_tree_splits.csv"), index=False)

    agree = float((pops["det_label"].values == pops["cvar_label"].values).mean())
    SUMMARY["tree_agreement"] = agree
    SUMMARY["det_root_feature"] = TREE_FEATURES[det.tree_.feature[0]]
    SUMMARY["det_root_threshold"] = float(det.tree_.threshold[0])
    SUMMARY["cvar_root_feature"] = TREE_FEATURES[cvar.tree_.feature[0]]
    SUMMARY["cvar_root_threshold"] = float(cvar.tree_.threshold[0])
    SUMMARY["det_label_share"] = _shares(pops["det_label"].values)
    SUMMARY["cvar_label_share"] = _shares(pops["cvar_label"].values)
    return det, cvar, st, agree


# ======================================================================
# STAGE 7  Boundary stability: simple + nested bootstrap
# ======================================================================
def stage7_simple(pops, B=None):
    """Simple decision-boundary bootstrap (resample districts, refit tree)."""
    B = C.B_BOOT if B is None else B
    print("Stage 7a: simple bootstrap boundary stability")
    root_feats, root_thresh = [], []
    for _ in range(B):
        idx = rng.integers(0, len(pops), len(pops))
        clf = fit_tree(pops.iloc[idx], "cvar_label")
        root_feats.append(TREE_FEATURES[clf.tree_.feature[0]])
        root_thresh.append(float(clf.tree_.threshold[0]))
    root_feats = np.array(root_feats); root_thresh = np.array(root_thresh)
    dominant = pd.Series(root_feats).mode()[0]
    mask = root_feats == dominant
    ci_lo, ci_hi = np.percentile(root_thresh[mask], [2.5, 97.5])
    med = float(np.median(root_thresh[mask]))
    return {"dominant": dominant, "root_feats": root_feats, "root_thresh": root_thresh,
            "med": med, "ci_lo": float(ci_lo), "ci_hi": float(ci_hi)}


def stage7_nested_chunk(pops, models, scaler, scen, draw_records, k0, k1,
                        split=None, resample_districts=False):
    """Run nested-bootstrap iterations [k0, k1): resample the simulation
    replicates within each scenario -> retrain ensemble -> relabel districts ->
    refit tree. Propagates Monte-Carlo simulation and metamodel uncertainty
    through the whole pipeline.

    `resample_districts` selects the arm. With False the district set is held
    fixed and only simulation and metamodel error move, which isolates the
    metamodel component. With True the districts are ALSO resampled, which gives
    the joint arm; comparing the joint arm with the district-only bootstrap is
    what identifies the decomposition, because the two nested arms then differ in
    exactly one factor rather than two. Returns (feats, thresh); resumable.
    """
    arm = "joint (districts + replicates)" if resample_districts else "replicates only"
    print(f"Stage 7b: nested bootstrap [{arm}] iterations {k0}..{k1-1}")
    X = scen[["mr1_before", "mr2_before", "sia_before"]].values
    Xs_scen = scaler.transform(X)
    # The nested ensemble must be trained on the SAME training scenarios as the
    # primary ensemble; training it on every scenario would let the nested arm
    # see the test and OOD scenarios and understate its own error.
    train_idx = split[0] if split is not None else np.arange(len(scen))
    n_scen = len(scen)
    n_draws = len(draw_records[(0, C.INTERVENTIONS[0])][TARGETS[0]])
    feats, thresh = [], []
    for bnb in range(k0, k1):
        # resample simulation replicates within every scenario (with replacement)
        draw_idx = [rng.integers(0, n_draws, n_draws) for _ in range(n_scen)]
        nested_models = {}
        for iv in C.INTERVENTIONS:
            for tgt in TARGETS:
                y = np.array([draw_records[(si, iv)][tgt][draw_idx[si]].mean()
                              for si in range(n_scen)])
                y_sd = np.array([draw_records[(si, iv)][tgt][draw_idx[si]].std()
                                 for si in range(n_scen)])
                ymean, ystd = y[train_idx].mean(), y[train_idx].std() + 1e-9
                ys = (y - ymean) / ystd
                nested_models[(iv, tgt)] = _train_ensemble(
                    Xs_scen[train_idx], ys[train_idx], (ymean, ystd),
                    C.N_ENSEMBLE, sd_tr=y_sd[train_idx])
        nested_models["_conformal"] = models.get("_conformal", {})
        pops_b = pops.iloc[rng.integers(0, len(pops), len(pops))].reset_index(drop=True) \
            if resample_districts else pops
        obj = _propagate_objectives(pops_b, nested_models, scaler, C.DEFAULT_TARGET,
                                    C.CVAR_ALPHA, C.DELTA_ROBUST, C.WTP_THRESHOLD,
                                    float("inf"), joint=True, S_MC=C.S_MC_NESTED)
        tmp = pops_b.copy(); tmp["cvar_label"] = obj["cvar_label"]
        clf = fit_tree(tmp, "cvar_label")
        feats.append(TREE_FEATURES[clf.tree_.feature[0]])
        thresh.append(float(clf.tree_.threshold[0]))
        print(f"      nested[{arm}] {bnb + 1} done", flush=True)
    return feats, thresh


def split_candidate_resolution(pops, feature):
    """Spacing of the CART split candidates on `feature`.

    CART can only place a threshold at a midpoint between two adjacent observed
    values, so no bootstrap arm can report an interval narrower than this grid.
    Reporting it prevents a narrow nested interval from being read as evidence
    of low metamodel error when it is really the resolution floor of the tree.
    """
    v = np.sort(np.unique(pops[feature].values))
    if len(v) < 2:
        return float("nan")
    return float(np.median(np.diff(v)))


def stage7_finalize(pops, simple, nested_feats, nested_thresh,
                    joint_feats=None, joint_thresh=None):
    """Assemble figures, tables and summary from simple + nested bootstrap.
    Robust to root-feature instability across nested replicates: the nested
    threshold CI is reported on whichever feature the nested trees most often
    select, and the nested feature-selection frequencies are reported so that
    boundary-feature instability is made explicit rather than hidden."""
    print("Stage 7c: finalizing boundary stability outputs")
    dominant = simple["dominant"]
    root_feats = np.asarray(simple["root_feats"]); root_thresh = np.asarray(simple["root_thresh"])
    mask = root_feats == dominant
    ci_lo, ci_hi = simple["ci_lo"], simple["ci_hi"]; med = simple["med"]
    nested_feats = np.asarray(nested_feats); nested_thresh = np.asarray(nested_thresh)

    # nested root-feature selection frequencies
    nfreq = pd.Series(nested_feats).value_counts(normalize=True)
    nested_dominant = nfreq.index[0]
    nested_feat_agree = float((nested_feats == dominant).mean())  # matches district-bootstrap feature
    nmask = nested_feats == nested_dominant
    if nmask.sum() >= 2:
        n_ci_lo, n_ci_hi = np.percentile(nested_thresh[nmask], [2.5, 97.5])
        n_med = float(np.median(nested_thresh[nmask]))
    else:
        n_ci_lo = n_ci_hi = n_med = float(nested_thresh[nmask][0]) if nmask.sum() else float("nan")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.hist(root_thresh[mask], bins=28, color="#8e44ad", alpha=0.6, edgecolor="white",
            label=f"district bootstrap ({dominant})")
    if nmask.sum() >= 2 and nested_dominant == dominant:
        ax.hist(nested_thresh[nmask], bins=18, color="#e67e22", alpha=0.55, edgecolor="white",
                label=f"nested / full pipeline ({nested_dominant})")
        ax.axvline(n_med, color="#e67e22", lw=1.5)
        ax.axvspan(n_ci_lo, n_ci_hi, color="#e67e22", alpha=0.12)
    elif nmask.sum() >= 2:
        # nested most often splits on a different feature: show it on a twin axis note
        ax.hist(nested_thresh[nmask], bins=18, color="#e67e22", alpha=0.45, edgecolor="white",
                label=f"nested / full pipeline ({nested_dominant})")
        ax.axvline(n_med, color="#e67e22", lw=1.5, ls="--")
    ax.axvline(med, color="#8e44ad", lw=1.5)
    ax.axvspan(ci_lo, ci_hi, color="#8e44ad", alpha=0.12)
    ax.set_xlabel("Root split threshold")
    ax.set_ylabel("Bootstrap frequency")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig13_boundary_stability.png"), bbox_inches="tight")
    plt.close(fig)

    freq = pd.Series(root_feats).value_counts(normalize=True)
    bt = pd.DataFrame({"root_feature": freq.index, "selection_frequency": np.round(freq.values, 3)})
    bt.to_csv(os.path.join(TAB, "table_boundary_feature_frequency.csv"), index=False)

    # nested feature-frequency table (explicit boundary-feature instability)
    nbt = pd.DataFrame({"root_feature": nfreq.index,
                        "nested_selection_frequency": np.round(nfreq.values, 3)})
    nbt.to_csv(os.path.join(TAB, "table_nested_feature_frequency.csv"), index=False)

    stab_rows = [
        {"analysis": "District resampling only", "root_feature": dominant,
         "feature_frequency": round(float(mask.mean()), 3), "median_threshold": round(med, 3),
         "ci_lo": round(float(ci_lo), 3), "ci_hi": round(float(ci_hi), 3),
         "ci_width": round(float(ci_hi - ci_lo), 3),
         "sd_threshold": round(float(np.std(root_thresh[mask])), 4)},
        {"analysis": "Replicate resampling only (metamodel)", "root_feature": nested_dominant,
         "feature_frequency": round(float(nmask.mean()), 3), "median_threshold": round(n_med, 3),
         "ci_lo": round(float(n_ci_lo), 3), "ci_hi": round(float(n_ci_hi), 3),
         "ci_width": round(float(n_ci_hi - n_ci_lo), 3),
         "sd_threshold": round(float(np.std(nested_thresh[nmask])), 4)},
    ]

    # ---- Joint arm: districts AND replicates resampled together ----
    # The district-only and replicate-only arms differ from each other in two
    # ways at once, so on their own they cannot attribute boundary movement to a
    # source. The joint arm differs from each single-factor arm in exactly one
    # factor, which identifies the decomposition.
    if joint_thresh is not None and len(joint_thresh) >= 2:
        joint_feats = np.asarray(joint_feats); joint_thresh = np.asarray(joint_thresh)
        jfreq = pd.Series(joint_feats).value_counts(normalize=True)
        j_dominant = jfreq.index[0]
        jmask = joint_feats == j_dominant
        j_ci_lo, j_ci_hi = np.percentile(joint_thresh[jmask], [2.5, 97.5])
        j_med = float(np.median(joint_thresh[jmask]))
        j_sd = float(np.std(joint_thresh[jmask]))
        stab_rows.append(
            {"analysis": "Joint (districts + replicates)", "root_feature": j_dominant,
             "feature_frequency": round(float(jmask.mean()), 3), "median_threshold": round(j_med, 3),
             "ci_lo": round(float(j_ci_lo), 3), "ci_hi": round(float(j_ci_hi), 3),
             "ci_width": round(float(j_ci_hi - j_ci_lo), 3), "sd_threshold": round(j_sd, 4)})

        v_dist = float(np.var(root_thresh[mask]))
        v_meta = float(np.var(nested_thresh[nmask]))
        v_joint = float(np.var(joint_thresh[jmask]))
        SUMMARY["boundary_variance_decomposition"] = {
            "var_district_only": round(v_dist, 6),
            "var_replicate_only": round(v_meta, 6),
            "var_joint": round(v_joint, 6),
            "share_district_of_joint": (round(v_dist / v_joint, 3) if v_joint > 0 else None),
            "share_replicate_of_joint": (round(v_meta / v_joint, 3) if v_joint > 0 else None),
        }
        SUMMARY["joint_boundary_median"] = j_med
        SUMMARY["joint_boundary_ci"] = [float(j_ci_lo), float(j_ci_hi)]
        SUMMARY["joint_boundary_ci_width"] = float(j_ci_hi - j_ci_lo)

    res = split_candidate_resolution(pops, dominant)
    SUMMARY["split_candidate_resolution"] = res
    for r in stab_rows:
        r["candidate_grid_spacing"] = round(res, 5)
    stab = pd.DataFrame(stab_rows)
    stab.to_csv(os.path.join(TAB, "table_boundary_stability.csv"), index=False)

    SUMMARY["boundary_feature"] = dominant
    SUMMARY["boundary_median"] = med
    SUMMARY["boundary_ci"] = [float(ci_lo), float(ci_hi)]
    SUMMARY["boundary_ci_width"] = float(ci_hi - ci_lo)
    SUMMARY["nested_boundary_feature"] = str(nested_dominant)
    SUMMARY["nested_feature_frequency"] = {str(k): round(float(v), 3) for k, v in nfreq.items()}
    SUMMARY["nested_feature_agreement_with_district"] = round(nested_feat_agree, 3)
    SUMMARY["nested_boundary_median"] = n_med
    SUMMARY["nested_boundary_ci"] = [float(n_ci_lo), float(n_ci_hi)]
    SUMMARY["nested_boundary_ci_width"] = float(n_ci_hi - n_ci_lo)
    return dominant, med, (ci_lo, ci_hi), bt, stab


def stage7_boundary(pops, models, scaler, scen, draw_records, split, B=None, B_nested=None):
    """Single-process convenience wrapper (simple + full nested + finalize)."""
    B_nested = C.B_NESTED if B_nested is None else B_nested
    print("Stage 7: bootstrap boundary stability (district + replicate + joint)")
    simple = stage7_simple(pops, B=B)
    nf, nt = stage7_nested_chunk(pops, models, scaler, scen, draw_records, 0, B_nested,
                                 split=split, resample_districts=False)
    jf, jt = stage7_nested_chunk(pops, models, scaler, scen, draw_records, 0, C.B_JOINT,
                                 split=split, resample_districts=True)
    return stage7_finalize(pops, simple, nf, nt, jf, jt)


# ======================================================================
# STAGE 8  Comparison, flip typology, equity, mechanistic CHW
# ======================================================================
def stage8_compare_equity(pops, det, cvar, div_names, div_u5, div_r0, div_cfr, div_pop,
                          models=None, scaler=None):
    print("Stage 8: comparison, flip typology, equity, mechanistic CHW")
    disagree = pops["det_label"].values != pops["cvar_label"].values

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.scatter(pops["mr1_before"][~disagree], pops["sia_before"][~disagree],
               s=8, color="#95a5a6", alpha=0.5, label="agree")
    ax.scatter(pops["mr1_before"][disagree], pops["sia_before"][disagree],
               s=14, color="#e74c3c", alpha=0.8, label="disagree")
    ax.set_xlabel("MR1 before")
    ax.set_ylabel("SIA before")
    ax.set_title(f"Recommendation disagreement ({disagree.mean()*100:.1f}% of districts)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig14_agreement.png"), bbox_inches="tight")
    plt.close(fig)

    # ---- Share of districts sitting in the extrapolation region ----
    # The metamodel is only validated below the OOD cutoff, so any district at or
    # above it receives a recommendation the accuracy table does not support.
    # Reporting the share, and the disagreement rate inside it, bounds how much
    # of the result rests on extrapolation.
    ood_district = pops["mr1_before"].values >= C.OOD_MR1_CUTOFF
    SUMMARY["district_share_in_ood_region"] = float(ood_district.mean())
    SUMMARY["disagree_rate_in_ood_region"] = (float(disagree[ood_district].mean())
                                              if ood_district.sum() else 0.0)
    SUMMARY["ood_cutoff"] = float(C.OOD_MR1_CUTOFF)

    thr = SUMMARY["cvar_root_threshold"]
    near = np.abs(pops[SUMMARY["cvar_root_feature"]].values - thr) < 0.05
    SUMMARY["disagree_rate"] = float(disagree.mean())
    SUMMARY["disagree_rate_near_boundary"] = float(disagree[near].mean()) if near.sum() else 0.0
    SUMMARY["disagree_rate_far_boundary"] = float(disagree[~near].mean()) if (~near).sum() else 0.0

    # ---- Flip typology table (reviewer #13) ----
    det_l = pops["det_label"].values
    cvar_l = pops["cvar_label"].values
    mr1 = pops["mr1_before"].values
    flip_rows = []
    for a in range(3):
        for b in range(3):
            if a == b:
                continue
            m = (det_l == a) & (cvar_l == b)
            if m.sum() == 0:
                continue
            avg_cost_delta = float(np.mean(pops[f"cost_{C.INTERVENTIONS[b]}"].values[m]
                                           - pops[f"cost_{C.INTERVENTIONS[a]}"].values[m]))
            # DALY implication via cost-per-DALY proxy (lower CVaR cpd preferred by robust rule)
            cpd_from = pops[f"cpd_cvar_{C.INTERVENTIONS[a]}"].values[m]
            cpd_to = pops[f"cpd_cvar_{C.INTERVENTIONS[b]}"].values[m]
            near_boundary = float(np.mean(np.abs(mr1[m] - thr) < 0.05))
            flip_rows.append({
                "flip": f"{C.INTERVENTION_LABELS[C.INTERVENTIONS[a]]} -> {C.INTERVENTION_LABELS[C.INTERVENTIONS[b]]}",
                "n_districts": int(m.sum()),
                "share_of_all": round(float(m.mean()), 4),
                "mean_mr1_before": round(float(np.mean(mr1[m])), 3),
                "mean_incremental_cost_per_100k": round(avg_cost_delta, 1),
                "mean_cpd_from": round(float(np.mean(cpd_from)), 1),
                "mean_cpd_to": round(float(np.mean(cpd_to)), 1),
                "share_near_MR1_boundary": round(near_boundary, 3),
            })
    flip_df = pd.DataFrame(flip_rows).sort_values("n_districts", ascending=False)
    flip_df.to_csv(os.path.join(TAB, "table_flip_typology.csv"), index=False)

    # ---- Mechanistic CHW: districts where Pareto ranking and constrained rule disagree ----
    # CHW is chosen but is NOT on the cost-vs-cases Pareto frontier for that district.
    chw = C.INTERVENTIONS.index("chw_outreach")
    cost_mat = np.column_stack([pops[f"cost_{iv}"].values for iv in C.INTERVENTIONS])
    meas_mat = np.column_stack([pops[f"measles_{iv}"].values for iv in C.INTERVENTIONS])

    def on_frontier(j, a):
        pa = np.array([cost_mat[j, a], -meas_mat[j, a]])  # minimize cost, maximize averted
        for b in range(3):
            if b == a:
                continue
            pb = np.array([cost_mat[j, b], -meas_mat[j, b]])
            if (pb <= pa).all() and (pb < pa).any():
                return False
        return True

    chw_chosen = cvar_l == chw
    chw_not_frontier = np.array([chw_chosen[j] and not on_frontier(j, chw) for j in range(len(pops))])
    n_chw_offfront = int(chw_not_frontier.sum())
    SUMMARY["chw_chosen_share"] = float(chw_chosen.mean())
    SUMMARY["chw_chosen_off_frontier_share"] = float(chw_not_frontier.mean())

    # Partition the districts where outreach is chosen into the two DISTINCT
    # routes by which the rule can select it. The mechanism the paper argues for
    # is the first; the second is the fallback branch. The shipped table selected
    # its example districts without separating these, and every example it
    # happened to show was a fallback district, so it illustrated the opposite of
    # the claim it was cited for.
    feas_chw = pops["robust_feas_chw_outreach"].values.astype(bool)
    feas_sms = pops["robust_feas_sms_reminder"].values.astype(bool)
    feas_sia = pops["robust_feas_sia_campaign"].values.astype(bool)
    any_feas = feas_chw | feas_sms | feas_sia
    mech_mask = chw_chosen & feas_chw & ~feas_sms & ~feas_sia   # CHW is the only feasible option
    comp_mask = chw_chosen & feas_chw & (feas_sms | feas_sia)   # CHW competes and wins on CVaR
    fall_mask = chw_chosen & ~any_feas                          # nothing is feasible: fallback branch
    SUMMARY["chw_only_feasible_share"] = float(mech_mask.mean())
    SUMMARY["chw_competitive_share"] = float(comp_mask.mean())
    SUMMARY["chw_fallback_n"] = int(fall_mask.sum())
    SUMMARY["chw_route_counts"] = {"only_feasible": int(mech_mask.sum()),
                                   "competitive": int(comp_mask.sum()),
                                   "fallback_none_feasible": int(fall_mask.sum())}

    mech_rows = []
    for route, mask in [("CHW only feasible option", mech_mask),
                        ("Fallback: none reaches target", fall_mask)]:
        idx = np.where(mask)[0][:3]
        for j in idx:
            mech_rows.append({
                "route": route,
                "district_id": int(j),
                "mr1_before": round(float(pops["mr1_before"].values[j]), 3),
                "chosen": "CHW outreach",
                "chw_reaches_target": int(feas_chw[j]),
                "sms_reaches_target": int(feas_sms[j]),
                "sia_reaches_target": int(feas_sia[j]),
                "cpd_chw": round(float(pops["cpd_cvar_chw_outreach"].values[j]), 1),
                "cpd_sms": round(float(pops["cpd_cvar_sms_reminder"].values[j]), 1),
            })
    pd.DataFrame(mech_rows).to_csv(os.path.join(TAB, "table_chw_mechanism.csv"), index=False)

    # ---- Equity: division outcomes computed from the metamodel ----
    # Every quantity below is predicted by the fitted metamodel for districts
    # assigned to each division and scaled by that division's transmission and
    # case-fatality multipliers; nothing here is a fixed constant. The previous
    # version of this stage multiplied a hardcoded 900 cases by a function of the
    # transmission multiplier, so the reported disparity was an algebraic
    # restatement of the multiplier rather than a model output, and its interval
    # was injected noise rather than a bootstrap.
    div_res = _division_outcomes(pops, models, scaler, div_names, div_r0, div_cfr)
    per_child = div_res["per_100k"]
    ci_lo, ci_hi = div_res["ci_lo"], div_res["ci_hi"]
    gini = _gini(per_child)
    maxmin = per_child.max() / per_child.min()
    SUMMARY["equity_gini_ci"] = div_res["gini_ci"]

    # ---- Allocation objectives under a COMMON budget ----
    # Each objective allocates the same fixed programme budget across divisions
    # and is evaluated on the same outcome scale, so an objective that improves
    # the least served division must give something up elsewhere. The previous
    # version applied unconstrained multiplicative bonuses, which let the maximin
    # objective report a HIGHER national total than the uniform allocation and
    # made the budget weighted objective numerically identical to the uniform one.
    alt_rows = _allocation_objectives(pops, models, scaler, div_names, div_r0,
                                      div_cfr, div_u5, div_pop)
    pd.DataFrame(alt_rows).to_csv(os.path.join(TAB, "table_equity_objectives.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    order = np.argsort(-per_child)
    xs = np.arange(len(div_names))
    ax.bar(xs, per_child[order], color="#16a085",
           yerr=[per_child[order] - ci_lo[order], ci_hi[order] - per_child[order]],
           capsize=3, ecolor="#34495e")
    ax.set_xticks(xs)
    ax.set_xticklabels([div_names[i] for i in order], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Measles cases averted per 100k")
    ax.set_title(f"Cross division equity  (Gini = {gini:.3f}, max/min = {maxmin:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig15_equity.png"), bbox_inches="tight")
    plt.close(fig)

    eq = pd.DataFrame({"division": div_names,
                       "population_2022": [int(p) for p in div_pop],
                       "under5_share": np.round(div_u5, 4),
                       "r0_mult": div_r0, "cfr_mult": div_cfr,
                       "cases_averted_per_100k": np.round(per_child, 1),
                       "ci_lo": np.round(ci_lo, 1), "ci_hi": np.round(ci_hi, 1)})
    eq.to_csv(os.path.join(TAB, "table_equity_by_division.csv"), index=False)
    SUMMARY["equity_gini"] = float(gini)
    SUMMARY["equity_maxmin"] = float(maxmin)
    SUMMARY["n_flip_types"] = int(len(flip_df))
    SUMMARY["n_chw_off_frontier"] = n_chw_offfront
    return eq, flip_df


def _gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    cum = np.cumsum(x)
    return (n + 1 - 2 * (cum / cum[-1]).sum()) / n


def _division_weights(pops, div_names):
    """Census-share weight of every sampled district in every division.

    Each district carries a Dirichlet division mix drawn from the real census
    population shares, so the natural division-level estimator is a weighted mean
    over ALL districts using those shares as weights. Assigning each district to
    its single largest division instead would starve the smaller divisions of
    districts entirely, which is what produced empty divisions and an infinite
    max-to-min ratio.
    """
    W = np.column_stack([pops[f"div_{d}"].values for d in div_names]).astype(float)
    return W / np.maximum(W.sum(0, keepdims=True), 1e-12)


def _predict_averted(pops, models, scaler, labels=None, intervention=None):
    """Metamodel-predicted measles cases averted per 100k for every district,
    either under each district's own recommended intervention or under one
    common intervention."""
    Xs = scaler.transform(pops[["mr1_before", "mr2_before", "sia_before"]].values)
    pred = np.zeros(len(pops))
    if intervention is not None:
        pred = ens_predict(models[(intervention, "measles_averted")], Xs).mean(0)
    else:
        for k, iv in enumerate(C.INTERVENTIONS):
            m = labels == k
            if m.any():
                pred[m] = ens_predict(models[(iv, "measles_averted")], Xs[m]).mean(0)
    return pred


def _division_outcomes(pops, models, scaler, div_names, div_r0, div_cfr, n_boot=500):
    """Cases averted per 100k by division under each district's own recommended
    intervention, predicted by the metamodel and scaled by that division's real
    transmission multiplier, with a genuine bootstrap over districts."""
    W = _division_weights(pops, div_names)
    pred = _predict_averted(pops, models, scaler, labels=pops["cvar_label"].values)

    per = np.zeros(len(div_names))
    lo = np.zeros(len(div_names)); hi = np.zeros(len(div_names))
    boot_mat = np.zeros((n_boot, len(div_names)))
    for i in range(len(div_names)):
        # burden, and therefore cases averted, scales with LOCAL transmission
        vals = pred * (0.9 + 0.1 * div_r0[i])
        per[i] = float(np.sum(W[:, i] * vals))
        for b in range(n_boot):
            bi = rng.integers(0, len(pops), len(pops))
            wb = W[bi, i]
            wb = wb / max(wb.sum(), 1e-12)
            boot_mat[b, i] = float(np.sum(wb * vals[bi]))
        lo[i] = float(np.percentile(boot_mat[:, i], 2.5))
        hi[i] = float(np.percentile(boot_mat[:, i], 97.5))
    # Basic (pivotal) bootstrap interval. The Gini of a small set of estimated
    # means is biased upward under resampling, because bootstrap noise adds
    # dispersion, so a plain percentile interval can exclude the point estimate.
    gini_hat = _gini(per)
    gini_bs = np.array([_gini(boot_mat[b]) for b in range(n_boot)])
    lo_p, hi_p = np.percentile(gini_bs, [2.5, 97.5])
    gini_ci = [round(float(max(0.0, 2 * gini_hat - hi_p)), 4),
               round(float(max(0.0, 2 * gini_hat - lo_p)), 4)]
    return {"per_100k": per, "ci_lo": lo, "ci_hi": hi, "gini_ci": gini_ci, "pred": pred}


def _allocation_objectives(pops, models, scaler, div_names, div_r0, div_cfr,
                           div_u5, div_pop):
    """Compare allocation objectives under a COMMON budget constraint.

    A planner chooses one intervention per division. Every objective faces the
    same under-5 weighted budget per 100k, so an objective that improves the
    least served division can do so only by giving something up elsewhere; that
    trade-off is the point of the comparison and the shipped version could not
    express it, because it applied unconstrained multiplicative bonuses.
    Outcomes and costs are metamodel predictions aggregated with the real census
    division weights.

    Under the estimated costs the four objectives converge to the SAME
    per-division allocation, and this is a substantive result rather than a bug:
    MR at six months is at once the largest-impact and the most equitable option
    that fits the shared budget in every division, so total-impact, maximin and
    high-CFR-priority objectives all select it everywhere and there is no
    allocation to trade. The function detects and flags this convergence
    (SUMMARY['allocation_objectives_coincide']) instead of hiding it, and the
    manuscript reports it as the finding. A genuine trade-off appears only if the
    budget is tightened below the cost of extending that option to every
    division, which is a different policy question (a hard rationing constraint)
    from the one asked here.
    """
    Xs = scaler.transform(pops[["mr1_before", "mr2_before", "sia_before"]].values)
    W = _division_weights(pops, div_names)
    nD, nI = len(div_names), len(C.INTERVENTIONS)

    averted = np.zeros((nD, nI))     # cases averted per 100k
    cost = np.zeros((nD, nI))        # USD per 100k
    for k, iv in enumerate(C.INTERVENTIONS):
        m_pred = ens_predict(models[(iv, "measles_averted")], Xs).mean(0)
        c_pred = ens_predict(models[(iv, "cost_total")], Xs).mean(0)
        for i in range(nD):
            averted[i, k] = float(np.sum(W[:, i] * m_pred * (0.9 + 0.1 * div_r0[i])))
            cost[i, k] = float(np.sum(W[:, i] * c_pred))

    w = np.asarray(div_u5, float); w = w / w.sum()

    def evaluate(choice):
        a = np.array([averted[i, choice[i]] for i in range(nD)])
        c = float(np.sum(w * np.array([cost[i, choice[i]] for i in range(nD)])))
        return a, c

    # A budget that binds: enough to lift some divisions above the cheapest
    # option, but not enough to give every division the most expensive one.
    c_cheap = float(np.sum(w * cost[:, int(np.argmin(cost.mean(0)))]))
    c_rich = float(np.sum(w * cost.max(1)))
    budget = c_cheap + 0.35 * (c_rich - c_cheap)

    def greedy(score_fn):
        """Start from the cheapest option everywhere, then repeatedly buy the
        upgrade with the best score gain per additional dollar until no upgrade
        both improves the score and fits the shared budget."""
        cheapest = int(np.argmin(cost.mean(0)))
        choice = [cheapest] * nD
        base_score = score_fn(*evaluate(choice))
        while True:
            best, best_ratio, best_score = None, 1e-12, base_score
            _, spent = evaluate(choice)
            for i in range(nD):
                for k in range(nI):
                    if k == choice[i]:
                        continue
                    trial = list(choice); trial[i] = k
                    a_t, c_t = evaluate(trial)
                    if c_t > budget:
                        continue
                    gain = score_fn(a_t, c_t) - base_score
                    dc = c_t - spent
                    ratio = gain / dc if dc > 1e-9 else gain * 1e9
                    if gain > 0 and ratio > best_ratio:
                        best, best_ratio, best_score = trial, ratio, score_fn(a_t, c_t)
            if best is None:
                break
            choice, base_score = best, best_score
        return choice

    objectives = {}
    # best single intervention applied everywhere, within the same budget
    feas = [k for k in range(nI) if float(np.sum(w * cost[:, k])) <= budget]
    k_uni = max(feas, key=lambda k: float(np.sum(w * averted[:, k]))) if feas \
        else int(np.argmin(cost.mean(0)))
    objectives["national uniform"] = [k_uni] * nD
    objectives["total impact"] = greedy(lambda a, c: float(np.sum(w * a)))
    objectives["max min"] = greedy(lambda a, c: float(a.min()))
    objectives["high cfr priority"] = greedy(lambda a, c: float(np.sum(w * div_cfr * a)))

    rows = []
    for name, choice in objectives.items():
        a, c = evaluate(choice)
        rows.append({
            "objective": name,
            "gini": round(_gini(a), 3),
            "maxmin_ratio": round(float(a.max() / max(a.min(), 1e-9)), 3),
            "min_division_per_100k": round(float(a.min()), 1),
            "total_weighted_per_100k": round(float(np.sum(w * a)), 1),
            "budget_used_per_100k": round(c, 1),
            "n_divisions_upgraded": int(sum(1 for k in choice if k != int(np.argmin(cost.mean(0))))),
        })
    SUMMARY["allocation_budget_per_100k"] = round(budget, 1)

    # Whether the objectives actually diverge is itself the finding, so report it
    # rather than presenting four rows and leaving the reader to notice they are
    # identical. When one intervention is simultaneously the largest-impact and
    # the most equitable option that fits the shared budget, every objective
    # selects it in every division and there is no allocation trade-off to
    # exploit; the objectives coincide by construction, not by numerical
    # accident. The chosen-intervention vectors identify that case exactly.
    distinct_allocations = {tuple(choice) for choice in objectives.values()}
    SUMMARY["allocation_objectives_coincide"] = bool(len(distinct_allocations) == 1)
    SUMMARY["allocation_n_distinct"] = int(len(distinct_allocations))
    if SUMMARY["allocation_objectives_coincide"]:
        dom = int(objectives["total impact"][0])
        SUMMARY["allocation_dominant_intervention"] = C.INTERVENTION_LABELS[C.INTERVENTIONS[dom]]
    return rows


# ======================================================================
# STAGE 9  Decision-rule sensitivity sweep + ICER tail / CVaR alpha
# ======================================================================
def stage9_sensitivity(pops, models, scaler):
    print("Stage 9: decision-rule sensitivity sweep and ICER safeguards")
    rows = []

    def root_boundary(labels):
        tmp = pops.copy(); tmp["lab"] = labels
        clf = fit_tree(tmp, "lab")
        return TREE_FEATURES[clf.tree_.feature[0]], float(clf.tree_.threshold[0])

    # sweep delta
    for delta in C.DELTA_GRID:
        obj = _propagate_objectives(pops, models, scaler, C.DEFAULT_TARGET, C.CVAR_ALPHA,
                                    delta, C.WTP_THRESHOLD, float("inf"), joint=True, S_MC=C.S_MC)
        feat, thr = root_boundary(obj["cvar_label"])
        sh = _shares(obj["cvar_label"])
        rows.append({"parameter": "delta", "value": delta, **sh,
                     "root_feature": feat, "root_threshold": round(thr, 3)})
    # sweep cvar alpha
    for a in C.CVAR_ALPHA_GRID:
        obj = _propagate_objectives(pops, models, scaler, C.DEFAULT_TARGET, a,
                                    C.DELTA_ROBUST, C.WTP_THRESHOLD, float("inf"), joint=True, S_MC=C.S_MC)
        feat, thr = root_boundary(obj["cvar_label"])
        sh = _shares(obj["cvar_label"])
        rows.append({"parameter": "cvar_alpha", "value": a, **sh,
                     "root_feature": feat, "root_threshold": round(thr, 3)})
    # sweep immunity target
    for tname in C.TARGET_GRID:
        obj = _propagate_objectives(pops, models, scaler, tname, C.CVAR_ALPHA,
                                    C.DELTA_ROBUST, C.WTP_THRESHOLD, float("inf"), joint=True, S_MC=C.S_MC)
        feat, thr = root_boundary(obj["cvar_label"])
        sh = _shares(obj["cvar_label"])
        rows.append({"parameter": "immunity_target", "value": tname, **sh,
                     "root_feature": feat, "root_threshold": round(thr, 3)})
    # sweep WTP
    for wtp in C.WTP_GRID:
        obj = _propagate_objectives(pops, models, scaler, C.DEFAULT_TARGET, C.CVAR_ALPHA,
                                    C.DELTA_ROBUST, wtp, float("inf"), joint=True, S_MC=C.S_MC)
        feat, thr = root_boundary(obj["cvar_label"])
        sh = _shares(obj["cvar_label"])
        rows.append({"parameter": "wtp", "value": wtp, **sh,
                     "root_feature": feat, "root_threshold": round(thr, 3)})
    # sweep budget cap
    for bud in C.BUDGET_GRID:
        obj = _propagate_objectives(pops, models, scaler, C.DEFAULT_TARGET, C.CVAR_ALPHA,
                                    C.DELTA_ROBUST, C.WTP_THRESHOLD, bud, joint=True, S_MC=C.S_MC)
        feat, thr = root_boundary(obj["cvar_label"])
        sh = _shares(obj["cvar_label"])
        rows.append({"parameter": "budget_cap", "value": ("inf" if not np.isfinite(bud) else bud), **sh,
                     "root_feature": feat, "root_threshold": round(thr, 3)})
    # sweep the DALY denominator floor. The floor is a modelling choice that
    # directly shapes the upper tail the CVaR objective acts on, so leaving it
    # out of a sweep that claims to cover every decision parameter would hide the
    # one parameter most entangled with the objective.
    for dmin in C.DMIN_GRID:
        obj = _propagate_objectives(pops, models, scaler, C.DEFAULT_TARGET, C.CVAR_ALPHA,
                                    C.DELTA_ROBUST, C.WTP_THRESHOLD, float("inf"),
                                    joint=True, S_MC=C.S_MC, dmin=dmin)
        feat, thr = root_boundary(obj["cvar_label"])
        sh = _shares(obj["cvar_label"])
        rows.append({"parameter": "daly_floor", "value": dmin, **sh,
                     "root_feature": feat, "root_threshold": round(thr, 3)})

    # sweep the derivation of the division transmission multiplier. Observed
    # outbreak incidence confounds transmissibility with accumulated
    # susceptibility, so the weight given to it is a modelling judgement; the
    # three derivations are reported side by side rather than assumed.
    for mode in C.R0_MULT_MODE_GRID:
        D = C._division_derived(mode)
        dn = list(D.keys())
        r0_m = np.array([D[d][2] for d in dn]); cfr_m = np.array([D[d][3] for d in dn])
        W = np.column_stack([pops[f"div_{d}"].values for d in dn])
        tmp = pops.copy()
        tmp["mean_r0_mult"] = W @ r0_m
        tmp["mean_cfr_mult"] = W @ cfr_m
        obj = _propagate_objectives(tmp, models, scaler, C.DEFAULT_TARGET, C.CVAR_ALPHA,
                                    C.DELTA_ROBUST, C.WTP_THRESHOLD, float("inf"),
                                    joint=True, S_MC=C.S_MC)
        t2 = tmp.copy(); t2["lab"] = obj["cvar_label"]
        clf = fit_tree(t2, "lab")
        feat = TREE_FEATURES[clf.tree_.feature[0]]; thr = float(clf.tree_.threshold[0])
        sh = _shares(obj["cvar_label"])
        rows.append({"parameter": "r0_multiplier_source", "value": mode, **sh,
                     "root_feature": feat, "root_threshold": round(thr, 3)})

    sens = pd.DataFrame(rows)
    sens.to_csv(os.path.join(TAB, "table_decision_sensitivity.csv"), index=False)

    # sensitivity heatmap: root threshold across (delta, cvar_alpha)
    grid = np.full((len(C.DELTA_GRID), len(C.CVAR_ALPHA_GRID)), np.nan)
    for i, delta in enumerate(C.DELTA_GRID):
        for k, a in enumerate(C.CVAR_ALPHA_GRID):
            obj = _propagate_objectives(pops, models, scaler, C.DEFAULT_TARGET, a,
                                        delta, C.WTP_THRESHOLD, float("inf"), joint=True, S_MC=C.S_MC)
            _, thr = root_boundary(obj["cvar_label"])
            grid[i, k] = thr
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(C.CVAR_ALPHA_GRID)))
    ax.set_xticklabels([f"{a:.2f}" for a in C.CVAR_ALPHA_GRID])
    ax.set_yticks(range(len(C.DELTA_GRID)))
    ax.set_yticklabels([f"{d:.2f}" for d in C.DELTA_GRID])
    ax.set_xlabel("CVaR level alpha")
    ax.set_ylabel("Robust tolerance delta")
    for i in range(grid.shape[0]):
        for k in range(grid.shape[1]):
            ax.text(k, i, f"{grid[i,k]:.3f}", ha="center", va="center",
                    color="white", fontsize=8)
    ax.set_title("Root MR1 split threshold across (delta, alpha)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig17_sensitivity_heatmap.png"), bbox_inches="tight")
    plt.close(fig)

    # ---- ICER tail behavior with denominator floor (reviewer #8) ----
    feats = pops[["mr1_before", "mr2_before", "sia_before"]].values
    Xs = scaler.transform(feats)
    j = int(np.argmax(pops["mr1_before"].values))   # high-coverage district: thin denominator
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for iv, col in zip(C.INTERVENTIONS, ["#2980b9", "#27ae60", "#c0392b"]):
        d_stack = ens_predict(models[(iv, "daly_averted")], Xs[j:j+1])
        c_stack = ens_predict(models[(iv, "cost_total")], Xs[j:j+1])
        d_mu, d_sd = d_stack.mean(), d_stack.std() + 1e-6
        c_mu, c_sd = c_stack.mean(), c_stack.std() + 1e-6
        dsamp = np.maximum(rng.normal(d_mu, d_sd + 0.06 * abs(d_mu), 4000), C.ICER_DALY_FLOOR)
        csamp = np.maximum(rng.normal(c_mu, c_sd + 0.05 * abs(c_mu), 4000), 1.0)
        icer = csamp / dsamp
        ax.hist(np.clip(icer, 0, np.percentile(icer, 99)), bins=60, alpha=0.5,
                color=col, label=C.INTERVENTION_LABELS[iv], density=True)
    ax.set_xlabel("Cost per DALY averted (USD), high-coverage district")
    ax.set_ylabel("Density")
    ax.set_title(f"ICER distribution with DALY floor = {C.ICER_DALY_FLOOR:.0f}/100k")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig18_icer_tail.png"), bbox_inches="tight")
    plt.close(fig)

    SUMMARY["sensitivity_thresholds"] = {
        "delta": [round(v, 3) for v in sens[sens.parameter == "delta"]["root_threshold"].tolist()],
        "cvar_alpha": [round(v, 3) for v in sens[sens.parameter == "cvar_alpha"]["root_threshold"].tolist()],
    }
    return sens


# ======================================================================
# Descriptive tables
# ======================================================================
def write_descriptive_tables():
    rows = []
    for iv in C.INTERVENTIONS:
        rows.append({"intervention": C.INTERVENTION_LABELS[iv],
                     "fixed_cost_per_child_USD": C.C_FIXED[iv],
                     "delivery_cost_per_dose_USD": (C.COST_DELIVERY_SIA if iv == "sia_campaign"
                                                    else C.COST_DELIVERY_ROUTINE),
                     "vaccine_cost_per_dose_USD": C.COST_MR_DOSE})
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "table_intervention_costs.csv"), index=False)

    epi_rows = [
        ["Measles basic reproduction number R0", f"{C.R0_MEASLES_MEAN} (SD {C.R0_MEASLES_SD})",
         "Guerra et al. 2017 systematic review"],
        ["Rubella basic reproduction number R0", f"{C.R0_RUBELLA_MEAN} (SD {C.R0_RUBELLA_SD})",
         "Literature range 5 to 7"],
        ["Latent period", f"{C.LATENT_WEEKS * 7:.0f} days", "Standard measles natural history"],
        ["Infectious period", f"{C.INFECTIOUS_WEEKS * 7:.0f} days", "Standard measles natural history"],
        ["One dose efficacy VE1", f"{C.VE1_MEAN} (SD {C.VE1_SD})", "WHO position paper 2017"],
        ["Two dose efficacy VE2", f"{C.VE2_MEAN} (SD {C.VE2_SD})", "WHO position paper 2017"],
        ["Campaign dose efficacy VEs", f"{C.VE_SIA_MEAN} (SD {C.VE_SIA_SD})", "Campaign literature"],
        # Parameters below appear in Equations 15 to 20 and were absent from the
        # shipped table, so the equations referred to symbols the reader could
        # not resolve.
        ["Measles CFR under 5 (CFRu)", f"{C.CFR_MEASLES_U5}",
         "Portnoy et al. 2019; Sbarra et al. 2023; WHO SEARO 2026 reported 0.9 to 1.2 percent"],
        ["Measles CFR 5 and over (CFR5+)", f"{C.CFR_MEASLES_5PLUS}", "Portnoy et al. 2019"],
        ["Under 5 share of measles deaths (wu)", "0.80", "GBD 2019 age distribution"],
        ["Rubella infections in the first trimester window (phi)",
         f"{C.FRAC_RUBELLA_IN_WCBA_FIRST_TRI}", "Vynnycky et al. 2016"],
        ["CRS risk given first trimester rubella (psi)", f"{C.CRS_PER_RUBELLA_WCBA}",
         "Vynnycky et al. 2016"],
        ["Disability weight, acute measles", f"{C.DW_MEASLES}", "GBD 2019 disability weights"],
        ["Disability weight, CRS", f"{C.DW_CRS}", "GBD 2019 disability weights"],
        ["Duration of CRS disability", f"{C.DUR_CRS:.0f} years", "GBD 2019"],
        ["Discount rate (r)", f"{C.DISCOUNT}", "WHO CHOICE reference case"],
        ["Life expectancy", f"{C.LIFE_EXPECTANCY} years", "World Bank Bangladesh"],
        ["Mean age at measles death", f"{C.MEAN_AGE_MEASLES_DEATH} years", "GBD 2019"],
        ["DALY denominator floor (Dmin)", f"{C.ICER_DALY_FLOOR} per 100k (swept)",
         "Modelling choice; swept in Table 14"],
        ["SMS gap closure share", f"{C.EFF_SMS_MEAN} (SD {C.EFF_SMS_SD})", "Reminder/recall meta-analyses"],
        ["CHW gap closure share (base)", f"{C.EFF_CHW_MEAN} (SD {C.EFF_CHW_SD})", "Outreach evaluations"],
        ["CHW low coverage slope", f"{C.EFF_CHW_LOWCOV}", "Outreach evaluations"],
        ["SIA reach of susceptibles", f"{C.EFF_SIA_MEAN} (SD {C.EFF_SIA_SD})", "Post-campaign coverage surveys"],
    ]
    pd.DataFrame(epi_rows, columns=["parameter", "value", "source"]).to_csv(
        os.path.join(TAB, "table_epi_parameters.csv"), index=False)

    div_rows = [{"division": d,
                 "population_2022": int(C.DIVISIONS_REAL[d][0]),
                 "under5_share": round(C.DIVISION_U5_SHARE[d], 4),
                 "urban_frac": C.DIVISIONS_REAL[d][1],
                 "incidence_per_million_2026": C.DIVISIONS_REAL[d][3],
                 "r0_mult": C.DIVISIONS[d][2], "cfr_mult": C.DIVISIONS[d][3],
                 "r0_mult_source": C.R0_MULT_MODE}
                for d in C.DIVISIONS]
    pd.DataFrame(div_rows).to_csv(os.path.join(TAB, "table_divisions.csv"), index=False)

    pd.DataFrame([{"key": k, "description": v} for k, v in C.DATA_SOURCES.items()]).to_csv(
        os.path.join(TAB, "table_data_sources.csv"), index=False)


# ======================================================================
# MAIN
# ======================================================================
def main():
    scen, grid = stage1_scenarios()
    df, draw_records, corr_summary = stage2_simulate(scen)
    (models, scaler, aleatoric, acc, calib, baseline, split) = stage4_metamodel(scen, draw_records)
    pops, div_names, div_u5, div_r0, div_cfr, div_pop = build_populations()
    pops = stage5_objectives(pops, models, scaler)
    det, cvar, st, agree = stage6_trees(pops)
    dominant, med, ci, bt, stab = stage7_boundary(pops, models, scaler, scen, draw_records, split)
    eq, flip_df = stage8_compare_equity(pops, det, cvar, div_names, div_u5, div_r0, div_cfr,
                                        div_pop, models, scaler)
    sens = stage9_sensitivity(pops, models, scaler)
    write_descriptive_tables()

    with open(os.path.join(RES, "results_summary.json"), "w") as f:
        json.dump(SUMMARY, f, indent=2)
    print("\n=== RESULTS SUMMARY ===")
    print(json.dumps(SUMMARY, indent=2))
    print("\nDone. Figures, tables, data and results written.")


if __name__ == "__main__":
    main()
