"""Stochastic mortality and longevity risk study: England & Wales males, ages 55-89, 1961-2011."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models import fit_cbd, fit_lee_carter, fit_lee_carter_svd, poisson_loglik
from src.projection import (annuity_values, cbd_rates, cohort_survival, cost_of_capital_premium,
                            lc_rates, life_expectancy, period_life_expectancy, simulate_cbd, simulate_lc)

ROOT = Path(__file__).parent
TABLES, FIGS = ROOT / "results" / "tables", ROOT / "results" / "figures"
for folder in (TABLES, FIGS):
    folder.mkdir(parents=True, exist_ok=True)
NAVY, RED, GREY, GOLD = "#1F3864", "#C0392B", "#7F8C8D", "#D4A017"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

AGES = np.arange(55, 90)
N_SIMS, HORIZON, RATE, SEED = 10_000, 60, 0.02, 2026
SF_LONGEVITY_SHOCK = 0.20            # Solvency II standard formula: permanent 20% fall in mortality rates


def load():
    D = pd.read_csv(ROOT / "data" / "ew_male_deaths.csv", index_col=0)
    E = pd.read_csv(ROOT / "data" / "ew_male_exposures.csv", index_col=0)
    years = D.columns.astype(int).values
    return D.loc[AGES].values, E.loc[AGES].values, years


def project(model, fit, n_sims, rng, noise=True):
    """Simulated rate surfaces (sims x ages x horizon) up to age 110."""
    if model == "LC":
        if noise:
            paths, _ = simulate_lc(fit, HORIZON, n_sims, rng)
        else:
            drift = np.diff(fit["k"]).mean()
            paths = (fit["k"][-1] + drift * np.arange(1, HORIZON + 1))[None, :]
        return lc_rates(fit, AGES, paths)
    if noise:
        paths, _ = simulate_cbd(fit, HORIZON, n_sims, rng)
    else:
        kap = np.vstack([fit["kappa1"], fit["kappa2"]])
        drift = np.diff(kap, axis=1).mean(axis=1)
        paths = (kap[:, -1][None, :] + np.outer(np.arange(1, HORIZON + 1), drift))[None, :, :]
    return cbd_rates(fit, AGES, paths)


def backtest(D, E, years, cutoff=1996, rng=None):
    """Fit on years <= cutoff, forecast the rest, compare with observed rates."""
    tr = years <= cutoff
    h = (~tr).sum()
    actual = D[:, ~tr] / E[:, ~tr]
    out, fans = [], {}
    for name in ["LC", "CBD"]:
        fit = fit_lee_carter(D[:, tr], E[:, tr]) if name == "LC" else fit_cbd(D[:, tr], E[:, tr], AGES)
        if name == "LC":
            paths, _ = simulate_lc(fit, h, N_SIMS, rng)
            m = np.exp(fit["a"][None, :, None] + fit["b"][None, :, None] * paths[:, None, :])
        else:
            paths, _ = simulate_cbd(fit, h, N_SIMS, rng)
            from scipy.special import expit
            lin = paths[:, None, :, 0] + (AGES - fit["x_bar"])[None, :, None] * paths[:, None, :, 1]
            m = -np.log1p(-expit(lin))
        lo, med, hi = np.quantile(m, [0.025, 0.5, 0.975], axis=0)
        inside = (actual >= lo) & (actual <= hi)
        out.append({"model": name, "coverage_95": inside.mean(),
                    "share_below_lower_band": (actual < lo).mean(),
                    "mean_log_error": np.log(actual / med).mean(),
                    "rmse_log": np.sqrt((np.log(actual / med) ** 2).mean())})
        fans[name] = (lo, med, hi)
    return pd.DataFrame(out), fans, actual, years[~tr], years[tr]


def main():
    rng = np.random.default_rng(SEED)
    D, E, years = load()

    # 1. Fit and compare models on a common Poisson likelihood
    lc, cbd, svd = fit_lee_carter(D, E), fit_cbd(D, E, AGES), fit_lee_carter_svd(D, E)
    score_gap = np.max(np.abs((D - E * lc["m"]).sum(axis=1)) / D.sum(axis=1))
    comparison = pd.DataFrame([
        ["Lee-Carter (Poisson MLE)", lc["loglik"], lc["n_par"], lc["bic"]],
        ["Lee-Carter (SVD)", poisson_loglik(D, E, svd["m"]), lc["n_par"], np.nan],
        ["CBD (Poisson MLE)", cbd["loglik"], cbd["n_par"], cbd["bic"]],
    ], columns=["model", "log_likelihood", "parameters", "BIC"])
    comparison.to_csv(TABLES / "model_comparison.csv", index=False)

    # 2. Backtest: calibrate on 1961-1996, forecast 1997-2011
    bt, fans, actual, test_years, train_years = backtest(D, E, years, rng=rng)
    bt.to_csv(TABLES / "backtest.csv", index=False)

    # 3. Projections from 2011, cohort aged 65 in 2012
    results, annuity_dist, surv_mean = [], {}, {}
    m_hist = D / E
    for name, fit in [("LC", lc), ("CBD", cbd)]:
        m, all_ages = project(name, fit, N_SIMS, rng)
        S = cohort_survival(m, all_ages, 65)
        a = annuity_values(S, RATE)
        e = life_expectancy(S)
        m_c, _ = project(name, fit, 1, rng, noise=False)
        S_c = cohort_survival(m_c, all_ages, 65)
        q_c = 1 - np.exp(-m_c)
        S_shock = cohort_survival(-np.log1p(-(1 - SF_LONGEVITY_SHOCK) * q_c), all_ages, 65)
        a_c, a_shock = annuity_values(S_c, RATE)[0], annuity_values(S_shock, RATE)[0]
        prem6, cap = cost_of_capital_premium(S.mean(axis=0), a, RATE, 0.06)
        prem475, _ = cost_of_capital_premium(S.mean(axis=0), a, RATE, 0.0475)
        results.append({
            "model": name,
            "cohort_e65_mean": e.mean(), "cohort_e65_p0_5": np.quantile(e, 0.005), "cohort_e65_p99_5": np.quantile(e, 0.995),
            "annuity_mean": a.mean(), "annuity_p99_5": np.quantile(a, 0.995),
            "stochastic_runoff_capital_pct": cap,
            "sf_20pct_shock_capital_pct": a_shock / a_c - 1,
            "swap_premium_coc_6pct": prem6, "swap_premium_coc_4_75pct": prem475,
        })
        annuity_dist[name] = a
        surv_mean[name] = S.mean(axis=0)
        fans[f"proj_{name}"] = (m, all_ages)
    period_e65 = period_life_expectancy(lc_rates(lc, AGES, lc["k"][-1:][None, :])[0][0, :, 0],
                                        np.arange(55, 111), 65)
    proj = pd.DataFrame(results)
    proj.insert(1, "period_e65_2011", period_e65)
    proj.to_csv(TABLES / "projection_results.csv", index=False)

    # 4. Charts
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    axes[0, 0].plot(AGES, lc["a"], color=NAVY); axes[0, 0].set_title("Lee-Carter a(x): age profile"); axes[0, 0].set_xlabel("Age")
    axes[0, 1].plot(AGES, lc["b"], color=NAVY); axes[0, 1].set_title("Lee-Carter b(x): sensitivity to k(t)"); axes[0, 1].set_xlabel("Age")
    axes[0, 2].plot(years, lc["k"], color=NAVY); axes[0, 2].set_title("Lee-Carter k(t): period index"); axes[0, 2].set_xlabel("Year")
    axes[1, 0].plot(years, cbd["kappa1"], color=RED); axes[1, 0].set_title("CBD kappa1(t): level"); axes[1, 0].set_xlabel("Year")
    axes[1, 1].plot(years, cbd["kappa2"], color=RED); axes[1, 1].set_title("CBD kappa2(t): slope by age"); axes[1, 1].set_xlabel("Year")
    for x in (65, 80):
        i = x - AGES[0]
        axes[1, 2].scatter(years, np.log(m_hist[i]), s=8, color=GREY)
        axes[1, 2].plot(years, np.log(lc["m"][i]), color=NAVY, lw=1)
        axes[1, 2].plot(years, np.log(cbd["m"][i]), color=RED, lw=1, ls="--")
    axes[1, 2].set_title("Fit at ages 65 and 80 (LC solid, CBD dashed)"); axes[1, 2].set_xlabel("Year"); axes[1, 2].set_ylabel("log m")
    fig.suptitle("Fitted parameters, England & Wales males, ages 55-89, 1961-2011")
    fig.tight_layout(); fig.savefig(FIGS / "fitted_parameters.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
    for ax, x in zip(axes, (65, 75, 85)):
        i = x - AGES[0]
        ax.plot(train_years, m_hist[i, :len(train_years)], color=GREY, lw=1, label="Observed (fit)")
        for name, col in [("LC", NAVY), ("CBD", RED)]:
            lo, med, hi = fans[name]
            ax.fill_between(test_years, lo[i], hi[i], color=col, alpha=0.18)
            ax.plot(test_years, med[i], color=col, lw=1.3, label=f"{name} forecast, 95% band")
        ax.scatter(test_years, actual[i], color="black", s=10, zorder=3, label="Observed (test)")
        ax.set_yscale("log"); ax.set_title(f"Age {x}"); ax.set_xlabel("Year")
    axes[0].set_ylabel("Central death rate (log scale)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Backtest: models fitted on 1961-1996, forecasting 1997-2011")
    fig.tight_layout(); fig.savefig(FIGS / "backtest.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    proj_years = years[-1] + np.arange(1, HORIZON + 1)
    for ax, x in zip(axes, (65, 85)):
        i = x - AGES[0]
        ax.plot(years, m_hist[i], color="black", lw=1, label="Observed")
        for name, col in [("LC", NAVY), ("CBD", RED)]:
            m, all_ages = fans[f"proj_{name}"]
            j = x - all_ages[0]
            lo, med, hi = np.quantile(m[:, j, :30], [0.025, 0.5, 0.975], axis=0)
            ax.fill_between(proj_years[:30], lo, hi, color=col, alpha=0.18)
            ax.plot(proj_years[:30], med, color=col, lw=1.3, label=f"{name} median, 95% band")
        ax.set_yscale("log"); ax.set_title(f"Projected death rate at age {x}"); ax.set_xlabel("Year")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "projection_fans.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    for name, col in [("LC", NAVY), ("CBD", RED)]:
        a = annuity_dist[name]
        ax.hist(a, bins=80, density=True, alpha=0.5, color=col, label=f"{name}: mean {a.mean():.2f}, 99.5% {np.quantile(a, 0.995):.2f}")
    ax.set_xlabel("Value of a whole-life annuity of 1 a year, male aged 65 in 2012 (2% discount)")
    ax.set_yticks([])
    ax.set_title("Longevity risk: distribution of the annuity value")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "annuity_distribution.png", dpi=150); plt.close(fig)

    # 5. Console summary
    pd.set_option("display.width", 200)
    print(f"Lee-Carter Poisson converged in {lc['iterations']} iterations; max score gap {score_gap:.1e}")
    print("\nModel comparison\n" + comparison.round(1).to_string(index=False))
    print("\nBacktest 1997-2011\n" + bt.round(3).to_string(index=False))
    print("\nProjections\n" + proj.round(4).T.to_string())


if __name__ == "__main__":
    main()
