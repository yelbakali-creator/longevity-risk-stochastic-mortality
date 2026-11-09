"""Stochastic projection of mortality and longevity risk applications.

Period indexes follow random walks with drift. Drift uncertainty is included by
drawing the drift of each simulated path from its estimation distribution, so
the fan widens with both process noise and parameter error.
"""
import numpy as np
from scipy.special import expit

MAX_AGE = 110


def simulate_lc(fit, horizon, n_sims, rng, drift_uncertainty=True):
    k = fit["k"]
    dk = np.diff(k)
    mu, sigma = dk.mean(), dk.std(ddof=1)
    drift = mu + (rng.standard_normal(n_sims) * sigma / np.sqrt(len(dk)) if drift_uncertainty else 0.0)
    shocks = rng.standard_normal((n_sims, horizon)) * sigma
    paths = k[-1] + np.cumsum(drift[:, None] + shocks, axis=1)          # n_sims x horizon
    return paths, {"drift": mu, "sigma": sigma}


def lc_rates(fit, ages, k_paths):
    """Projected central rates; ages above the fitted range are extrapolated log-linearly in age."""
    logm = fit["a"][None, :, None] + fit["b"][None, :, None] * k_paths[:, None, :]   # sims x ages x years
    top = ages >= ages.max() - 9                                  # fit the age slope on the last 10 ages
    y = logm[:, top, :].transpose(1, 0, 2).reshape(top.sum(), -1)
    slope = np.polyfit(ages[top], y, 1)[0].reshape(logm.shape[0], logm.shape[2])
    extra = np.arange(ages.max() + 1, MAX_AGE + 1)
    ext = logm[:, -1:, :] + slope[:, None, :] * (extra - ages.max())[None, :, None]
    return np.exp(np.concatenate([logm, ext], axis=1)), np.concatenate([ages, extra])


def simulate_cbd(fit, horizon, n_sims, rng, drift_uncertainty=True):
    kap = np.vstack([fit["kappa1"], fit["kappa2"]])
    dk = np.diff(kap, axis=1)
    mu, cov = dk.mean(axis=1), np.cov(dk)
    L = np.linalg.cholesky(cov)
    drift = mu[None, :] + ((rng.standard_normal((n_sims, 2)) @ L.T) / np.sqrt(dk.shape[1])
                           if drift_uncertainty else 0.0)
    shocks = np.einsum("ij,snj->sni", L, rng.standard_normal((n_sims, horizon, 2)))
    paths = kap[:, -1][None, None, :] + np.cumsum(drift[:, None, :] + shocks, axis=1)   # sims x horizon x 2
    return paths, {"drift": mu, "cov": cov}


def cbd_rates(fit, ages, kappa_paths):
    all_ages = np.arange(ages.min(), MAX_AGE + 1)
    lin = kappa_paths[:, None, :, 0] + (all_ages - fit["x_bar"])[None, :, None] * kappa_paths[:, None, :, 1]
    return -np.log1p(-expit(lin)), all_ages


def cohort_survival(m, all_ages, age0):
    """Survival curve of the cohort aged age0 in the first projection year (diagonal of the rate surface)."""
    n_sims, _, horizon = m.shape
    idx = age0 - all_ages[0]
    steps = min(horizon, len(all_ages) - idx)
    q = np.stack([1 - np.exp(-m[:, idx + t, t]) for t in range(steps)], axis=1)
    return np.cumprod(1 - q, axis=1)                         # S(t+1) for t = 0..steps-1


def annuity_values(survival, rate):
    """Whole-life annuity of 1 a year paid in arrears while alive."""
    v = (1 + rate) ** -np.arange(1, survival.shape[1] + 1)
    return survival @ v


def life_expectancy(survival):
    """Complete life expectancy approximation: curtate expectancy + 0.5."""
    return survival.sum(axis=1) + 0.5


def period_life_expectancy(m_col, all_ages, age0):
    q = 1 - np.exp(-m_col[all_ages >= age0])
    return np.cumprod(1 - q).sum() + 0.5


def cost_of_capital_premium(survival_best, annuity_dist, rate, coc):
    """Indicative longevity swap risk premium by the cost-of-capital method.

    Capital at time 0 is the 99.5% run-off stress on the annuity value; future capital
    runs off in proportion to the best-estimate liability. The premium is the risk margin
    expressed as a loading on expected payments.
    """
    best = annuity_dist.mean()
    capital0 = np.quantile(annuity_dist, 0.995) - best
    n = len(survival_best)
    v = (1 + rate) ** -np.arange(1, n + 1)
    # Expected liability still to run at each future date, per initial annuitant
    bel_t = np.array([(survival_best[t:] * v[: n - t]).sum() for t in range(n)])
    capital_t = capital0 * bel_t / bel_t[0]
    risk_margin = coc * (capital_t * v).sum()
    return risk_margin / best, capital0 / best
