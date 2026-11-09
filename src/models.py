"""Stochastic mortality models fitted from scratch on deaths D[x,t] and central exposures E[x,t].

* Lee-Carter (1992):            log m(x,t) = a_x + b_x k_t
  fitted by Poisson maximum likelihood (Brouhns, Denuit & Vermunt, 2002) and,
  for comparison, by the original SVD method.
* Cairns-Blake-Dowd (2006):     logit q(x,t) = kappa1_t + kappa2_t (x - x_bar)
  fitted year by year under the same Poisson likelihood, with m = -log(1 - q),
  so the two models' log-likelihoods and BIC are directly comparable.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammaln


def poisson_loglik(D, E, m):
    mu = E * m
    return float(np.sum(D * np.log(mu) - mu - gammaln(D + 1)))


def fit_lee_carter_svd(D, E):
    logm = np.log(D / E)
    a = logm.mean(axis=1)
    U, s, Vt = np.linalg.svd(logm - a[:, None], full_matrices=False)
    b, k = U[:, 0], s[0] * Vt[0]
    b, k = b / b.sum(), k * b.sum()
    return {"a": a, "b": b, "k": k, "m": np.exp(a[:, None] + np.outer(b, k))}


def fit_lee_carter(D, E, tol=1e-10, max_iter=5000):
    """Poisson Lee-Carter by Newton-Raphson updates; constraints sum(b)=1, sum(k)=0."""
    start = fit_lee_carter_svd(D, E)
    a, b, k = start["a"].copy(), start["b"].copy(), start["k"].copy()
    ll_old = -np.inf
    for it in range(max_iter):
        Dhat = E * np.exp(a[:, None] + np.outer(b, k))
        a += (D - Dhat).sum(axis=1) / Dhat.sum(axis=1)
        Dhat = E * np.exp(a[:, None] + np.outer(b, k))
        k += ((D - Dhat) * b[:, None]).sum(axis=0) / (Dhat * b[:, None] ** 2).sum(axis=0)
        shift = k.mean(); k -= shift; a += b * shift
        Dhat = E * np.exp(a[:, None] + np.outer(b, k))
        b += ((D - Dhat) * k[None, :]).sum(axis=1) / (Dhat * k[None, :] ** 2).sum(axis=1)
        m = np.exp(a[:, None] + np.outer(b, k))
        ll = poisson_loglik(D, E, m)
        if abs(ll - ll_old) < tol:
            break
        ll_old = ll
    scale = b.sum()
    b, k = b / scale, k * scale
    m = np.exp(a[:, None] + np.outer(b, k))
    n_ages, n_years = D.shape
    n_par = 2 * n_ages + n_years - 2
    ll = poisson_loglik(D, E, m)
    return {"a": a, "b": b, "k": k, "m": m, "loglik": ll, "n_par": n_par,
            "bic": -2 * ll + n_par * np.log(D.size), "iterations": it + 1}


def cbd_m(kappa1, kappa2, ages, x_bar):
    q = expit(kappa1[None, :] + np.outer(ages - x_bar, kappa2))
    return -np.log1p(-q)


def fit_cbd(D, E, ages):
    """CBD fitted year by year by Poisson maximum likelihood on m = -log(1 - q)."""
    x_bar = ages.mean()
    n_years = D.shape[1]
    k1, k2 = np.empty(n_years), np.empty(n_years)
    for t in range(n_years):
        q0 = np.clip(1 - np.exp(-D[:, t] / E[:, t]), 1e-6, 1 - 1e-6)
        start = np.polyfit(ages - x_bar, np.log(q0 / (1 - q0)), 1)[::-1]

        def nll(p):
            m = -np.log1p(-expit(p[0] + p[1] * (ages - x_bar)))
            mu = E[:, t] * m
            return -np.sum(D[:, t] * np.log(mu) - mu)

        res = minimize(nll, start, method="BFGS")
        k1[t], k2[t] = res.x
    m = cbd_m(k1, k2, ages, x_bar)
    n_par = 2 * n_years
    ll = poisson_loglik(D, E, m)
    return {"kappa1": k1, "kappa2": k2, "x_bar": x_bar, "m": m, "loglik": ll, "n_par": n_par,
            "bic": -2 * ll + n_par * np.log(D.size)}
