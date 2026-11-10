"""Validation tests: Poisson Lee-Carter optimality, constraints, and CBD fit on real HMD data."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import fit_cbd, fit_lee_carter, fit_lee_carter_svd, poisson_loglik

ROOT = Path(__file__).parents[1]
AGES = np.arange(55, 90)
D = pd.read_csv(ROOT / "data" / "ew_male_deaths.csv", index_col=0).loc[AGES].values
E = pd.read_csv(ROOT / "data" / "ew_male_exposures.csv", index_col=0).loc[AGES].values


def test_lee_carter_score_equation():
    """At the Poisson MLE, fitted deaths equal observed deaths for every age."""
    lc = fit_lee_carter(D, E)
    gap = np.abs((D - E * lc["m"]).sum(axis=1)) / D.sum(axis=1)
    assert gap.max() < 1e-8


def test_lee_carter_constraints():
    lc = fit_lee_carter(D, E)
    assert abs(lc["b"].sum() - 1) < 1e-10
    assert abs(lc["k"].sum()) < 1e-8


def test_poisson_mle_beats_svd():
    lc, svd = fit_lee_carter(D, E), fit_lee_carter_svd(D, E)
    assert lc["loglik"] > poisson_loglik(D, E, svd["m"])


def test_cbd_fitted_rates_close_to_observed():
    cbd = fit_cbd(D, E, AGES)
    assert np.median(np.abs(np.log(cbd["m"] / (D / E)))) < 0.05
