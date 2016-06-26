"""Create fast absorption predictors."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.extmath import cartesian

import formulas as fml


def absorption_model(ν, refractivity_gaseous, refractivity_lwc):
    """"""
    def absorption(p, T, lnq):
        θ = 300 / T
        qsat, qvap, qliq = partition_lnq(p, T, lnq)
        e = qvap/qsat * fml.esat(T=T)
        ρliq = qliq * fml.ρ(p=p, T=T, e=e)
        N = refractivity_gaseous(ν, θ, p-e, e) + ρliq * refractivity_lwc(ν, θ)
        return 4 * np.pi * ν / fml.c0 * np.imag(N)
    return absorption


def partition_lnq(p, T, lnq):
    """Separate lnq into qsat, qvap and qliq."""
    qsat = fml.qsat(p=p, T=T)
    qtot = np.exp(lnq)
    rhtot = qtot/qsat
    rhvap = np.minimum(rhtot, 1.05)
    qliq = 0.5 * qsat * (rhvap - 0.95 - 0.1/np.pi*np.cos(np.pi*rhvap/0.1))
    try: qliq[rhtot<0.95] = 0
    except TypeError:
        if rhtot < 0.95: qliq = 0
    try: qliq[rhtot>1.05] += (qsat*(rhtot-1.05))[rhtot>1.05]
    except TypeError:
        if rhtot > 1.05: qliq += qsat*(rhtot-1.05)
    return qsat, qtot-qliq, qliq


def training_data():
    """Generate training data for FAP training."""
    # make Ts, ps and rhs then use cartesian and goff_gratch for lnq


def fit(model, training_data):
    """Determine regression coefficients."""


def codegen(coefficients, name):
    """Generate Python code of FAP from coefficients."""

