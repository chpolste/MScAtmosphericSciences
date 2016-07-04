"""Train fast absorption predictors for gaseous and cloud absorption."""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

from mwrt.fap import partition_lnq, density, esat, qsat


__all__ = ["as_absorption", "absorption_model", "CloudFAPGenerator",
        "GaseousFAPGenerator"]


def as_absorption(ν, refractivity):
    """"""
    def absorption(*args, **kwargs):
        N = refractivity(ν, *args, **kwargs)
        return 4 * np.pi * ν * 1.0e9 / 299792458. * np.imag(N)
    return absorption


def absorption_model(ν, refractivity_gaseous, refractivity_lwc):
    """"""
    def absorption(p, T, lnq):
        θ = 300 / T
        qvap, qliq = partition_lnq(p, T, lnq)
        qvap, qliq = qvap.fwd, qliq.fwd
        qsat_ = qsat(p=p, T=T).fwd
        e = qvap/qsat_ * esat(T=T).fwd
        ρliq = qliq * density(p=p, T=T).fwd
        N = refractivity_gaseous(ν, θ, p-e, e) + ρliq * refractivity_lwc(ν, θ)
        return 4 * np.pi * ν * 1.0e9 / 299792458. * np.imag(N)
    return absorption


class FAPGenerator:
    """"""

    def __init__(self, name="FAP", alpha=0.1, degree=3):
        """"""
        self.name = name
        self.pf = PolynomialFeatures(degree=degree, include_bias=True)
        # Intercept is modelled by column of 1s in PolyFeatures
        self.lm = Ridge(alpha=alpha, fit_intercept=False)

    @staticmethod
    def _generate_terms(coeffs, powers, names):
        out = []
        for coeff, power in zip(coeffs, powers):
            if coeff == 0:  continue
            term = [(" - {}" if coeff < 0 else " + {}").format(abs(coeff))]
            for name, pwr in zip(names, power):
                if pwr == 0: continue
                elif pwr == 1: term.append(name)
                else: term.append("{}**{}".format(name, pwr))
            out.append("*".join(term))
        return out if out else " 0."

    @staticmethod
    def _derive(coeffs, powers, component):
        if component >= powers.shape[1]:
            return np.zeros_like(coeffs), np.zeros_like(powers)
        out_coeffs = []
        out_powers = []
        for coeff, power in zip(coeffs, powers):
            deriv = power.copy()
            deriv[component] = max(deriv[component]-1, 0)
            out_coeffs.append(coeff*power[component])
            out_powers.append(deriv)
        return out_coeffs, out_powers
    

    @staticmethod
    def generate_code(fap_gaseous, fap_cloud):
        """"""


class CloudFAPGenerator(FAPGenerator):
    """Input: T."""

    train_T_range = (233., 303., 500000)

    def fit(self, model, predictors=None):
        if predictors is None:
            # Realistic cloud range is not greater than -40°C to 27°C
            predictors = np.linspace(*self.train_T_range).reshape(-1, 1)
        assert predictors.shape[1] == 1
        # Absorption/Refractivity models take inverse temperature as input
        target = model(θ=300./predictors).flatten()
        # The FAP is trained against "normal" temperature though
        self.pf.fit(predictors)
        self.lm.fit(self.pf.transform(predictors), target)

    def generate_method(self):
        out = []
        out.append("    @staticmethod\n")
        out.append("    def cloud_FAP(self, T):\n")
        out.append("        return VectorValue(\n")
        out.append("                fwd =")
        out.append("".join(self._generate_terms(self.lm.coef_, self.pf.powers_, ["T"])))
        out.append(",\n                dT =")
        out.append("".join(self._generate_terms(*self._derive(self.lm.coef_, self.pf.powers_, 0), ["T"])))
        out.append(",\n                dlnq = 0.\n")
        out.append("                )\n")
        return "".join(out)


class GaseousFAPGenerator(FAPGenerator):
    """Input: p, T, q."""

    train_p_range = 80., 980., 50
    train_T_range = 170., 330., 50
    train_rh_range = (0.0001, 1., 50) # Everything > 1 is liquid

    def fit(self, model, predictors=None):
        if predictors is None:
            predictors = self.generate_predictor_data()
        assert predictors.shape[1] == 3
        self.pf.fit(predictors)
        # Train against log of absorption coefficient to guarantee positivity
        θ = 300/predictors[:,1]
        e = (predictors[:,2]/qsat(p=predictors[:,0], T=predictors[:,1]).fwd
                * esat(T=predictors[:,1]).fwd)
        target = np.log(model(θ=θ, pd=predictors[:,0]-e, e=e)).flatten()
        self.lm.fit(self.pf.transform(predictors), target)

    def generate_predictor_data(self):
        from sklearn.utils.extmath import cartesian
        ps = np.linspace(*self.train_p_range)
        Ts = np.linspace(*self.train_T_range)
        rhs = np.linspace(*self.train_rh_range)
        data = cartesian([ps, Ts, rhs])
        # Remove some (for Innsbruck) unrealistic data
        remove = (
                # Lower atmosphere is rather warm
                ((data[:,0] > 700) & (data[:,1] < 230))
                # Middle atmosphere
                | ((data[:,0] < 700) & (data[:,0] > 400)
                    & (data[:,1] > 300) | (data[:,1] < 200))
                # Upper atmosphere is rather cold
                | ((data[:,0] < 400) & (data[:,1] > 270))
                )
        data = data[~remove]
        # Calculate q
        data[:,2] = data[:,2] * qsat(p=data[:,0], T=data[:,1]).fwd
        return data

    def generate_method(self):
        out = []
        out.append("    @staticmethod\n")
        out.append("    def gaseous_FAP(self, p, T, qvap):\n")
        out.append("        return VectorValue(\n")
        out.append("                fwd =")
        out.append("".join(self._generate_terms(self.lm.coef_, self.pf.powers_, ["p", "T", "qvap"])))
        out.append(",\n                dT =")
        out.append("".join(self._generate_terms(*self._derive(self.lm.coef_, self.pf.powers_, 1), ["p", "T", "qvap"])))
        out.append(",\n                dlnq =") # TODO: apply chain rule, this is currently wrong!
        out.append("".join(self._generate_terms(*self._derive(self.lm.coef_, self.pf.powers_, 2), ["p", "T", "qvap"])))
        out.append("\n                )\n")
        return "".join(out)
