"""Training and code generation gaseous and cloud FAPs."""

from functools import wraps

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

from mwrt.interpolation import atanspace
from mwrt.fap import partition_lnq, density, esat, qsat


__all__ = ["as_absorption", "generate_code", "CloudFAPGenerator",
        "GasFAPGenerator"]


def as_absorption(refractivity):
    """Convert a refractivity model to an absorption model.
    
    This function is supposed to be used to provide input for the model
    argument of the fit methods of FAPGenerators.
    """
    @wraps(refractivity)
    def absorption(ν, *args, **kwargs):
        N = refractivity(ν, *args, **kwargs)
        return 4 * np.pi * ν * 1.0e9 / 299792458. * np.imag(N)
    return absorption


def absorption_model(refractivity_gaseous, refractivity_lwc):
    """Make a full absorption model from gas and liquid refractivity models.

    Use for performance evaluation of a FAP.
    """
    def absorption(ν, p, T, lnq):
        θ = 300 / T
        qvap, qliq = partition_lnq(p, T, lnq)
        qsat_ = qsat(p=p, T=T)
        e = qvap/qsat_ * esat(T=T)
        ρliq = qliq * density(p=p, T=T)
        N = refractivity_gaseous(ν, θ, p-e, e) + ρliq * refractivity_lwc(ν, θ)
        return 4 * np.pi * ν * 1.0e9 / 299792458. * np.imag(N)
    return absorption


def generate_code(name, gasfap, cloudfap, with_import=True):
    """Generate code for a complete FAP based on the given to components.
    
    Generated is a class inheriting from mwrt.fap.FastAbsorptionPredictor
    with cloud and gas absorption methods implemented. The code can
    directly be executed with exec() or written to a file for later use.
    """
    out = []
    if with_import:
        out.append("from mwrt import FastAbsorptionPredictor\n\n\n")
    out.append("class {}(FastAbsorptionPredictor):\n\n".format(name))
    out.append(gasfap.generate_method())
    out.append("\n")
    out.append(cloudfap.generate_method())
    out.append("\n")
    return "".join(out)


class FAPGenerator:
    """Base class for fast absorption predictor generators."""

    def __init__(self, degree=3, alpha=0.01):
        """A new FAPGenerator instance.

        Emits a polynomial of given degree in the state vector variables p, T
        and lnq, trained with a Ridge-regression model (regularization
        parameter alpha).
        """
        self.pf = PolynomialFeatures(degree=degree, include_bias=True)
        # Intercept is modelled by column of 1s in PolyFeatures
        self.lm = Ridge(alpha=alpha, fit_intercept=False)

    @staticmethod
    def _generate_terms(coeffs, powers, names):
        """Generate code for calculation of polynomial."""
        out = []
        for coeff, power in zip(coeffs, powers):
            if coeff == 0:  continue
            term = [("- {}" if coeff < 0 else "+ {}").format(abs(coeff))]
            for name, pwr in zip(names, power):
                if pwr == 0: continue
                elif pwr == 1: term.append(name)
                else: term.append("{}**{}".format(name, pwr))
            out.append(" * ".join(term))
        return out if out else " 0."


class CloudFAPGenerator(FAPGenerator):
    """Train a FAP for specific absorption by cloud liquid water.
    
    Only the dependence on temperature is modelled, both Liebe et al. (1993)
    and Turner et al. (2016) describe absorption by liquid water only as
    a function of temperature.
    """

    train_T_range = 233., 303., 500000

    def fit(self, model, predictors=None):
        """Determine the fit coefficients.
        
        Targets are provided by calling model on the training data (temperature
        is converted to inverse temperature θ=300/T before evaluation). These
        are generated automatically (the range and amount can be controlled
        with train_T_range) or can be given with the predictors attribute.
        """
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
        out.append("    def cloud_absorption(T):\n")
        out.append("        return (")
        out.append("\n                ".join(self._generate_terms(
                self.lm.coef_, self.pf.powers_, ["T"])))
        out.append("\n                )\n")
        return "".join(out)


class GasFAPGenerator(FAPGenerator):
    """Input: p, T, q."""

    train_p_range = 80., 980., 100
    train_T_range = 170., 313., 100
    train_rh_range = 0., 1., 120 # Everything > 1 is liquid

    def fit(self, model, predictors=None):
        """Determine the fit coefficients.

        Targets are provided by calling model on the training data (temperature
        is converted to inverse temperature θ=300/T before evaluation). These
        are generated automatically (the range and amount can be controlled
        with train_p_range, train_T_range and train_rh_range) or can be given
        with the predictors attribute. The predictor input data are formulated
        in terms of p, T and RH to achive a more realistic training data range.
        """
        if predictors is None:
            predictors = self.generate_predictor_data()
        assert predictors.shape[1] == 3
        self.pf.fit(predictors)
        θ = 300/predictors[:,1]
        e = (predictors[:,2]/qsat(p=predictors[:,0], T=predictors[:,1])
                * esat(T=predictors[:,1]))
        target = model(θ=θ, pd=predictors[:,0]-e, e=e).flatten()
        self.lm.fit(self.pf.transform(predictors), target)

    def generate_predictor_data(self):
        from sklearn.utils.extmath import cartesian
        ps = np.linspace(*self.train_p_range)
        Ts = np.linspace(*self.train_T_range)
        rhs = atanspace(*self.train_rh_range, scaling=2.5)
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
        data[:,2] = data[:,2] * qsat(p=data[:,0], T=data[:,1])
        return data

    def generate_method(self):
        out = []
        out.append("    @staticmethod\n")
        out.append("    def gas_absorption(p, T, qvap):\n")
        out.append("        return (")
        out.append("\n                ".join(self._generate_terms(
                self.lm.coef_, self.pf.powers_, ["p", "T", "qvap"])))
        out.append("\n                )\n")
        return "".join(out)

