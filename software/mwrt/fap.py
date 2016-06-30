"""Create fast absorption predictors."""

import numpy as np

import formulas as fml


def absorption_model(ν, refractivity_gaseous, refractivity_lwc):
    """"""
    def absorption(p, T, lnq):
        θ = 300 / T
        qsat, qvap, qliq = partition_lnq(p, T, lnq)
        e = qvap/qsat * fml.esat(T=T)
        ρliq = qliq * fml.ρ(p=p, T=T, e=e)
        N = refractivity_gaseous(ν, θ, p-e, e) + ρliq * refractivity_lwc(ν, θ)
        return 4 * np.pi * ν * 1.0e9 / fml.c0 * np.imag(N)
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


def predictor_data(n):
    """Generate training data for FAP training."""
    from sklearn.utils.extmath import cartesian
    ps = np.linspace(80, 1000, n)
    Ts = np.linspace(170, 330, n)
    ss = np.linspace(0.0001, 1.3, n)
    data = cartesian([ps, Ts, ss])
    # Remove some unrealistic data
    remove = (
            # Lower atmosphere is rather warm
            ((data[:,0] > 700) & (data[:,1] < 230))
            # Middle atmosphere
            | ((data[:,0] < 700) & (data[:,0] > 400)
                & (data[:,1] > 300) | (data[:,1] < 200))
            # Upper atmosphere is rather cold
            | ((data[:,0] < 400) & (data[:,1] > 270))
            # No liquid water below -40 °C
            | ((data[:,1] < 233.15) & (data[:,2] > 1))
            )
    data = data[~remove]
    data[:,2] = np.log(data[:,2] * fml.qsat(p=data[:,0], T=data[:,1]))
    return data


class FastAbsorptionPredictor:
    """"""

    def __init__(self, model, *, alpha=1.0, degree=2, name="FAP"):
        """"""
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import Ridge
        self.name = name
        self.model = model
        self.degree = 2
        self.polyfeatures = PolynomialFeatures(degree=degree)
        # Intercept is modelled by columns of 1s in polyfeatures
        self.regression = Ridge(alpha=alpha, fit_intercept=False)

    def fit(self, predictors):
        """"""
        self.polyfeatures.fit(predictors)
        target = np.log10(self.model(p=predictors[:,0], T=predictors[:,1],
                lnq=predictors[:,2]))
        self.regression.fit(self.polyfeatures.transform(predictors), target)

    def generate_code(self):
        """"""
        out = []
        coeffs = self.regression.coef_
        powers = self.polyfeatures.powers_
        method_indent = " "*16
        out.append("class {}:\n".format(self.name))
        out.append('    """Fast Absorption Predictor."""\n\n')
        out.append("    @staticmethod\n")
        out.append("    def forward(p, T, lnq):\n")
        out.append("        return 10**(\n")
        out.append(method_indent)
        out.append(method_indent.join(self._generate_terms(coeffs, powers)))
        out.append("                )\n\n")
        out.append("    @staticmethod\n")
        out.append("    def jacobian_T(p, T, lnq):\n")
        out.append("        return np.diag(\n")
        out.append(method_indent)
        out.append(method_indent.join(self._generate_terms(
                *self._derive(coeffs, powers, 1))))
        out.append("                )\n\n")
        out.append("    @staticmethod\n")
        out.append("    def jacobian_lnq(p, T, lnq):\n")
        out.append("        return np.diag(\n")
        out.append(method_indent)
        out.append(method_indent.join(self._generate_terms(
                *self._derive(coeffs, powers, 2))))
        out.append("                )\n\n")
        return "".join(out)

    @staticmethod
    def _generate_terms(coeffs, powers):
        out = []
        for coeff, power in zip(coeffs, powers):
            if coeff == 0:  continue
            term = [("- {}" if coeff < 0 else "+ {}").format(abs(coeff))]
            for name, pwr in zip(["p", "T", "lnq"], power):
                if pwr == 0: continue
                elif pwr == 1: term.append(name)
                else: term.append("{}**{}".format(name, pwr))
            out.append(" * ".join(term))
            out.append("\n")
        return out

    @staticmethod
    def _print_power(powers, names):
        terms = []
        for name, power in zip(names, powers):
            if power == 0: continue
            elif power == 1: terms.append(name)
            else: terms.append("{}**{}".format(name, power))
        return " * ".join(terms)

    @staticmethod
    def _derive(coeffs, powers, component):
        out_coeffs = []
        out_powers = []
        for coeff, power in zip(coeffs, powers):
            deriv = power.copy()
            deriv[component] = max(deriv[component]-1, 0)
            out_coeffs.append(coeff*power[component])
            out_powers.append(deriv)
        return out_coeffs, out_powers

