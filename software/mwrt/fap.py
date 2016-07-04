"""Create fast absorption predictor with full forward differentiation.

Murphy, D. M., and T. Koop, 2005: Review of the vapour pressures of ice and
    supercooled water for atmospheric applications. Q. J. R. Meteorol. Soc.,
    131, 1539–1565, doi:10.1256/qj.04.94.
"""

import numpy as np

from mwrt.model import Value


def qtot(lnq):
    """Total specific water content from lnq."""
    out = np.exp(lnq)
    return Value(fwd=out, dT=0., dlnq=out)


def density(p, T):
    """Density of air according to ideal gas law (ρ = p/R/T).
    
    The specific gas constant is chosen as 288 J/kg/K. The additional 1 J/kg/K
    relative to dry air takes into account water vapor (which has 461.5 J/kg/K)
    without the need to calculate e explicitly. The errors of this
    approximation should be below 2 %.
    """
    out = p * 100 / 288 / T # p is given in hPa
    return Value(fwd=out, dT=-out/T, dlnq=0.)


def esat(T):
    """Saturation water vapor pressure over water.
    
    Formula taken from Murphy and Koop (2005) who have taken it from the US
    Meteorological Handbook (1997). It is supposed to be valid for
    temperatures down to -50°C.
    """
    TT = T - 32.18
    out = 6.1121 * np.exp(17.502 * (T-273.15) / TT)
    return Value(fwd=out, dT=4217.45694/TT/TT*out, dlnq=0.)


def qsat(p, T):
    """Saturation specific humidity."""
    esat_ = esat(T)
    factor = 0.622/p
    return Value(
            fwd = factor * esat_.fwd,
            dT = factor * esat_.dT,
            dlnq = factor * esat_.dlnq
            )


def rh(qtot, qsat):
    """Fraction of total specific water and saturation specific humidity."""
    return Value(
            fwd = qtot.fwd / qsat.fwd,
            dT = (qtot.dT*qsat.fwd - qtot.fwd*qsat.dT) / qsat.fwd**2,
            dlnq = (qtot.dlnq*qsat.fwd - qtot.fwd*qsat.dlnq) / qsat.fwd**2
            )


def partition_lnq(p, T, lnq):
    """Separate specific water into vapor and liquid components."""
    qtot_ = qtot(lnq)
    qsat_ = qsat(p, T)
    rh_ = rh(qtot_, qsat_)
    mid = (0.95 <= rh_.fwd) & (rh_.fwd <= 1.05)
    high = 1.05 < rh_.fwd
    # Calculate partition function
    fliq = np.zeros_like(p)
    fliq_drh = np.zeros_like(p)
    # 0.95 <= s <= 1.05: transition into cloud starting at RH = 95%
    fliq[mid] = 0.5*(rh_.fwd[mid]-0.95-0.1/np.pi*np.cos(10*np.pi*rh_.fwd[mid]))
    fliq_drh[mid] = np.cos(5*np.pi*(rh_.fwd[mid]-1.05))**2
    # s > 1.05: RH is capped at 100% from here on only cloud water increases
    fliq[high] = rh_.fwd[high] - 1.
    fliq_drh[high] = 1.
    # Multiply with qsat to obtain specific amount
    qliq = Value(
            fwd = qsat_.fwd * fliq,
            #             product rule         (  chain rule     )
            dT = qsat_.dT * fliq + qsat_.fwd * (rh_.dT * fliq_drh),
            dlnq = qsat_.dlnq * fliq + qsat_.fwd * (rh_.dlnq * fliq_drh),
            )
    # All water that's not liquid is vapor
    qvap = Value(
            fwd = qtot_.fwd - qliq.fwd,
            dT = qtot_.dT - qliq.dT,
            dlnq = qtot_.dlnq - qliq.dlnq
            )
    return qvap, qliq



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

