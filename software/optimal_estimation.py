"""Optimal estimation tools and configuration.

Naming conventions:
x, μ - state vector (temperature on top of total water content)
y - radiometer observation (brightness temperatures)
"""

import numpy as np
from scipy.integrate import cumtrapz

from mwrt import MWRTM, LinearInterpolation
from faps_hatpro import faps, bgs


# Retrieval grid
z_hatpro = 612.
z_top = 12612.
# Retrieval grid
rgrid = np.round(np.logspace(np.log10(z_hatpro), np.log10(z_top), 50)).astype(float)
# Internal model grid
mgrid = np.logspace(np.log10(z_hatpro), np.log10(z_top), 2500)


class Gaussian:
    """Gaussian distribution with convenience methods/properties."""

    def __init__(self, mean, cov):
        self.mean = np.array(mean).reshape(-1,1)
        self.cov = np.array(cov)
        assert self.mean.shape[0] == self.cov.shape[0] == self.cov.shape[1]

    def sample(self, size):
        """Generate a random sample of numbers based on the distribution.
        
        Uses numpy.random.multivariate_normal.
        """
        return np.random.multivariate_normal(mean=self.mean.flatten(),
                cov=self.cov, size=size)

    @property
    def covi(self):
        """Memoized inverse of covariance."""
        if not hasattr(self, "_covi"):
            self._covi = np.linalg.inv(self.cov)
        return self._covi

    @classmethod
    def read_csv(cls, mean, cov):
        """Assemble a Gaussian object based on mean and cov in csv files."""
        from db_tools import read_csv_covariance, read_csv_mean
        cov = read_csv_covariance(cov)
        if mean is None:
            mean = np.zeros(cov.shape[0])
        else:
            mean = read_csv_mean(mean)
        return cls(mean, cov)

    def __len__(self):
        return self.mean.shape[0]


class OptimalEstimationRetrieval:
    """Iteration helper for optimal estimation retrievals.
    
    Automatically evaluates cost function values (.costs), observation vector
    distances (.obs_measures) and state vector distances (.state_measures) for
    determination of convergence.
    """

    def __init__(self, *, model, y, p0, μ0, prior, obs_error):
        """Set up an optimal estimation retrieval.

        z: retrival grid
        model: forward model (accepts state vector and surface pressure,
            returns simulated observation and Jacobian)
        params: a sequence of parameters to control the Levenberg-Marquard
            minimization. Last value is repeated if sequence is too short.
        y: observation vector
        p0: surface pressure in hPa
        μ0: first guess of state vector
        prior: prior distribution of atmospheric state
        obs_error: observation/model error distribution
        """
        self.model = model
        self.y = y
        self.p0 = p0
        self.μs = [μ0]
        self.Fμs = [0]
        self.covs = [prior.cov]
        self.prior = prior
        self.obserr = obs_error
        self.counter = 0
        self.obs_measures = []
        self.state_measures = []
        self.costs = []
        self.γs = []


    def iterate(self, γ, only=None):
        """Levenberg-Marquard step with 5.36 from Rodgers (2000).

        This method does not update γ, instead the current γ has to be
        specified during the method call. The used value of γ is however added
        to .γs for later reference.

        The 'only' parameter is just for test purposes. Use the specialized
        Virtual HATPROs instead.
        """
        μ = self.μs[-1]
        Fμ, jac = self.model(μ, self.p0)
        rhs = (jac.T @ self.obserr.covi @ (self.y - Fμ - self.obserr.mean)
                - self.prior.covi @ (μ - self.prior.mean))
        covi = self.prior.covi + jac.T @ self.obserr.covi @ jac
        lhs = covi + γ * self.prior.covi
        diff = np.linalg.solve(lhs, rhs)
        # Save new values
        if only is not None:
            diff_ = diff
            diff = np.zeros_like(diff)
            diff[only] = diff_[only]
        self.μs.append(μ + diff)
        self.Fμs.append(Fμ)
        self.covs.append(np.linalg.inv(covi))
        self.counter += 1
        self.γs.append(γ)
        # Calculate state space measure
        self.state_measures.append(float(diff.T @ covi @ diff))
        # Calculate observation space measure
        m = (self.obserr.cov
                @ np.linalg.inv(jac @ self.prior.cov @ jac.T + self.obserr.cov)
                @ self.obserr.cov)
        d = self.Fμs[-2] - self.Fμs[-1]
        self.obs_measures.append(float(d.T @ m @ d))
        # Cost function
        v1 = self.y - Fμ - self.obserr.mean
        v2 = μ - self.prior.mean
        cost = v1.T @ self.obserr.covi @ v1 + v2.T @ self.prior.covi @ v2
        self.costs.append(float(cost))


class VirtualHATPRO:
    """Optimal estimation wrapper preconfigured for HATPRO."""

    # Absorption model for each channel
    absorptions = faps

    # Cosmic background temperature for each channel
    backgrounds = bgs

    # HATPRO elevation scan angles
    angles = [0., 60., 70.8, 75.6, 78.6, 81.6, 83.4, 84.6, 85.2, 85.8]

    def __init__(self, z_retrieval, z_model, error,
            scanning=(10, 11, 12, 13)):
        """Set up missing optimal estimation parameters.

        z_retrieval     Retrieval height grid
        z_model         Internal model height grid
        error           Observation error distribution
        scanning        Angles used for elevation scanning. Default: only
                        zenith for K band and three most transparent channels
                        of V band but all angles for four most opaque channels
                        of V band.
        """
        self.z = z_retrieval
        itp = LinearInterpolation(source=z_retrieval, target=z_model)
        state_dims = 0
        # Create MWRTM instances for each channel and save corresponding angles
        self.mod_ang = []
        for i, (a, bg) in enumerate(zip(self.absorptions, self.backgrounds)):
            angles = self.angles if i in scanning else [0.]
            self.mod_ang.append([MWRTM(itp, a, background=bg), angles])
            state_dims += len(angles)
        self.error = error
        assert state_dims == len(self.error)

    def separate(self, x, p0):
        """Take apart the state vector and calculate pressure.

        Approximate pressure by barometric height formula. The specific gas
        constant is set to 288 to account for water vapor. The problem is that
        a good estimation of the actual R requires qvap but partition_lnq
        requires p to determine qsat, so there is a circular dependency. The
        assumption of 288 is made in the density calculation of the FAP as well
        and the error is small enough that an iterative procedure for
        determining R is unnecessary.
        """
        n = x.shape[0]//2
        T, lnq = x[:n,:].flatten(), x[n:,:].flatten()
        p = p0 * np.exp(-9.8076 * cumtrapz(1/(288*T), self.z, initial=0))
        return p, T, lnq

    def simulate(self, x, p0, only_forward=False):
        """Calculate brightness temperatures and Jacobian."""
        p, T, lnq = self.separate(x, p0)
        fwd, jac = [], []
        # Run model for each channel
        for model, angles in self.mod_ang:
            if only_forward:
                result = model.forward(angles=angles, p=p, T=T, lnq=lnq)
            else:
                result = model(angles=angles, p=p, T=T, lnq=lnq)
                jac.append(np.hstack([result.dT, result.dlnq]))
            fwd.append(result.fwd)
        # Combine all channels into a single result
        fwd = np.vstack(fwd)
        jac = np.vstack(jac)
        return fwd if only_forward else (fwd, jac)

    def retrieve(self, y, p0, μ0, prior, iterations=0, only=None):
        """Set up an OptimalEstimationRetrieval object based on this HATPRO.
        
        The iteration parameter currently does nothing and is kept only for
        compatibility.
        """
        optest = OptimalEstimationRetrieval(
                model=self.simulate,
                y=y, p0=p0, μ0=μ0,
                prior=prior, obs_error=self.error
                )
        #for i in range(iterations):
        #    optest.iterate(only=only)
        return optest


class VirtualHATPRO_zenith(VirtualHATPRO):
    """Uses only zenith observations for the retrieval."""

    absorptions = faps
    backgrounds = bgs
    angles = [0.]

    def __init__(self, z_retrieval, z_model, error, scanning=()):
        super().__init__(z_retrieval, z_model, error, scanning)


class VirtualHATPRO_Kband(VirtualHATPRO):
    """Uses only K band channels for the retrieval."""

    absorptions = faps[:7]
    backgrounds = bgs[:7]
    angles = [0.]

    def __init__(self, z_retrieval, z_model, error, scanning=()):
        super().__init__(z_retrieval, z_model, error, scanning)


class VirtualHATPRO_Vband(VirtualHATPRO):
    """Uses only V band channels for the retrieval."""

    absorptions = faps[7:]
    backgrounds = bgs[7:]

    def __init__(self, z_retrieval, z_model, error, scanning=None):
        super().__init__(z_retrieval, z_model, error,
                scanning=(3, 4, 5, 6))


def iterate_to_convergence(ret, γ0=3000, max_iterations=20, debug=False):
    """Iterate an OptimalEstimationRetrieval object until convergence
    based on a cost function criterion is achieved, adjusting the iteration
    parameter also based on the cost function."""
    # Initialize some helper variables, set initial costs high so that
    # convergence is never triggered prematurely
    min_cost, min_cost_at = 1.0e50, 0
    last_cost, current_cost = 1.0e50, 1.0e50
    cost_diff_counter = 0
    counter = 0
    γ = γ0

    if debug: print("Start")
    while counter < max_iterations:
        # Advance retrieval and obtain new value of cost function
        counter += 1
        if debug: print("Next iteration. Counter at {}.".format(counter))
        ret.iterate(γ=γ)
        current_cost = ret.costs[-1]

        # Convergence condition: relative cost function change
        # - If cost function decreases by less than 2 % or increases: increase
        #   a counter
        # - If cost function decreases by more than 2 %: reset the counter
        # Cost function is always positive, no abs necessary
        relative_cost_diff = (
                (last_cost - current_cost)
                / ((current_cost + last_cost) / 2)
                )
        if relative_cost_diff * 100 <= 2:
            cost_diff_counter += 1
        else:
            cost_diff_counter = 0

        if debug: print("    Current cost: {:10.3f}".format(current_cost))
        if debug: print("    Relative difference of {:5.2f} %".format(relative_cost_diff*100))

        # New cost minimum found?
        if current_cost < min_cost:
            min_cost = current_cost
            min_cost_at = counter

        # If convergence condition counter is at 3: stop iteration
        if cost_diff_counter > 2:
            # Convergence, use state at min_cost
            if debug: print("Converged, cost minimum at {}".format(min_cost_at))
            # min_cost_at indexes ret.μs which has one element more than
            # ret.costs i.e. the minimum of ret.costs is at min_cost_at - 1
            return True, min_cost_at
        
        # Update gamma (Schneebeli 2009)
        if current_cost < last_cost:
            γ = γ * 0.5
        elif current_cost >= last_cost:
            γ = γ * 5
        else:
            pass

        last_cost = current_cost

    if debug: print("No convergence after {} iterations".format(counter))
    return False, min_cost_at

