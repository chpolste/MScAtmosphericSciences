"""Components for automatic, forward-mode differentiation."""

from functools import singledispatch, wraps
from numbers import Number

import numpy as np
import scipy.sparse as sp
import scipy.integrate as it


class VectorBase:
    """"""

    def __init__(self, fwd, dT, dlnq):
        """Create a new instance with autodiff capabilities.
        
        Components:
        fwd     the actual value of the calculation
        dT      the derivative wrt temperature
        dlnq    the derivative wrt the logarithm of specific water content
        """
        self.fwd = fwd
        self.dT = dT
        self.dlnq = dlnq


class Vector(VectorBase):
    """A vector of independent scalars with automatic differentiation.

    Overloaded operators: +, *, partially / (to be understood element-wise)
    Operators work between Vector instances and with scalars.
    
    jacobians: rows are derivatives of fixed component, cols are derivatives
               wrt the same variable of all components
    """


    def __repr__(self):
        return "Vector(fwd={}, dT={}, dlnq={})".format(
                repr(self.fwd), repr(self.dT), repr(self.dlnq))

    def __mul__(self, other):
        """Element-wise multiplication, product rule."""
        if isinstance(other, Vector):
            return Vector(
                    fwd = self.fwd * other.fwd,
                    # [:,None] causes row-wise multiplication
                    dT = self.dT * other.fwd[:,None] + self.fwd[:,None] * other.dT,
                    dlnq = self.dlnq * other.fwd[:,None] + self.fwd[:,None] * other.dlnq
                    )
        if isinstance(other, Number):
            return Vector(
                    fwd = self.fwd * other,
                    dT = self.dT * other,
                    dlnq = self.dlnq * other
                    )
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Number):
            return self * other # Multiplication is associative
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Number):
            return Vector(
                    fwd = self.fwd / other,
                    dT = self.dT / other,
                    dlnq = self.dlnq / other
                    )
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(
                    fwd = self.fwd + other.fwd,
                    dT = self.dT + other.dT,
                    dlnq = self.dlnq + other.dlnq
                    )
        if isinstance(other, Number):
            return Vector(fwd=self.fwd+other, dT=self.dT, dlnq=self.dlnq)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, Number):
            return self + other # Addition is associative
        return NotImplemented

    def __neg__(self):
        return Vector(fwd=-self.fwd, dT=-self.dT, dlnq=-self.dlnq)

    def __getitem__(self, key):
        return Vector(
                fwd = self.fwd[key],
                dT = self.dT[key,:],
                dlnq = self.dlnq[key,:]
                )



class DiagVector(VectorBase):
    """A vector of independent scalars with automatic differentiation.

    Overloaded operators: +, -, *, /, ** (to be understood element-wise)
    Operators work between DiagVector instances and with scalars.

    The container is taylored to the needs of the fast absorption predictors.
    It is suitable for a vector of scalars, whose derivatives (dT, dlnq) can be
    calculated element-wise and are of the same number as the vector. This is
    the case for the absorption coefficients, which are only dependent on the
    properties of their own layer. The Jacobians of the vector are diagonal
    matrices, therefore it is sufficient to calculate and store only the
    diagonals.
    """

    def __repr__(self):
        return "DiagVector(fwd={}, dT={}, dlnq={})".format(
                repr(self.fwd), repr(self.dT), repr(self.dlnq))

    def as_vector(self):
        return Vector(
                fwd = self.fwd,
                dT = sp.diags(self.dT),
                dlnq = sp.diags(self.dlnq)
                )

    def __pow__(self, other):
        """Power rule of derivation, implemented for integer exponents."""
        if isinstance(other, int):
            return DiagVector(
                    fwd = self.fwd**other,
                    dT = self.fwd**(other-1) * self.dT * other,
                    dlnq = self.fwd**(other-1) * self.dlnq * other,
                    )

    def __mul__(self, other):
        """Product rule of derivation."""
        if isinstance(other, DiagVector):
            return DiagVector( 
                    fwd = self.fwd * other.fwd,
                    dT = self.dT*other.fwd + self.fwd*other.dT,
                    dlnq = self.dlnq*other.fwd + self.fwd*other.dlnq
                    )
        if isinstance(other, Number):
            return DiagVector(
                    fwd = self.fwd * other,
                    dT = self.dT * other,
                    dlnq = self.dlnq * other
                    )
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Number):
            return self * other # Multiplication is associative
        return NotImplemented

    def __truediv__(self, other):
        """Quotient rule of derivation."""
        if isinstance(other, DiagVector):
            return DiagVector( 
                    fwd = self.fwd / other.fwd,
                    dT = (self.dT*other.fwd-self.fwd*other.dT) / other.fwd**2,
                    dlnq = (self.dlnq*other.fwd-self.fwd*other.dlnq) / other.fwd**2
                    )
        if isinstance(other, Number):
            return DiagVector(
                    fwd = self.fwd / other,
                    dT = self.dT / other,
                    dlnq = self.dlnq / other
                    )
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, Number):
            return DiagVector(
                    fwd = other / self.fwd,
                    dT = - other * self.dT / self.fwd**2,
                    dlnq = - other * self.dlnq / self.fwd**2
                    )
        return NotImplemented

    def __add__(self, other):
        """Derivation is linear."""
        if isinstance(other, DiagVector):
            return DiagVector( 
                    fwd = self.fwd + other.fwd,
                    dT = self.dT + other.dT,
                    dlnq = self.dlnq + other.dlnq
                    )
        if isinstance(other, Number):
            return DiagVector(fwd=self.fwd+other, dT=self.dT, dlnq=self.dlnq)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, Number):
            return self + other # Addition is associative
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, DiagVector):
            return DiagVector( 
                    fwd = self.fwd - other.fwd,
                    dT = self.dT - other.dT,
                    dlnq = self.dlnq - other.dlnq
                    )
        if isinstance(other, Number):
            return DiagVector(fwd=self.fwd-other, dT=self.dT, dlnq=self.dlnq)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, Number):
            return DiagVector( 
                    fwd = other - self.fwd,
                    dT = - self.dT,
                    dlnq = - self.dlnq
                    )
        return NotImplemented

    @classmethod
    def pack_args(cls, f):
        """Wrap a function such that all args of calls to the wrapper are
        packaged into a ValueVector with zero-valued derivatives. This might be
        useful for testing, when only the forward value is relevant and the
        function to test only accepts DiagVector arguments.
        """
        @wraps(f)
        def packed(*args, **kwargs):
            args = [arg if isinstance(arg, cls) else cls(arg, 0, 0)
                    for arg in args]
            kwargs = {key: arg if isinstance(arg, cls) else cls(arg, 0, 0)
                    for key, arg in kwargs.items()}
            return f(*args, **kwargs)
        return packed

    @classmethod
    def init_T(cls, T):
        """Initialize a vector of temperature values (dT = 1, dlnq = 0)."""
        return cls(T, np.ones_like(T), np.zeros_like(T))

    @classmethod
    def init_lnq(cls, lnq):
        """Initialize a vector of ln(q) values (dT = 0, dlnq = 1)."""
        return cls(lnq, np.zeros_like(lnq), np.ones_like(lnq))

    @classmethod
    def init_p(cls, p):
        """Initialize a vector of p values (dT = 0, dlnq = 0)."""
        return cls(p, np.zeros_like(p), np.zeros_like(p))


# Additional functions that support automatic differentiation:

@singledispatch
def exp(value):
    """Exponential function."""
    return np.exp(value)

@exp.register(Vector)
def _(value):
    fwd = np.exp(value.fwd)
    return Vector(
            fwd = fwd,
            dT = value.dT * fwd[:,None],
            dlnq = value.dlnq * fwd[:,None]
            )

@exp.register(DiagVector)
def _(value):
    fwd = np.exp(value.fwd)
    return DiagVector(fwd=fwd, dT=value.dT*fwd, dlnq=value.dlnq*fwd)


@singledispatch
def trapz(value, grid):
    """Trapezoidal quadrature."""
    return it.trapz(value, grid)

@trapz.register(Vector)
def _(value, grid):
    return Vector(
            fwd = it.trapz(value.fwd, grid),
            dT = it.trapz(value.dT, grid, axis=0),
            dlnq = it.trapz(value.dlnq, grid, axis=0),
            )


@singledispatch
def cumtrapz(value, grid, initial=None):
    """Cumulative trapezoidal quadrature."""
    return it.cumtrapz(value, grid, initial=initial)

@cumtrapz.register(Vector)
def _(value, grid, initial=None):
    return Vector(
            fwd = it.cumtrapz(value.fwd, grid, initial=initial),
            dT = it.cumtrapz(value.dT, grid, initial=initial, axis=0),
            dlnq = it.cumtrapz(value.dlnq, grid, initial=initial, axis=0),
            )

