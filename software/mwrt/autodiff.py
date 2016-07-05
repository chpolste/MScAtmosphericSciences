"""Components for automatic, forward-mode differentiation."""

from functools import singledispatch, wraps
from numbers import Number

import numpy as np


class VectorValue:
    """A vector of independent scalars with automatic differentiation.

    Overloaded operators: +, -, *, /, **
    Operators work between VectorValue instances and with scalars.

    The container is taylored to the needs of the fast absorption predictors.
    It is suitable for a vector of scalars, whose derivatives (dT, dlnq) can be
    calculated element-wise. This is the case for the absorption coefficients,
    which are only dependent on the properties of their own layer. The Jacobian
    of the vector is a diagonal matrix, therefore it is sufficient to calculate
    and store only the diagonal.
    """

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

    def __repr__(self):
        return "VectorValue(fwd={}, dT={}, dlnq={})".format(
                repr(self.fwd), repr(self.dT), repr(self.dlnq))

    def __pow__(self, other):
        """Power rule of derivation, implemented for integer exponents."""
        if isinstance(other, int):
            return VectorValue(
                    fwd = self.fwd**other,
                    dT = self.fwd**(other-1) * self.dT * other,
                    dlnq = self.fwd**(other-1) * self.dlnq * other,
                    )

    def __mul__(self, other):
        """Product rule of derivation."""
        if isinstance(other, VectorValue):
            return VectorValue( 
                    fwd = self.fwd * other.fwd,
                    dT = self.dT*other.fwd + self.fwd*other.dT,
                    dlnq = self.dlnq*other.fwd + self.fwd*other.dlnq
                    )
        if isinstance(other, Number):
            return VectorValue(
                    fwd = self.fwd * other,
                    dT = self.dT * other,
                    dlnq = self.dlnq * other
                    )
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Number):
            return VectorValue(
                    fwd = other * self.fwd,
                    dT = other * self.dT,
                    dlnq = other * self.dlnq
                    )
        return NotImplemented

    def __truediv__(self, other):
        """Quotient rule of derivation."""
        if isinstance(other, VectorValue):
            return VectorValue( 
                    fwd = self.fwd / other.fwd,
                    dT = (self.dT*other.fwd-self.fwd*other.dT) / other.fwd**2,
                    dlnq = (self.dlnq*other.fwd-self.fwd*other.dlnq) / other.fwd**2
                    )
        if isinstance(other, Number):
            return VectorValue(
                    fwd = self.fwd / other,
                    dT = self.dT / other,
                    dlnq = self.dlnq / other
                    )
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, Number):
            return VectorValue(
                    fwd = other / self.fwd,
                    dT = - other * self.dT / self.fwd**2,
                    dlnq = - other * self.dlnq / self.fwd**2
                    )
        return NotImplemented

    def __add__(self, other):
        """Derivation is linear."""
        if isinstance(other, VectorValue):
            return VectorValue( 
                    fwd = self.fwd + other.fwd,
                    dT = self.dT + other.dT,
                    dlnq = self.dlnq + other.dlnq
                    )
        if isinstance(other, Number):
            return VectorValue( 
                    fwd = self.fwd + other,
                    dT = self.dT,
                    dlnq = self.dlnq
                    )
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, Number):
            return VectorValue( 
                    fwd = other + self.fwd,
                    dT = self.dT,
                    dlnq = self.dlnq
                    )
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, VectorValue):
            return VectorValue( 
                    fwd = self.fwd - other.fwd,
                    dT = self.dT - other.dT,
                    dlnq = self.dlnq - other.dlnq
                    )
        if isinstance(other, Number):
            return VectorValue( 
                    fwd = self.fwd - other,
                    dT = self.dT,
                    dlnq = self.dlnq
                    )
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, Number):
            return VectorValue( 
                    fwd = other - self.fwd,
                    dT = - self.dT,
                    dlnq = - self.dlnq
                    )
        return NotImjlemented

    @classmethod
    def pack_args(cls, f):
        """Wrap a function such that all args of calls to the wrapper are
        packaged into a ValueVector with zero-valued derivatives. This might be
        useful for testing, when only the forward value is relevant and the
        function to test only accepts VectorValue arguments.
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


# Additional functions that support automatic differentiation:

@singledispatch
def exp(value):
    """Exponential function."""
    err = "No implementation for type {}".format(type(value))
    raise NotImplementedError(err)

@exp.register(np.ndarray)
@exp.register(Number)
def _(value):
    return np.exp(value)

@exp.register(VectorValue)
def _(value):
    fwd = np.exp(value.fwd)
    return VectorValue(fwd=fwd, dT=value.dT*fwd, dlnq=value.dlnq*fwd)

