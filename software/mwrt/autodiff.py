from functools import singledispatch
from numbers import Number

import numpy as np


class VectorValue:
    """Container for 1-dimensional arrays with automatic differentiation.
    
    Warning: this is a very limited implementation suitable only for a
        vector of scalars whose derivative is calculated element-wise.
        This is good enough for the absorption coefficient calculation as
        these are only dependent on the properties of their own layer (i.e.
        the jacobian of the full vector is a diagonal matrix, and the
        differentiation only calculates the diagonal).
    """

    def __init__(self, fwd, dT, dlnq):
        self.fwd = fwd
        self.dT = dT
        self.dlnq = dlnq

    def __repr__(self):
        return "VectorValue(fwd={}, dT={}, dlnq={})".format(
                repr(self.fwd), repr(self.dT), repr(self.dlnq))

    def __pow__(self, other):
        if isinstance(other, int):
            return VectorValue(
                    fwd = self.fwd**other,
                    dT = self.fwd**(other-1) * self.dT * other,
                    dlnq = self.fwd**(other-1) * self.dlnq * other,
                    )

    def __mul__(self, other):
        """Product rule."""
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
        """Quotient rule."""
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

    @staticmethod
    def pack_args(f):
        """For testing purposes when only the forward value is relevant."""
        def packed(*args, **kwargs):
            return f(*(VectorValue(a, 0, 0) for a in args),
                    **{k: VectorValue(v, 0, 0) for k, v in kwargs.items()})
        return packed


@singledispatch
def exp(value):
    raise NotImplementedError("No implementation for type {}".format(type(value)))

@exp.register(np.ndarray)
@exp.register(Number)
def _(value):
    return np.exp(value)

@exp.register(VectorValue)
def _(value):
    fwd = np.exp(value.fwd)
    return VectorValue(
            fwd = fwd,
            dT = value.dT * fwd,
            dlnq = value.dlnq * fwd
            )


