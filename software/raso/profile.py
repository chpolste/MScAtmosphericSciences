import inspect, numbers
from collections import OrderedDict

import numpy as np

import raso.tools as tools
from raso.accessor import get_var


__all__ = ["Profile"]

__doc__ = """"""


class Profile:
    """Data container with conversion capabilities and other conveniences.

    Profile is not unit-aware, these are the variable names and units that make
    the converters work properly:

    Variable    Unit        Description
    ---------------------------------------------------------
    time        s           Flight time
    p           Pa          Pressure
    z           m           Altitude
    geopot      gpdm        Geopotential height
    ff          m/s         Wind speed
    dd          °           Wind direction
    T           K           Temperature
    Td          K           Dew point temperature
    Tpot        K           Potential temperature
    e           Pa          Water vapor partial pressure
    es          Pa          Water vapor saturation pressure
    RH          %           Relative humidity
    r           kg/kg       Water vapor mixing ratio
    Lvap        J/kg/K      Latent heat of evaporation
    LRmoist     K/km        Moist adiabatic lapse rate
    density     kg/m³       Density
    """

    def __init__(self, *, metadata=None, **data):
        """Create a new Profile based on the data given via kwargs."""
        self.metadata = {} if metadata is None else metadata
        self.data = (np.rec.fromarrays(data.values(), names=list(data.keys()))
                if data else None)

    def __repr__(self):
        if self.data is not None:
            return "\n".join(["radiosonde.Profile", repr(self.metadata),
                    ", ".join(self.data.dtype.names)])
        return "\n".join(["radiosonde.Profile", repr(self.metadata), "no data"])

    def __getitem__(self, variables):
        """Obtain a variable"""
        if isinstance(variables, list):
            return np.rec.fromarrays((get_var(self, var) for var in variables),
                    names=variables)
        return get_var(self, variables)

    @property
    def variables(self):
        return set(self.data.dtype.names) if self.data is not None else set()

    @classmethod
    def from_dataframe(cls, df, metadata=None):
        """"""
        out = cls(metadata=metadata, **dict(df))
        return out

    def plot(self, *, y="p", ylim=None, size=5, kind="thermodynamics",
            compare=None, fig=None):
        """Plot the profile.
        
        The vertical coordinate can be selected with y, ylim sets the range
        of the vertical axis. kind selects the types of additional plots,
        available are: wind (speed and direction), thermodynamics (relative
        humidity and potential temperature). A second Profile instance can be
        given to compare, then both are plotted.
        """
        from matplotlib.pyplot import figure
        from matplotlib.gridspec import GridSpec

        assert y in ["p", "z", "geopot"]
        assert ylim is None or len(ylim) == 2

        if fig is None:
            fig = figure(figsize=(2*size, size))
        gs = GridSpec(ncols=3, nrows=1, width_ratios=[70, 15, 15])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])

        y_values = self[y]
        # Set up y-axis
        if ylim is None:
            ylim = (min(y_values), max(y_values))
        assert ylim[0] < ylim[1]
        y_ticks = tools.get_yticks(ylim)
        selected = (y_values >= ylim[0]) & (y_values <= ylim[1])
        if y == "p":
            # Switch y-limit for pressure coordinates
            ylim = (ylim[1], ylim[0])
        # Apply to all y-axes
        for ax in [ax1, ax2, ax3]:
            if y == "p": ax.set_yscale("log")
            ax.set_ylim(*ylim)
            ax.set_yticks(y_ticks)
            ax.grid(linestyle="-", color="#CCCCCC", zorder=20)
        label_scaling = 1/100 if y == "p" else 1
        ax1.set_yticklabels([int(y*label_scaling) for y in y_ticks])

        labelstrftime = "%Y-%m-%d %H:%M:%S"
        # Temperature and dew point
        try: label1 = self.metadata["datetime"].strftime(labelstrftime)
        except KeyError: label1 = "Profile 1"
        temp, dewtemp = self["T"], self["Td"]
        ax1.plot(temp, y_values, linewidth=2, color="#000000", zorder=55,
                label=label1)
        ax1.plot(dewtemp, y_values, linewidth=2, color="#666666", zorder=50)
        Tlow = min(np.nanmin(temp[selected]), np.nanmin(dewtemp[selected]))
        Thigh = max(np.nanmax(temp[selected]), np.nanmax(dewtemp[selected]))
        ax1.set_xlim(Tlow-5, Thigh+5)
        ax1.set_title("T, Td [K]", loc="right")
        # Second profile if specified
        if compare is not None:
            try: label2 = compare.metadata["datetime"].strftime(labelstrftime)
            except KeyError: label2 = "Profile 2"
            ax1.plot(compare["T"], compare[y], linewidth=2, color="#336699",
                    zorder=45, label=label2)
            ax1.plot(compare["Td"], compare[y], linewidth=2, color="#AFC9E4",
                    zorder=40)
        ax1.legend(loc="upper right", fontsize=10)

        if kind == "wind":
            # Speed
            ff, dd = self["ff"], self["dd"]
            ax2.plot(ff, y_values, linewidth=2, color="#000000", zorder=55)
            ax2.set_xticks(range(0, 100, 10))
            ax2.set_xticklabels(map(str, range(0, 100, 10)))
            ax2.set_xlim(0, np.nanmax(ff[selected])+2)
            ax2.set_title("ff [m/s]", loc="right")
            # Direction
            ax3.plot(dd, y_values, linewidth=2, color="#000000", zorder=55)
            dd_levels = [0, 90, 180, 270, 360]
            dd_names = ["N", "E", "S", "W", "N"]
            ax3.set_xlim(min(dd_levels), max(dd_levels))
            ax3.set_xticks(dd_levels)
            ax3.set_xticklabels(dd_names)
            ax3.set_title("dd [°]", loc="right")
            if compare is not None:
                ax2.plot(compare["ff"], compare[y], linewidth=2,
                        color="#336699", zorder=45)
                ax3.plot(compare["dd"], compare[y], linewidth=2,
                        color="#336699", zorder=45)
        elif kind == "thermodynamics":
            # Relative Humidity
            rh = self["RH"]
            ax2.plot(rh, y_values, linewidth=2, color="#000000", zorder=55)
            ax2.set_xlim(0, 100)
            ax2.set_title("RH [%]", loc="right")
            ax2.fill_betweenx(y_values, 0, 100, color="#DDDDDD", zorder=-15,
                    where=(rh>=90)&(temp>253.15))
            ax2.fill_betweenx(y_values, 0, 100, color="#BBBBBB", zorder=-10,
                    where=(rh>=95)&(temp>253.15))
            # Stability
            tpot = self["Tpot"]
            ax3.set_xlim(np.nanmin(tpot[selected])-5,
                    np.nanmax(tpot[selected])+5)
            ax3.plot(tpot, y_values, linewidth=2, color="#000000", zorder=50)
            ax3.set_title("Tpot [K]", loc="right")
            if compare is not None:
                ax2.plot(compare["RH"], compare[y], linewidth=2,
                        color="#336699", zorder=45)
                ax3.plot(compare["Tpot"], compare[y], linewidth=2,
                        color="#336699", zorder=45)

        ax2.label_outer()
        ax3.label_outer()
        fig.tight_layout()

