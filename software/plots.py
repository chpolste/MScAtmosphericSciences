import numpy as np
from matplotlib.pyplot import figure
from matplotlib.gridspec import GridSpec

import formulas as fml


def raso(df, *, y="p", ylim=None, title="", size=5, fig=None):
    """Plot T, Td in (nonskew) T-p or T-geopot diagram.

    Additionally, relative humidity and potential temperature are plotted.
    From the given dataframe the p, T, Td and - if selected - geopot columns
    are used. RH is calculated from T and Td, Tpot is calculated from T and p.
    """
    assert y == "p" or y == "geopot"
    assert ylim is None or len(ylim) == 2

    if fig is None:
        fig = figure(figsize=(2*size, size))
    gs = GridSpec(ncols=3, nrows=1, width_ratios=[70, 15, 15])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    y_values = df[y]
    # Set up y-axis
    if ylim is None: ylim = (min(y_values), max(y_values))
    assert ylim[0] < ylim[1]
    y_ticks = get_yticks(ylim)
    selected = (y_values >= ylim[0]) & (y_values <= ylim[1])
    # Swap y limits for pressure coordinates
    if y == "p": ylim = (ylim[1], ylim[0])
    # Apply to all y-axes
    for ax in [ax1, ax2, ax3]:
        if y == "p": ax.set_yscale("log")
        ax.set_ylim(*ylim)
        ax.set_yticks(y_ticks)
        ax.grid(linestyle="-", color="#CCCCCC", zorder=20)
    ax1.set_yticklabels([int(y) for y in y_ticks])

    temp, dewtemp = df["T"], df["Td"]
    ax1.plot(temp, y_values, linewidth=2, color="#000000", zorder=55)
    ax1.plot(dewtemp, y_values, linewidth=2, color="#336699", zorder=50)
    Tlow = min(temp.ix[selected].min(), dewtemp.ix[selected].min())
    Thigh = max(temp.ix[selected].max(), dewtemp.ix[selected].max())
    ax1.set_xlim(Tlow-5, Thigh+5)
    ax1.set_title("T, Td [K]", loc="right")

    # Relative Humidity
    rh = fml.RH(temp, dewtemp)
    ax2.plot(rh, y_values, linewidth=2, color="#000000", zorder=55)
    ax2.set_xlim(0, 100)
    ax2.set_title("RH [%]", loc="right")
    ax2.fill_betweenx(y_values, 0, 100, color="#BBBBBB", zorder=-15,
            where=(rh>=90)&(temp>253.15))
    ax2.fill_betweenx(y_values, 0, 100, color="#336699", zorder=-10,
            where=(rh>=95)&(temp>253.15))
    # Stability
    tpot = fml.Tpot(temp, df["p"])
    ax3.set_xlim(tpot.ix[selected].min()-5, tpot.ix[selected].max()+5)
    ax3.plot(tpot, y_values, linewidth=2, color="#000000", zorder=50)
    ax3.set_title("Tpot [K]", loc="right")

    ax2.label_outer()
    ax3.label_outer()
    fig.tight_layout()


def get_yticks(ylim):
    """Determine decent spacing of vertical ticks."""
    difflg = np.log10(abs(ylim[0] - ylim[1]))
    cond = int(difflg*4) % 4
    spacing = cond + 1 if cond < 2 else 5
    spacing = spacing * 10**(int(difflg)-1)
    y, ticks = 0, []
    while y <= ylim[1]:
        if y >= ylim[0]:
            ticks.append(y)
        y += spacing
    return ticks

