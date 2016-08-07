import numpy as np
from matplotlib.pyplot import figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from optimal_estimation import rgrid, z_hatpro, z_top
import formulas as fml


def raso(df, *, y="p", ylim=None, title="", size=5, fig=None):
    """Plot T, Td in (nonskew) T-p or T-z diagram.

    Additionally, relative humidity and potential temperature are plotted.
    From the given dataframe the p, T, Td and - if selected - z columns
    are used. RH is calculated from T and Td, Tpot is calculated from T and p.
    """
    assert y == "p" or y == "z"
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


def retrieval_template(figsize, *, ratio=1.7, hum="q", Tlims=None, qlims=None):
    if hum == "q":
        hum = "specific water content [g/kg]"
    elif hum == "lnq":
        hum = "ln(specific water content) [ln(kg/kg)]"
    else:
        raise ValueError()
    gs = GridSpec(2, 2, height_ratios=[ratio, 1]) 
    fig = figure(figsize=figsize)
    axT1 = fig.add_subplot(gs[0,0])
    axT2 = fig.add_subplot(gs[1,0])
    axT1.set_xlabel("temperature [K]")
    axT2.set_xlabel("temperature [K]")
    if Tlims is not None:
        axT1.set_xlim(*Tlims[0])
        axT1.set_ylim(*Tlims[1])
        axT2.set_xlim(*Tlims[2])
        axT2.set_ylim(*Tlims[3])
        rectT = Rectangle([Tlims[2][0], Tlims[3][0]],
                         Tlims[2][1]-Tlims[2][0], Tlims[3][1]-Tlims[3][0],
                         facecolor="#F0F0F0", linewidth=0, zorder=-100)
        axT1.add_patch(rectT)
    axq1 = fig.add_subplot(gs[0,1])
    axq2 = fig.add_subplot(gs[1,1])
    axq1.set_xlabel(hum)
    axq2.set_xlabel(hum)
    if qlims is not None:
        axq1.set_xlim(*qlims[0])
        axq1.set_ylim(*qlims[1])
        axq2.set_xlim(*qlims[2])
        axq2.set_ylim(*qlims[3])
        rectq = Rectangle([qlims[2][0], qlims[3][0]],
                         qlims[2][1]-qlims[2][0], qlims[3][1]-qlims[3][0],
                         facecolor="#F0F0F0", linewidth=0, zorder=-100)
        axq1.add_patch(rectq)
    return fig, (axT1, axT2, axq1, axq2)


def statistical_eval(ax, reference, *data, labels=None, colors=None, bias=True):
    if labels is None: labels = ["Data {}".format(i) for i in range(len(data))]
    if colors is None: colors = ["#000000", "#1f78b4", "#33a02c", "#666666", "r"]
    for df, name, clr in zip(data, labels, colors):
        diff = reference - df
        ax.plot(diff.std().values, (rgrid-z_hatpro)/1000, color=clr, label=name, linewidth=2)
        if bias: ax.plot(diff.mean().values, (rgrid-z_hatpro)/1000, "--", color=clr, linewidth=1.5)
    ax.set_ylabel("height above ground [km]")
    ax.vlines(0, z_top/1000, 0, color="#BBBBBB", zorder=-80)
