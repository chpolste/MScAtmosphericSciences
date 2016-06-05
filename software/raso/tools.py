import numpy as np


def get_yticks(ylim):
    """Determine optimal spacing of vertical ticks."""
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

