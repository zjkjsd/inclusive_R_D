# +
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tool to apply offline cuts (and DecayHash) to Ntuples and save/split them to one/multiple parquet/root files.

Usage: python3 1_Apply_DecayHash.py -d folder -i -o -l -n (--nohash)

Example: python3 1_Apply_DecayHash.py -d Samples/Signal_MC14ri/MC14ri_sigDDst_bengal_e_2 \
         -i sigDDst_bengal_e_2.root -o parquet -l e --mctype signal (--nohash)
"""

import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from cycler import cycler

from typing import List, Tuple


class BrightColorScheme:
    """
    Colour scheme from https://personal.sron.nl/~pault/
    """
    blue = '#4477AA'
    cyan = '#66CCEE'
    green = '#228833'
    yellow = '#CCBB44'
    pink = '#EE6677'
    purple = '#AA3377'
    grey = '#BBBBBB'
    gray = '#BBBBBB'
    black = '#000000'

    ## added colours to expand the scheme
    # these are probably no longer colour blind friendly
    blue_medium = '#55A2CC'
    teal = '#44AA91'
    green_light = '#77A23C'
    orange = '#DD915E'

    default_colors = [blue, purple, green, yellow, cyan, pink, grey, orange, green_light, teal, blue_medium]


bright_color_cycler = cycler('color', BrightColorScheme.default_colors)

binary_cmap = matplotlib.colors.ListedColormap([BrightColorScheme.grey, BrightColorScheme.blue_medium])


class FixedLabelWidthScalarFormatter(matplotlib.ticker.ScalarFormatter):
    """
    Override the __call__ method to ensure a fixed width of the label formats
    """

    #https://github.com/matplotlib/matplotlib/blob/c23ccdde6f0f8c071b09a88770e24452f2859e99/lib/matplotlib/ticker.py
    def __call__(self, x, pos=None):
        """
        Return the format for tick value *x* at position *pos*.
        """
        if len(self.locs) == 0:
            return ''

        xp = (x - self.offset) / (10.**self.orderOfMagnitude)
        if abs(xp) < 1e-8:
            xp = 0

        if abs(xp) >= 1 or abs(xp) == 0.0:
            fmt = '%#.3g'
        else:
            fmt = '%#.2g'

        if self._usetex or self._useMathText:
            fmt = r'$\mathdefault{%s}$' % fmt

        s = fmt % xp
        return self.fix_minus(s)


class dummyAxisForFormat():
    """
    The things I do for nicely formatted axis.
    Used if we just want to use the formatter without having an axis.
    """

    def __init__(self, locs):
        self.locs = locs
        self.min = np.min(locs)
        self.max = np.max(locs)

    def get_view_interval(self):
        return self.min, self.max


def format_with_mpl_scalerFormatter(locs, useOffset=None) -> Tuple[List[str], str]:
    """
       Uses the default matplotlib.ScalarFormatter to format numbers WITHOUT requiring an active axis object.

    Args:
        locs (_type_): list of numbers to be formatted. Can repeat.
        useOffset (_type_, optional): Whether to use an offset. Defaults to using what's in the rcParams if None.

    Returns:
        _type_: List of formatted strings for each point, offset string
    """
    if useOffset is None:
        useOffset = mpl.rcParams['axes.formatter.useoffset']
    sf = matplotlib.ticker.ScalarFormatter(useMathText=mpl.rcParams['axes.formatter.use_mathtext'], useOffset=useOffset)
    sf.axis = dummyAxisForFormat(locs)
    sf.set_locs(locs)
    labels = [sf(x) for x in locs]
    # unfortunately if we have an offset its always set to %g1.10
    # see https://github.com/matplotlib/matplotlib/blob/c23ccdde6f0f8c071b09a88770e24452f2859e99/lib/matplotlib/ticker.py#L673
    # if we dont want that we need to overwrite this method
    # but that requires finding a fancy way to calculate how many zeroes to keep.
    return labels, sf.get_offset()


def set_matplotlibrc_params(
    errorbar_caps=False,
    top_right_ticks=True,
):
    """
    Sets default parameters in the matplotlibrc.
    Copied from WG1 Template with some tweaks.
    :return: None
    """
    xtick = {'top': top_right_ticks, 'minor.visible': True, 'direction': 'in', 'labelsize': 10}
    ytick = {'right': top_right_ticks, 'minor.visible': True, 'direction': 'in', 'labelsize': 10}

    axes = {
        'labelsize': 12,
        'prop_cycle': bright_color_cycler,
        'formatter.limits': (-2, 2),
        'formatter.use_mathtext': True,
        'titlesize': 'large',
        'labelpad': 4.0,
    }
    lines = {'lw': 1.5}
    legend = {'fontsize': 10, 'frameon': False}
    errorbar = {'capsize': 2 if errorbar_caps else 0}

    plt.rc('lines', **lines)
    plt.rc('axes', **axes)
    plt.rc('xtick', **xtick)
    plt.rc('ytick', **ytick)
    plt.rc('legend', **legend)
    plt.rc('errorbar', **errorbar)
    return


def color_with_alpha_to_color(color, alpha, bg_rgb=(1, 1, 1)):
    """
    Blends a colour with white (bg_rgb = (1,1,1)) using some alpha.
    """
    rgb_tuple = matplotlib.colors.to_rgb(color)
    return tuple([x * alpha + (1 - alpha) * y for x, y in zip(rgb_tuple, bg_rgb)])

