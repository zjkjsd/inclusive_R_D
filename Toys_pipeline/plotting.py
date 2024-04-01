# +
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-

def round_pdg_style(val: float, err: float, extra_sig_figs: int = 0, return_round_index=False):
    """
    Rounds a value and error following the rules of the PDG: (https://pdg.lbl.gov/2019/reviews/rpp2019-rev-rpp-intro.pdf (s5.3))
    If the three highest order digits are between:
        100 and 354: two sig digits
        355 and 949: one sig digit
        949 and 999: round to 1000.

    :param val: value to be rounded.
    :param err: error of value to be rounded.
    :param extra_sig_figs: optional parameter to increase the number of significant figures rounded to.
    """
    round_index = -int(np.floor(np.log10(abs(err))))
    three_highest_digits = round(err, round_index + 2) * 10**(round_index + 2)

    round_bonus = 0
    if (three_highest_digits < 355):
        round_bonus = 1
        # case is not technically needed but kept for clarity
    elif (three_highest_digits < 950):
        round_bonus = 0

    round_index = round_bonus + round_index + extra_sig_figs
    rounded_val = round(val, round_index)
    rounded_err = round(err, round_index)
    if return_round_index:
        return rounded_val, rounded_err, round_index
    return rounded_val, rounded_err


def round_pdg_style_str(val: float, err: float, extra_sig_figs: int = 0):
    """
    Rounds a value and error following the rules of the PDG: (https://pdg.lbl.gov/2019/reviews/rpp2019-rev-rpp-intro.pdf (s5.3))
    If the three highest order digits are between:
        100 and 354: two sig digits
        355 and 949: one sig digit
        949 and 999: round to 1000.

    :param val: value to be rounded.
    :param err: error of value to be rounded.
    :param extra_sig_figs: optional parameter to increase the number of significant figures rounded to.
    """
    rounded_val, rounded_err, round_index = round_pdg_style(val, err, extra_sig_figs, True)
    return f'{rounded_val:.{round_index}f}', f'{rounded_err:.{round_index}f}'

# import xulnu.labels
# import xulnu.lumi
# import xulnu.plotting_collections
# import xulnu.systematics
# import xulnu.utility
# import xulnu.weights

# from dataclasses import dataclass
# from functools import reduce
from plotting_style import *

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import uncertainties
import uncertainties.unumpy as unp
from typing import List, Optional, Tuple, Union


def gaussian(x, mu, sig):
    return 1. / (((2. * np.pi)**0.5) * sig) * np.e**(-(((x - mu) / sig)**2) / 2)


def plot_toy_gaussian(
        x: list,
        mu: uncertainties.ufloat,
        sigma: uncertainties.ufloat,
        file_name: str,
        vertical_lines: List[float] = None,
        center_bins_on_zero=True,
        extra_info=None,
        title_info=None,
        xlabel: str = '$(\mu-\mu_{in}) /\sigma_{\mu}$',
        ylabel='Trials',
        gauss_color=None,
        figsize=(6, 6 / 1.618),
        show: bool = False,
):
    """
    Small function to plot a histogram and a fitted gauss. The fit must already have been done separately

    Args:
        x (list): Entries to be histogrammed.
        mu (uncertainties.ufloat): mu of gauss, should be a ufloat with uncertainty.
        sigma (uncertainties.ufloat): sigma of gauss, should be a ufloat with uncertainty.
        file_name (str): output file_name.
        center_bins_on_zero (bool, optional): Wether to center the plot on 0 or mu. Defaults to True.
        extra_info (_type_, optional): Info written inside top left of plot. Defaults to None.
        title_info (_type_, optional): Info written on the right of the title line. Defaults to None.
        gauss_color (_type_, optional): Colour of gaussian line. Defaults to None.
    """

    fig = plt.figure(figsize=figsize)

    if center_bins_on_zero:
        bins = np.linspace(-5 * sigma.n, +5 * sigma.n, 101)
    else:
        bins = np.linspace(mu.n - 5 * sigma.n, mu.n + 5 * sigma.n, 101)

    bin_mids = 0.5 * (bins[1:] + bins[:-1])
    bin_width = bins[1] - bins[0]
    gaussian_x = np.linspace(bins[0], bins[-1], 2001)
    gaussian_y = gaussian(gaussian_x, mu, sigma)

    hist, _ = np.histogram(x, bins=bins)
    g = plt.plot(gaussian_x, sum(hist) * bin_width * unp.nominal_values(gaussian_y), lw=1, color=gauss_color)
    plt.fill_between(gaussian_x,
                     sum(hist) * bin_width * (unp.nominal_values(gaussian_y) + unp.std_devs(gaussian_y)),
                     sum(hist) * bin_width * (unp.nominal_values(gaussian_y) - unp.std_devs(gaussian_y)),
                     color=g[0].get_color(),
                     alpha=0.3)

    plt.errorbar(
        x=bin_mids,
        y=hist,
        yerr=np.sqrt(hist),
        fmt='.',
        color='black',
        markeredgecolor='white',
        markeredgewidth=0.5,
    )

    plt.ylim(0)
    plt.xlim(bins[0], bins[-1])
    if vertical_lines is not None:
        for v in vertical_lines:
            plt.axvline(v, color='gray', ls='--', zorder=-100)

    mu_str, mu_err_str = round_pdg_style_str(mu.n, mu.s)
    sigma_str, sigma_err_str = round_pdg_style_str(sigma.n, sigma.s)

    plt.text(0.95,
             0.95,
             fr'$\mu_{{G}}=${mu_str}$\pm${mu_err_str}',
             va='top',
             ha='right',
             usetex=False,
             transform=plt.gca().transAxes)
    plt.text(0.95,
             0.88,
             fr'$\sigma_{{G}}=${sigma_str}$\pm${sigma_err_str}',
             va='top',
             ha='right',
             usetex=False,
             transform=plt.gca().transAxes)

    if extra_info is not None:
        plt.text(0.05, 0.95, extra_info, va='top', ha='left', usetex=False, transform=plt.gca().transAxes, fontsize=18)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if title_info is not None:
        plt.title(title_info, loc='right')

    plt.savefig(file_name, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_linearity_test(
        x: Union[list, List[list]],
        y: Union[list, List[list]],
        yerr: Union[list, List[list]],
        slope: Union[uncertainties.ufloat, List[uncertainties.ufloat]],
        intercept: Union[uncertainties.ufloat, List[uncertainties.ufloat]],
        color: Union[str, List[str]],
        file_name: str,
        bonds: Optional[List[float]] = [0,1],
        x_offset: Optional[List[float]] = [0],
        extra_info=None,
        title_info=None,
        line_infos: Optional[List[str]] = None,
        xlabel: str = r'$\mu_{in}$',
        ylabel=r'$\mu$',
        figsize=(6, 6 / 1.618),
        show: bool = False,
):

    # assume the shapes are all the same and hope
    if not isinstance(slope, (list, tuple)):
        slope = [slope]
        intercept = [intercept]
        y = [y]
        yerr = [yerr]
        color = [color]

    plt.figure(figsize=figsize)
    x_array_line = np.linspace(bonds[0], bonds[1], 1001)

    for i in range(len(slope)):
        slope_str, slope_err_str = round_pdg_style_str(slope[i].n, slope[i].s)
        intercept_str, intercept_err_str = round_pdg_style_str(intercept[i].n, intercept[i].s)

        # plot the fit:
        y_line = x_array_line * slope[i] + intercept[i]
        line = plt.plot(np.array([x_array_line[0], x_array_line[-1]]) + x_offset[i],
                        unp.nominal_values([y_line[0], y_line[-1]]),
                        color=color[i],
                        lw=1.0)
        plt.fill_between(x_array_line + x_offset[i],
                         unp.nominal_values(y_line) + unp.std_devs(y_line),
                         unp.nominal_values(y_line) - unp.std_devs(y_line),
                         color=line[0].get_color(),
                         alpha=0.3)

        if intercept[i] >= 0:
            plusminus = '+'
        else:
            plusminus = '-'

        eq = f"""({slope_str}$\pm${slope_err_str})$\mu_{{in}}$${plusminus}$({intercept_str[1:]}$\pm${intercept_err_str})"""
        plt.text(0.02,
                 0.85 + 0.07 * (len(slope) - 1 - i),
                 eq,
                 usetex=False,
                 transform=plt.gca().transAxes,
                 color=line[0].get_color(),
                 fontsize=9)

        plt.errorbar(np.array(x) + x_offset[i], np.array(y[i]), yerr=yerr[i], label=None, fmt='.', color='k')
        plt.plot(np.array([bonds[0], bonds[1]]) + x_offset[i], [bonds[0], bonds[1]], color='gray', label='Diagonal', lw=0.5, zorder=-100, ls='--')

        # if we have multiple lines put the extra info there
        if line_infos is not None:
            plt.text(x[-1] + x_offset[i],
                     y[i][-1] + 0.12 * (y[i][-1] - y[i][0]),
                     line_infos[i],
                     va='top',
                     ha='center',
                     usetex=False,
                     fontsize=9,
                     color=line[0].get_color())
            if i == 0:
                bot, top = plt.ylim()
                plt.ylim(bot, top + 0.08 * (top - bot))

    if extra_info is not None:
        plt.text(0.05, 0.95, extra_info, va='top', ha='left', usetex=False, transform=plt.gca().transAxes, fontsize=18)

    left, right = plt.xlim()
    plt.xlim(left, right + 0.1 * (right - left))

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if title_info is not None:
        plt.title(title_info, loc='right')
    plt.legend()

    plt.savefig(file_name, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

