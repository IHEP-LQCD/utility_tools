#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import sys
from matplotlib import rcParams
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoLocator, AutoMinorLocator)
import os
import gvar as gv

mpl.rc('xtick', labelsize=24)
mpl.rc('ytick', labelsize=24)
rcParams['xtick.major.size'] = 4
rcParams['xtick.major.width'] = 1.1
rcParams['xtick.minor.size'] = 2
rcParams['xtick.minor.width'] = 1.0

rcParams['ytick.major.size'] = 4
rcParams['ytick.major.width'] = 1.1
rcParams['ytick.minor.size'] = 2
rcParams['ytick.minor.width'] = 1.0
mpl.rc('text', usetex=True)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--infile', nargs='+')
    parser.add_argument('-o', '--outfile')
    parser.add_argument('-x', '--xlabel',    type=str,
                        default="$t/a_t$", help="x-axis label of the plot")
    parser.add_argument('-y', '--ylabel',    type=str,
                        default="$m_{eff}(t)$", help="y-axis label of the plot")
    parser.add_argument('-subplot', '--subplot', type=int, nargs=2,
                        default=[1, 1], help="the number of rows and columns of subplot layout")
    parser.add_argument('-figsize', '--figsize', type=int, nargs=2,
                        default=[16, 9], help="the figure size of subplot")
    return parser.parse_args()


def main():
    options = parse_args()
    infile = options.infile
    outfile = options.outfile
    x = options.xlabel
    y = options.ylabel
    subplot = options.subplot
    figsize = options.figsize

    label = ['$0^{-+}$', '$1^{--}$', '$0^{++}$', '$1^{++}$',
             '$1^{+-}$', '$2^{++}$', '$2^{-+}$', '$1^{-+}$']
    xmin = [8, 8, 10, 9, 9,  5,  5, 3]
    xmax = [23, 23, 21, 22, 22, 16, 12, 11]

    # minimium of y axis of every row of the subplots
    ymin = [2.80, 2.90, 3.0, 3.0, 3.0, 3.0, 3.0, 3.5]
    ymax = [3.00, 3.15, 3.6, 4.0, 4.0, 4.0, 4.5, 5.5]
    yticks_major = [0.05, 0.05, 0.2,  0.2,  0.2,  0.2,  0.5, 0.5]
    yticks_minor = [0.01, 0.01, 0.04, 0.04, 0.04, 0.04, 0.1, 0.1]

    fig, ax = plt.subplots(
        subplot[0], subplot[1], sharex=False, sharey=False, squeeze=False,
        figsize=(figsize[0], figsize[1]))

    j = 0
    for i in infile:
        data = np.loadtxt(i)
        nrow = data.shape[0]
        data_mean = data[:, 1]
        data_err = data[:, 2]

        fit_mean = data[:, 3]
        fit_err = data[:, 4]
        fit_upper = fit_mean + fit_err
        fit_lower = fit_mean - fit_err

        plot_low = xmin[j] - 3
        plot_high = xmax[j] + 3

        m = j // subplot[1]
        n = j % subplot[1]
        myax = ax[m, n]

        myax.errorbar(np.arange(plot_low, plot_high),
                      np.abs(data_mean[plot_low:plot_high]),
                      data_err[plot_low:plot_high],
                      fmt='o', color='firebrick', alpha=0.84, linewidth=1.0,
                      markersize=7, capsize=6, capthick=1.0, markeredgecolor='black',
                      linestyle='none', fillstyle='full',
                      label=label[j])

        myax.set_ylim(float(ymin[j]), float(ymax[j]))

        myax.fill_between(np.arange(plot_low, plot_high), fit_lower[plot_low:plot_high],
                          fit_upper[plot_low:plot_high], alpha=0.25, color='lightseagreen')
        myax.fill_between(np.arange(xmin[j], xmax[j]+1), fit_lower[xmin[j]:xmax[j]+1],
                          fit_upper[xmin[j]:xmax[j]+1], alpha=0.95, color='teal')

        myax.legend(fontsize=22, edgecolor='gray',
                    framealpha=1, loc='lower left')

        myax.xaxis.set_major_locator(MultipleLocator(5))
 #       ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        myax.xaxis.set_minor_locator(MultipleLocator(1))
 #       ax.xaxis.set_minor_formatter(FormatStrFormatter('%d'))
        myax.yaxis.set_major_locator(MultipleLocator(yticks_major[j]))
        myax.yaxis.set_minor_locator(MultipleLocator(yticks_minor[j]))

        j += 1

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False,
                    left=False, right=False)
    plt.xlabel(x, fontsize=26)
    plt.ylabel(y, labelpad=30, fontsize=26)

    plt.tight_layout()
    plt.savefig(outfile+".png")
    plt.show()


if __name__ == '__main__':
    main()
