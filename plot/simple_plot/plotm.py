#!/usr/bin/env python
'''
Description: plot scripts
version: 1.0
Author: Wei Sun
Date: 2020-11-05 23:04:30
LastEditors: Wei Sun
LastEditTime: 2020-11-06 09:13:16
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import sys
from matplotlib import rcParams
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoLocator, AutoMinorLocator)
import os

mpl.rc('xtick', labelsize=22)
mpl.rc('ytick', labelsize=22)
mpl.rc('text', usetex=True)

rcParams['axes.linewidth'] = 1.2

rcParams['xtick.major.size'] = 8
rcParams['xtick.major.width'] = 1.4
rcParams['xtick.minor.size'] = 4
rcParams['xtick.minor.width'] = 1.2

rcParams['ytick.major.size'] = 8
rcParams['ytick.major.width'] = 1.4
rcParams['ytick.minor.size'] = 4
rcParams['ytick.minor.width'] = 1.2


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--infile', nargs='+')
    parser.add_argument('-o', '--outfile')
    parser.add_argument('-xl', '--xlow')
    parser.add_argument('-xh', '--xhigh')
    parser.add_argument('-yl', '--ylow')
    parser.add_argument('-yh', '--yhigh')
    parser.add_argument('-s', '--scale', nargs='+', type=float, default=[1.0])
    parser.add_argument('-x', '--xlabel',    type=str,
                        default="$t/a_t$", help="x-axis label of the plot")
    parser.add_argument('-y', '--ylabel',    type=str,
                        default="$m_{eff}(t)$", help="y-axis label of the plot")
    parser.add_argument('-l', '--logy',      action="store_true",
                        default=False, help="whether to use log scale in y-axis")
    parser.add_argument('-b', '--binary',    action="store_true",
                        default=False, help="whether the file is binary or not")
    parser.add_argument('-c', '--col', nargs='+', type=int,
                        help="Columns need to draw, counting from 0")
    parser.add_argument('-tkxy', '--tkxy', nargs=4, type=float, default=[5.0, 1.0, 1.0, 0.1],
                        help="Major and minor ticks of X and Y axis")
    parser.add_argument('-hline', '--hline', nargs='+', type=float,
                        help="Plot a horizontal line")
    parser.add_argument('-errband', '--errband', nargs='+', type=float,
                        help="Plot the error band: mean, err, xmin, ymin")
    return parser.parse_args()


def main():
    options = parse_args()
    infile = options.infile
    xl = options.xlow
    xh = options.xhigh
    yl = options.ylow
    yh = options.yhigh
    s = options.scale
    x = options.xlabel
    y = options.ylabel
    outfile = options.outfile
    logy = options.logy
    binary = options.binary
    col = options.col
    tkxy = options.tkxy
    hline = options.hline
    errband = options.errband

    color = ['r', 'b', 'g', 'k', 'gray', 'lightcoral', 'saddlebrown', 'gold', 'olive', 'lime',
             'powderblue', 'deepskyblue', 'cyan', 'royalblue','navy', 'violet', 'purple', 'magneta',
             'hotpink', 'pink']
    scale = []
    for i in s:
        if len(s) == 1:
            scale = [float(i)] * len(infile)
        elif len(s) < len(infile):
            raise ValueError(
                "Length of scale multiply parameter {} should match with number of input files {}".format(s, infile))
        else:
            scale.append(float(i))

    fig, ax = plt.subplots()

    j = 0
    for i in infile:
        if binary:
            indata = np.fromfile(i)
            row = indata.size

            ax.plot(np.arange(row), indata,
                    marker='s', markeredgewidth=1.2, linewidth=1.1,
                    linestyle='dashed',
                    fillstyle='full', markersize=6,
                    label="$"+i+"$")
        elif not binary:
            data = np.loadtxt(i)
            nrow = data.shape[0]
            if len(col) == 1 and nrow == data.size:
                ax.plot(np.arange(nrow), data[:],
                        marker='s', markeredgewidth=1.0, linewidth=1.1,
                        linestyle='dashed', color=color[j],
                        fillstyle='full',
                        markersize=6,
                        label="$"+i+"$")
                j += 1
            elif len(col) == 1 and nrow != data.size:
                ax.plot(np.arange(nrow), data[:, col],
                        marker='s', markeredgewidth=1.0, linewidth=1.1,
                        linestyle='dashed', markeredgecolor='black',
                        fillstyle='full', color=color[j],
                        markersize=6,
                        label="$"+i+"$")
                j += 1
            elif len(col) == 2:
                ax.plot(data[:, col[0]], data[:, col[1]],
                        marker='s', markeredgewidth=1.0, linewidth=1.1,
                        linestyle='dashed', markeredgecolor='black',
                        fillstyle='full', markersize=6, alpha=0.95,
                        color=color[j],
                        label="$"+i+"$")
                j += 1
            elif len(col) == 3:
                xdata = data[:, col[0]]
                ydata = (data[:, col[1]]) * scale[j]
                ydata_err = data[:, col[2]] * scale[j]
                ax.errorbar(xdata, ydata, ydata_err,
                            fmt='s', linewidth=1.5,
                            markersize=6, capsize=7, capthick=1.5,
                            markeredgecolor='black',  # markeredgewidth=0.85,
                            fillstyle='full', alpha=0.95,
                            linestyle='dashed', color=color[j],
                            label="$"+i+"$")
                if errband is not None:
                    mean_band, error_band = errband[0], errband[1]
                    start, end = int(errband[2]), int(errband[3])
                    errband_xdata = xdata[start:end+1]
                    errband_ydata = np.full(errband_xdata.shape, mean_band)
                    errband_ydata_low = errband_ydata - error_band
                    errband_ydata_high = errband_ydata + error_band

                    ax.fill_between(errband_xdata, errband_ydata_low, errband_ydata_high,
                                    alpha=0.74, color='r')

                # save the scaled data to another file
                #out = np.zeros((data.shape[0], len(col)))
                #datalist = [data[:, col[0]], data[:, col[1]]
                #            * scale[j], data[:, col[2]] * scale[j]]
                #np.stack(datalist, axis=1, out=out)
                #np.savetxt(os.getcwd() + '/' + os.path.dirname(i) + '/scaled.' +
                #           os.path.basename(i), out, ["%2d", "%20.8e", "%20.8e"])

                j += 1
            else:
                print("col chosen to draw should between 1 to 3")
                sys.exit()

    ax.set_xlim(float(xl), float(xh))
    ax.set_ylim(float(yl), float(yh))
    ax.set_xlabel(x, fontsize=22)
    ax.set_ylabel(y, fontsize=22)

    ax.xaxis.set_major_locator(MultipleLocator(tkxy[0]))
 #   ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(tkxy[1]))
 #   ax.xaxis.set_minor_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(MultipleLocator(tkxy[2]))
    ax.yaxis.set_minor_locator(MultipleLocator(tkxy[3]))

    if logy:
        ax.set_yscale('log')
        ax.set_ylim(auto=True)
     #   ax.yaxis.set_major_locator(AutoLocator())
     #   ax.yaxis.set_minor_locator(AutoMinorLocator())
    if hline is not None:
        for i in hline:
            plt.axhline(y=i, xmin=float(xl), xmax=float(
                xh), color='black', ls='--')

    ax.legend(fontsize=9, edgecolor='gray', framealpha=0.74)
    plt.tight_layout()
    plt.savefig(outfile+".png")
    plt.show()


if __name__ == '__main__':
    main()
