#!/usr/bin/env python
'''
Description: 
version: 1.0
Author: Wei Sun
Date: 2020-08-11 15:21:25
LastEditors: Wei Sun
LastEditTime: 2020-08-11 18:07:15
'''
import numpy as np
import matplotlib.pyplot as plt
import gvar as gv
import matplotlib as mpl
import sys
from matplotlib import rcParams
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoLocator, AutoMinorLocator)
import os

mpl.rc('xtick', labelsize=38)
mpl.rc('ytick', labelsize=38)
mpl.rc('text', usetex=True)

rcParams['axes.linewidth'] = 1.6

rcParams['xtick.major.size'] = 8
rcParams['xtick.major.width'] = 1.6
rcParams['xtick.minor.size'] = 4
rcParams['xtick.minor.width'] = 1.4

rcParams['ytick.major.size'] = 8
rcParams['ytick.major.width'] = 1.6
rcParams['ytick.minor.size'] = 4
rcParams['ytick.minor.width'] = 1.4

def get_data(infile):
    data_reg = np.loadtxt(infile)
    data = gv.gvar(data_reg[:,0], data_reg[:,1])
    
    return data

def plot_mass(datain, ylabel, ytick, ylim, filename, loc='upper left'):
    xlabel = ['$0^{-+}$', '$1^{--}$', 
              '$0^{++}$', '$1^{++}$', '$1^{+-}$', '$2^{++}$',
              '$2^{-+}$', '$1^{-+}$']
    color = ['k', 'b', 
            'g', 'r', 'tab:olive', 'orange', 
            'darkblue', 'dimgray']
    
    data = []
    for i in range(len(datain)):
        data_mean = gv.mean(datain[i])
        data_err = gv.sdev(datain[i])
        data_upper = data_mean + data_err
        data_lower = data_mean - data_err
        data.append([data_lower, data_upper])

    fig, ax = plt.subplots(figsize=(16, 8))
    plt.xticks(np.arange(len(xlabel)), xlabel)
    lbandwidth=0.42
    rbandwidth=0.02
    for i in range(len(xlabel)):
        ax.fill_between([i-lbandwidth, i-rbandwidth],
                        data[0][0][i], data[0][1][i],
                        alpha=0.88, color=color[0], lw=0.8)
        ax.fill_between([i+rbandwidth, i+lbandwidth],
                        data[1][0][i], data[1][1][i],
                        alpha=0.88, color=color[1], lw=0.8)
    for i in range(len(data[2][0])):
        ax.fill_between([i-lbandwidth, i+lbandwidth],
                        data[2][0][i], data[2][1][i],
                        alpha=0.88, color=color[2], lw=0.8)
    
    plt.axvspan(6+lbandwidth+0.1, 7-lbandwidth-0.1, facecolor='#d62700', alpha=0.5)

    ax.set_ylabel(ylabel, fontsize=36)
    ax.set_ylim(ylim[0], ylim[1])
    ax.yaxis.set_major_locator(MultipleLocator(ytick[0]))
    ax.yaxis.set_minor_locator(MultipleLocator(ytick[1]))

    import matplotlib.patches as mpatches
    patch1 = mpatches.Patch(color=color[0], alpha=0.88, label=r"$32^3\times 64, a=0.0828~\mathrm{fm}$")
    patch2 = mpatches.Patch(color=color[1], alpha=0.88, label=r"$48^3\times 96, a=0.0711~\mathrm{fm}$")
    patch3 = mpatches.Patch(color=color[2], alpha=0.88, label=r"$\mathrm{PDG}$")
    ax.legend(handles=[patch1, patch2, patch3],loc=loc, fontsize=27)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def main():
    data_prefix="./data/"
    output_prefix="./output/"
    physical_mass_32 = get_data(data_prefix+"result.32.Mass")
    physical_mass_48 = get_data(data_prefix+"result.48.Mass")

    #PDG value of 0-+, 1--, 0++, 1++, 1+-, 2++
    pdg_mass = gv.gvar(['2.9839(5)', '3.096900(6)', '3.41471(30)', '3.51067(5)',
            '3.52538(11)', '3.55617(7)'])

    plot_mass([physical_mass_32, physical_mass_48, pdg_mass], 
            r'$M(\mathrm{GeV})$', [0.5, 0.1], [2.5, 5.0], 
            output_prefix+'mass_pdg_32_48.png')

if __name__ == '__main__':
    main()
