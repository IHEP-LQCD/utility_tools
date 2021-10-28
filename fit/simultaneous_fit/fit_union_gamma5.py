#!/usr/bin/env python
import numpy as np
import gvar as gv
import lsqfit
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
from matplotlib import rcParams
from matplotlib.ticker import (
    MultipleLocator,
    FormatStrFormatter,
    AutoLocator,
    AutoMinorLocator,
)

mpl.rc("xtick", labelsize=22)
mpl.rc("ytick", labelsize=22)
rcParams["xtick.major.size"] = 8
rcParams["xtick.major.width"] = 1.1
rcParams["xtick.minor.size"] = 4
rcParams["xtick.minor.width"] = 1.0

rcParams["ytick.major.size"] = 8
rcParams["ytick.major.width"] = 1.1
rcParams["ytick.minor.size"] = 4
rcParams["ytick.minor.width"] = 1.0
mpl.rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
mpl.rc("text", usetex=True)
# plt.style.use('dark_background')

NT = 128    # number of time slice


def parse_args():
    parser = argparse.ArgumentParser(description="Fitting program")
    parser.add_argument("-i", "--infile", nargs="+",
                        type=str, help="input file name")
    parser.add_argument("-o", "--outdir", type=str, help="output directory")
    parser.add_argument(
        "-dim",
        "--dim",
        nargs="+",
        type=int,
        help="rank of every dimension in input data",
    )
    parser.add_argument(
        "-xl", "--xlow", nargs="+", type=float, help="mininum of x axis of plot range"
    )
    parser.add_argument(
        "-xh", "--xhigh", nargs="+", type=float, help="maxmium of x axis plot range"
    )
    parser.add_argument(
        "-yl", "--ylow", nargs="+", type=float, help="mininum of y axis of plot range"
    )
    parser.add_argument(
        "-yh", "--yhigh", nargs="+", type=float, help="maxmium of y axis of plot range"
    )
    parser.add_argument(
        "-rx",
        "--xrange",
        nargs=2,
        type=int,
        help="mininum and maximum of fitting range",
    )
    parser.add_argument(
        "-ry",
        "--yrange",
        nargs=2,
        type=float,
        help="mininum and maximum of y axis in tmin plot",
    )
    parser.add_argument(
        "-rmin",
        "--rmin",
        nargs="+",
        type=int,
        help="mininum of simultaneous fitting range",
    )
    parser.add_argument(
        "-rmax",
        "--rmax",
        nargs="+",
        type=int,
        help="maximum of simultaneous fitting range",
    )
    parser.add_argument(
        "-choose",
        "--choose",
        nargs="+",
        type=int,
        help="choose the optimal lower fitting bound to plot",
    )
    parser.add_argument(
        "-x", "--xlabel", type=str, default="$t_{min}$", help="x-axis label of the plot"
    )
    parser.add_argument(
        "-y", "--ylabel", type=str, default="$Ea_t$", help="y-axis label of the plot"
    )
    parser.add_argument(
        "-tkxy",
        "--tkxy",
        nargs=4,
        type=float,
        default=[5.0, 1.0, 1.0, 0.1],
        help="Major and minor ticks of X and Y axis",
    )

    return parser.parse_args()


def jacksample(data, jdim):
    dsum = np.sum(data, axis=jdim)
    njdim = (data.shape)[jdim]
    temp = data.copy()
    slc = [slice(None)] * data.ndim
    for i in np.arange(njdim):
        slc[jdim] = slice(i, i + 1)
        temp[tuple(slc)] = (dsum - data[tuple(slc)]) / (njdim - 1)

    return temp


def jackerror(data, jdim):
    njdim = (data.shape)[jdim]
    err = np.sqrt((njdim - 1) * np.var(data, axis=jdim))
    mean = np.mean(data, axis=jdim)

    return mean, err


# data for 2pt of eta_c, glueball and off-diagnol term
def make_data_all(corre, tmin, tmax):

    x = {}
    x["tc"] = np.arange(tmin[0], tmax[0] + 1)
    x["tg"] = np.arange(tmin[1], tmax[1] + 1)
    x["tgc"] = np.arange(tmin[2], tmax[2] + 1)

    data = {}
    data["CC"] = corre[0][:, tmin[0]: tmax[0] + 1]
    data["GG"] = corre[1][:, tmin[1]: tmax[1] + 1]
    data["GC"] = corre[2][:, tmin[2]: tmax[2] + 1]
    print("CC: xmin = {}, xmax = {}".format(tmin[0], tmax[0]))
    print("GG: xmin = {}, xmax = {}".format(tmin[1], tmax[1]))
    print("GC: xmin = {}, xmax = {}".format(tmin[2], tmax[2]))

    datacov = gv.dataset.avg_data(data)
    # print(gv.evalcov(datacov))

    return x, datacov


def fcn_all_twostate_theta(x, p):

    ans = {}
    Zc1 = p["Zc1"]
    Zc2 = p["Zc2"]
    Zcg = p["Zcg"]
    Mc1 = p["Mc1"]
    Mc2 = p["Mc2"]

    Zg1 = p["Zg1"]
    Zg2 = p["Zg2"]
    Mg1 = p["Mg1"]
    Mg2 = p["Mg2"]

    theta_1 = p["theta_1"]
    theta_2 = p["theta_2"]

    corr_cc_1 = gv.exp(-Mc1 * x["tc"]) + gv.exp(-Mc1 * (NT - x["tc"]))
    corr_cc_2 = gv.exp(-Mc2 * x["tc"]) + gv.exp(-Mc2 * (NT - x["tc"]))

    corr_gg_1 = gv.exp(-Mg1 * x["tc"]) + gv.exp(-Mg1 * (NT - x["tc"]))
    corr_gg_2 = gv.exp(-Mg2 * x["tc"]) + gv.exp(-Mg2 * (NT - x["tc"]))

    ans["CC"] = Zc1 * (
        gv.cos(theta_1) ** 2 * corr_cc_1 + gv.sin(theta_1) ** 2 * corr_gg_1
    ) + Zc2 * (gv.cos(theta_2) ** 2 * corr_cc_2 + gv.sin(theta_2) ** 2 * corr_gg_2)

    corr_cc_1 = gv.exp(-Mc1 * x["tg"]) + gv.exp(-Mc1 * (NT - x["tg"]))
    corr_cc_2 = gv.exp(-Mc2 * x["tg"]) + gv.exp(-Mc2 * (NT - x["tg"]))

    corr_gg_1 = gv.exp(-Mg1 * x["tg"]) + gv.exp(-Mg1 * (NT - x["tg"]))
    corr_gg_2 = gv.exp(-Mg2 * x["tg"]) + gv.exp(-Mg2 * (NT - x["tg"]))

    ans["GG"] = Zg1 * (
        gv.cos(theta_1) ** 2 * corr_gg_1 + gv.sin(theta_1) ** 2 * corr_cc_1
    ) + Zg2 * (gv.cos(theta_2) ** 2 * corr_gg_2 + gv.sin(theta_2) ** 2 * corr_cc_2)

    ans["GC"] = -(
        np.sqrt(gv.abs(Zg1 * Zc1))
        * gv.cos(theta_1)
        * gv.sin(theta_1)
        * (
            gv.exp(-Mg1 * x["tgc"])
            - gv.exp(-Mg1 * (NT - x["tgc"]))
            - gv.exp(-Mc1 * x["tgc"])
            + gv.exp(-Mc1 * (NT - x["tgc"]))
        )
        + np.sqrt(gv.abs(Zg2 * Zc2))
        * gv.cos(theta_2)
        * gv.sin(theta_2)
        * (
            gv.exp(-Mg2 * x["tgc"])
            - gv.exp(-Mg2 * (NT - x["tgc"]))
            - gv.exp(-Mc2 * x["tgc"])
            + gv.exp(-Mc2 * (NT - x["tgc"]))
        )
    )

    return ans


def make_prior_all_twostate_theta():
    prior = gv.BufferDict()
    prior["log(Mc1)"] = gv.log(gv.gvar(0.28, 0.2))
    prior["log(Mc2)"] = gv.log(gv.gvar(0.4, 0.2))
    prior["log(Mg1)"] = gv.log(gv.gvar(0.24, 0.2))
    prior["log(Mg2)"] = gv.log(gv.gvar(0.4, 0.2))
    prior["Zc1"] = gv.gvar(50, 50)
    prior["Zc2"] = gv.gvar(10, 10)
    prior["Zcg"] = gv.gvar(0.02, 0.05)
    prior["Zg1"] = gv.gvar(1, 5)
    prior["Zg2"] = gv.gvar(0, 1)
    prior["theta_1"] = gv.gvar(0.1, 0.2)
    prior["theta_2"] = gv.gvar(-0.3, 0.2)

    return prior


def plot_fit_tmin(
    options, paramlabel, num, xlabel, ylabel, choose_t, rmin, rmax, paramfit
):
    index = choose_t - rmin
    outdir = options.outdir

    color_data = "blue"
    color_xylabel = "black"
    color_data_choose = "black"
    color_chi2 = "magenta"
    color_chi2_choose = "black"

    for i, param in enumerate(paramlabel):
        (chi2dof, Q, paramout) = paramfit
        print("param = ", param)
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        t = np.arange(rmin, rmin + 8)
        choose_t = t[index]
        choose_res = gv.mean(paramout[param][index])
        choose_err = gv.sdev(paramout[param][index])
        choose_chi2 = chi2dof[index]
        res_fit = gv.mean(paramout[param])
        res_err = gv.sdev(paramout[param])

        ax1.set_title("$t_{max} = " + str(rmax) + "$", fontsize=18)
        ax1.set_ylabel(ylabel[i], fontsize=22, color=color_xylabel)

        # plot the chosen one of fitting
        ax1.errorbar(
            choose_t,
            choose_res,
            choose_err,
            fmt="o",
            color=color_data_choose,
            markersize=7,
            capsize=6,
            linewidth=1.0,
            markeredgecolor="black",
            capthick=1.0,
            fillstyle="full",
        )

        # plot other points
        t = np.delete(t, index)
        res_fit = np.delete(res_fit, index)
        res_err = np.delete(res_err, index)
        ax1.errorbar(
            t,
            res_fit,
            res_err,
            fmt="o",
            color=color_data,
            markersize=7,
            capsize=6,
            linewidth=1.0,
            markeredgecolor="black",
            capthick=1.0,
            fillstyle="full",
        )

        ax2.set_xlabel(xlabel, fontsize=18)
        ax2.set_ylabel(r"$\chi^2/dof$", fontsize=16, color=color_xylabel)
        ax2.set_ylim(0, 2)
        ax2.plot(
            choose_t,
            choose_chi2,
            "D",
            color=color_chi2_choose,
            linewidth=1.0,
            markersize=7,
            markeredgecolor="black",
            fillstyle="full",
        )
        chi2dof = np.delete(chi2dof, index)
        ax2.plot(
            t,
            chi2dof,
            "D",
            color=color_chi2,
            linewidth=1.0,
            markersize=7,
            markeredgecolor="black",
            fillstyle="full",
        )

        plt.tight_layout()
        fig.subplots_adjust(hspace=0.1)
        plt.savefig(outdir + param + "_" + str(num) + ".pdf")
        plt.show()


def bin_data(corre, dim, binsize):
    nsample = corre.shape[dim]
    newsize = nsample // binsize

    newdim = list(corre.shape)
    subdim = list(corre.shape)

    newdim[dim] = newsize  # number of index in binned dimension
    subdim[dim] = 1  # set the rank to be 1 for averaged binned dimension

    data = np.zeros(tuple(newdim))
    print("Binning data, shape ", data.shape)

    slcin = [slice(None)] * data.ndim
    slcout = [slice(None)] * data.ndim
    for i in range(newsize):
        slcin[dim] = slice(i * binsize, (i + 1) * binsize)
        slcout[dim] = slice(i, i + 1)
        data[tuple(slcout)] = np.mean(corre[tuple(slcin)], axis=dim).reshape(
            tuple(subdim)
        )

    return data


def read_data(dim, infile):
    data = []
    # load input data files
    for i in infile:
        print("read file: {}".format(i))
        data.append(np.real(np.load(i)).reshape(*dim))

    return data


def plot_effem_and_fit(options, num, labels, data, fcnlabel, fcn, choosefit):
    xlow = options.xlow
    xhigh = options.xhigh
    ylow = options.ylow
    yhigh = options.yhigh
    tkxy = options.tkxy
    xmax = options.rmax[num]
    choose_t = options.choose[num]
    outdir = options.outdir

    t = np.arange(NT)
    x = {"tc": t, "tg": t, "tgc": t}
    fcnout = fcn(x, choosefit.p)[fcnlabel]
    effemfit = gv.log(fcnout / np.roll(fcnout, -1, axis=0))
    upper = gv.mean(effemfit) + gv.sdev(effemfit)
    lower = gv.mean(effemfit) - gv.sdev(effemfit)

    idata = data.copy()
    jacksamp = jacksample(idata, 0)
    _, jkerr = jackerror(
        np.log(jacksamp / np.roll(jacksamp, -1, axis=1)), 0
    )  # * at_inverse
    twopt_mean = np.average(idata, axis=0)
    exmean = np.log(twopt_mean / np.roll(twopt_mean, -1, axis=0))

    fig, ax = plt.subplots()
    ax.errorbar(
        t,
        exmean,
        jkerr,
        fmt="o",
        color="r",
        markersize=5,
        capsize=4,
        linewidth=1.0,
        capthick=1.0,
        markeredgecolor="black",
        fillstyle="full",
        markerfacecolor="w",
    )

    ax.fill_between(t, lower, upper, alpha=0.5, color="dimgray", lw=0.2)
    ax.fill_between(
        np.arange(choose_t, xmax + 1),
        lower[choose_t: xmax + 1],
        upper[choose_t: xmax + 1],
        alpha=1.0,
        color="tomato",
        lw=0.2,
    )

    ax.xaxis.set_major_locator(MultipleLocator(tkxy[0]))
    ax.xaxis.set_minor_locator(MultipleLocator(tkxy[1]))
    # ax.yaxis.set_major_locator(MultipleLocator(tkxy[2]))
    # ax.yaxis.set_minor_locator(MultipleLocator(tkxy[3]))

    ax.set_xlabel(options.xlabel, fontsize=24)
    ax.set_ylabel(labels, fontsize=24)
    ax.set_xlim(xlow[num], xhigh[num])
    ax.set_ylim(ylow[num], yhigh[num])

    plt.tight_layout()
    plt.savefig(outdir + "fit_" + fcnlabel + ".pdf")
    plt.show()


def plot_corre_and_fit(options, num, labels, data, fcnlabel, fcn, choosefit):
    xlow = options.xlow
    xhigh = options.xhigh
    ylow = options.ylow
    yhigh = options.yhigh
    tkxy = options.tkxy
    xmax = options.rmax[num]
    choose_t = options.choose[num]
    outdir = options.outdir

    t = np.arange(NT)
    x = {"tc": t, "tg": t, "tgc": t}
    fcnout = -(fcn(x, choosefit.p)[fcnlabel])

    upper = gv.mean(fcnout) + gv.sdev(fcnout)
    lower = gv.mean(fcnout) - gv.sdev(fcnout)

    idata = data.copy()
    jacksamp = jacksample(idata, 0)
    _, jkerr = jackerror(jacksamp, 0)  # * at_inverse
    exmean = -(np.average(idata, axis=0))

    fig, ax = plt.subplots()
    ax.errorbar(
        t,
        exmean,
        jkerr,
        fmt="o",
        color="blue",
        markersize=5,
        capsize=4,
        linewidth=1.0,
        capthick=1.0,
        markeredgecolor="black",
        fillstyle="full",
        markerfacecolor="w",
    )

    ax.fill_between(t, lower, upper, alpha=0.5, color="dimgray")
    ax.fill_between(
        np.arange(choose_t, xmax + 1),
        lower[choose_t: xmax + 1],
        upper[choose_t: xmax + 1],
        alpha=1.0,
        color="royalblue",
    )

    ax.xaxis.set_major_locator(MultipleLocator(tkxy[0]))
    ax.xaxis.set_minor_locator(MultipleLocator(tkxy[1]))
    # ax.yaxis.set_major_locator(MultipleLocator(tkxy[2]))
    # ax.yaxis.set_minor_locator(MultipleLocator(tkxy[3]))

    ax.set_xlabel(options.xlabel, fontsize=24)
    ax.set_ylabel(labels, fontsize=24)
    ax.set_xlim(xlow[num], xhigh[num])
    ax.set_ylim(ylow[num], yhigh[num])

    plt.tight_layout()
    plt.savefig(outdir + "fit_" + fcnlabel + ".pdf")
    plt.show()


# fit the data with varing lower bound of fitting range
def do_fit(options, paramlabel, icorre, data):
    xmin = options.xrange[0]
    xmax = options.xrange[1]
    choose_t = options.choose
    rmin = options.rmin
    rmax = options.rmax

    paramout = {}
    chi2dof = []
    Q = []
    for i in paramlabel:
        paramout[i] = []

    newtmin = choose_t.copy()
    for tmin in range(rmin[icorre], rmin[icorre] + 8):
        print(
            "*" * 25
            + "tmin = "
            + str(tmin)
            + ", tmax = "
            + str(rmax[icorre])
            + "*" * 25
        )
        newtmin[icorre] = tmin
        x, y = make_data_all(data, newtmin, rmax)

        prior = make_prior_all_twostate_theta()

        fit = lsqfit.nonlinear_fit(
            data=(x, y), fcn=fcn_all_twostate_theta, prior=prior, p0=None
        )
        print(fit)

        if tmin == choose_t[icorre]:
            print("-" * 25 + "choosed tmin = " + str(tmin) + "-" * 25)
            print("\n")
            choose_fit = fit

        for i in paramlabel:
            paramout[i].append(fit.p[i])

        chi2dof.append(fit.chi2 / fit.dof)
        Q.append(fit.Q)

    return (chi2dof, Q, paramout), choose_fit


def main():
    options = parse_args()
    infile = options.infile
    outdir = options.outdir
    dim = options.dim
    choose_t = options.choose
    rmin = options.rmin
    rmax = options.rmax

    labels = [r"$m_{CC}^{eff}(t)a_t$", r"$m_{GG}^{eff}(t)a_t$", r"$C_{GC}(t)$"]

    data = read_data(dim, infile)
    # normalize the data with GG 2pt at t=0
    avegg = np.average(data[1], axis=0)
    normgg = avegg[1]
    print("avegg: {}".format(normgg))
    data0 = data[0]
    data1 = data[1] / normgg
    data2 = data[2] / np.sqrt(normgg)
    data = [data0, data1, data2]
    print(data2)

    paramlabel = ["Mc1", "Mg1", "Mc2", "Mg2", "theta_1"]
    xlabels = [r"$t_{cc_{min}}/a_t$",
               r"$t_{gg_{min}}/a_t$", r"$t_{gc_{min}}/a_t$"]
    ylabels = [
        r"$m_{\eta_1}a_t$",
        r"$m_{g_1}a_t$",
        r"$m_{\eta_2}a_t$",
        r"$m_{g_2}a_t$",
        r"$\theta_1$",
    ]

    for i in range(3):
        paramout, choose_fit = do_fit(options, paramlabel, i, data)
        plot_fit_tmin(
            options,
            paramlabel,
            i,
            xlabels[i],
            ylabels,
            choose_t[i],
            rmin[i],
            rmax[i],
            paramout,
        )

    plot_effem_and_fit(
        options, 0, labels[0], data[0], "CC", fcn_all_twostate_theta, choose_fit
    )
    plot_effem_and_fit(
        options, 1, labels[1], data[1], "GG", fcn_all_twostate_theta, choose_fit
    )
    plot_corre_and_fit(
        options, 2, labels[2], data[2], "GC", fcn_all_twostate_theta, choose_fit
    )


if __name__ == "__main__":
    main()
