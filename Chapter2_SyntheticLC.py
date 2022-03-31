# get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
from IPython.display import Latex, Math
import numpy as np
from astroML.time_series import lomb_scargle
import peakutils
from numpy.fft import *
from scipy.signal import argrelextrema
import matplotlib.gridspec as gridspec
from matplotlib import cm
import pandas as pd


# class NewRotTerm(TermSum):
#     r"""A new kernel for stellar rotation envisioned by A. Cameron (St Andrews)
#         Implemented by J. Llama (Lowell). ACC to write up and publish
#     """
#     parameter_names = ("amp", "tdecay", "prot")
#
#     def __init__(self, **kwargs):
#         super(NewRotTerm, self).__init__(**kwargs)
#
#         ombeat = 1. / self.tdecay
#         omrot = 2. * np.pi / self.prot
#         c = ombeat
#         d = omrot
#         x = c / d
#         a = self.amp / 2.
#         b = self.amp * x / 2.
#         e = self.amp / 8.
#         f = self.amp * x / 4.
#         g = self.amp * (3. / 8. + 0.001)
#         self.terms = (
#             terms.ComplexTerm(a=a, b=b, c=c, d=d),
#             terms.ComplexTerm(a=e, b=f, c=c, d=2 * d),
#             terms.RealTerm(a=g, c=c)
#         )
#         self.coefficients = self.get_coefficients()

def lombscargle(x, y, yerr, Pmin=0.20, Pmax=20, res=1000):
    periods = np.linspace(Pmin, Pmax, res)
    ang_freqs = 2 * np.pi / periods
    powers = lomb_scargle(x, y, yerr, ang_freqs, generalized=True)
    return periods, powers


def plot_lcper5(target, time1, flux1, time2, flux2, time3, flux3, periods1, powers1, periods2, powers2, periods3, powers3, savehere):
    scatter_kwargs = {"zorder": 100}

    fig4 = plt.figure(num=None, figsize=(7, 4), facecolor='w')

    gs1 = gridspec.GridSpec(2, 1)
    gs1.update(left=0.0, right=0.57, hspace=0.0)
    ax1 = plt.subplot(gs1[0, 0])
    ax3 = plt.subplot(gs1[1, 0])

    gs2 = gridspec.GridSpec(2, 1)
    gs2.update(left=0.665, right=0.97, hspace=0.15)
    ax2 = plt.subplot(gs2[0, 0])
    ax4 = plt.subplot(gs2[1, 0])

    font_size = 'small'

    # plt.subplots_adjust(hspace=0)
    # ax1 = fig4.add_subplot(221)
    # ax1.set_ylabel(r'K2SC Flux (e$^{\rm -}$/s)', fontsize=font_size, style='normal', family='sans-serif')
    ax1.set_ylabel(r'Normalized Flux (ppt)', fontsize=font_size, style='normal', family='sans-serif')
    ax1.set_xlim([min(time1) - 1, max(time1) + 1])
    if target == 'CITau':
        ax1.set_ylim([-300, 375])
    ax1.plot(time1, flux1, color='#000000', lw=0.85, alpha=1)
    ax1.plot(time2, flux2, color='#ff6600', lw=0.65)
    ax1.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True)
    ax1.set_xticklabels([])

    # -------
    # ax2 = fig4.add_subplot(222)
    ax2.set_ylabel('Power', fontsize=font_size, style='normal', family='sans-serif')
    mx = max(periods1)
    ax2.set_xlim([0, mx])
    ax2.set_ylim([0, 0.5])
    ax2.plot(periods1, powers1, color='#000000', lw=0.85, alpha=1)
    ax2.plot(periods2, powers2, color='#ff6600', lw=0.65)  # ff8533
    # peaks = peakutils.indexes(powers1, thres=0.2, min_dist=10)
    # for i in peaks:
    #     ax2.vlines(periods1[peaks], 0, 0.5, color='#000000',lw=0.5, alpha=0.1, linestyles='--')
    #     #ax2.text(periods1[i] - 0.58, powers1[i] + 0.015, '%5.2f' % periods1[i], horizontalalignment='center', fontsize='x-small', style='normal', family='sans-serif')
    #     ax2.text(periods1[i], powers1[i] + 0.015, '%5.2f' % periods1[i], horizontalalignment='center', fontsize='x-small', style='normal', family='sans-serif')
    ax2.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True)
    # ax2.set_xticks((0,5,10,15,20,25,30,35))
    # ax2.set_xscale('symlog')
    # ax2.set_xticklabels([])

    # -----------------------------
    # ax3 = fig4.add_subplot(223)
    # ax41.set_title(target + '   ' + epic, fontsize=20)
    ax3.set_ylabel(r'Normalized Flux $-$ Model', fontsize=font_size, style='normal', family='sans-serif')
    ax3.set_xlabel('Time', fontsize=font_size, style='normal', family='sans-serif')
    ax3.set_xlim([min(time1) - 1, max(time1) + 1])
    if target == 'UZTauE':
        ax3.set_ylim([-20000, 20000])
    ax3.plot(time3, flux3, color='#000000', lw=0.5)
    ax3.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True)

    # -------
    # ax4 = fig4.add_subplot(224)
    ax4.set_ylabel('Power', fontsize=font_size, style='normal', family='sans-serif')
    ax4.set_xlabel('Period (d)', fontsize=font_size, style='normal', family='sans-serif')
    mx = max(periods3)
    ax4.set_xlim([0, mx])
    ax4.set_ylim([0, 0.175])
    ax4.plot(periods3, powers3, color='#000000', lw=0.5)
    ax4.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True)
    # ax4.set_xticks((0, 5, 10, 15, 20, 25, 30, 35))
    # -----------------------------

    # tight_layout()
    # fig4.savefig(savehere+target+'.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    # plt.close()


def plot_lcper6(target, time1, flux1, time2, flux2, time3, flux3, periods1, powers1, periods2, powers2, periods3, powers3, savehere):
    scatter_kwargs = {"zorder": 100}

    plt.close()


    # import pdb; pdb.set_trace()

    fig4 = plt.figure(num=None, figsize=(8, 4), facecolor='w')

    gs1 = gridspec.GridSpec(2, 1)
    gs1.update(left=0.0, right=0.57, hspace=0.0)
    ax1 = plt.subplot(gs1[0, 0])
    # ax3 = plt.subplot(gs1[1, 0])

    gs2 = gridspec.GridSpec(2, 1)
    gs2.update(left=0.665, right=0.97, hspace=0.0)
    ax2 = plt.subplot(gs2[0, 0])
    # ax4 = plt.subplot(gs2[1, 0])

    font_size = 'medium'

    plt.subplots_adjust(hspace=0)
    # ax1 = fig4.add_subplot(221)
    ax1.set_ylabel(r'Flux', fontsize=font_size, style='normal', family='sans-serif')
    ax1.set_xlabel('Time (d)', fontsize=font_size, style='normal', family='sans-serif')
    ax1.set_xlim([min(time1), max(time1)])
    if target == 'CITau':
        ax1.set_ylim([-300, 375])
    ax1.plot(time1, flux1, color='#80002a', lw=1.2, alpha=1)
    # ax1.plot(time2, flux2, color='#ff6600', lw=0.65)
    ax1.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True)
    #ax1.set_xticklabels([])

    # -------
    # ax2 = fig4.add_subplot(222)
    ax2.set_ylabel('Power', fontsize=font_size, style='normal', family='sans-serif')
    ax2.set_xlabel('Period (d)', fontsize=font_size, style='normal', family='sans-serif')
    mx = max(periods1)
    ax2.set_xlim([0, mx])
    ax2.set_ylim([0, 1])
    ax2.plot(periods1, powers1, color='#80002a', lw=1.2, alpha=1)
    # ax2.plot(periods2, powers2, color='#ff6600', lw=0.65) #ff8533
    # ax2.vlines(per_expected_lit, 0, 0.5, color='blue',lw=0.75, alpha=1, linestyles='--',label='Expected P from lit')
    # ax2.text(0.82, 0.865, target, horizontalalignment='center', fontsize='large', style='normal', family='sans-serif', transform=ax2.transAxes)
    peaks = peakutils.indexes(powers1, thres=0.10, min_dist=10)
    for i in peaks:
        ax2.vlines(periods1[peaks], 0, 1., color='#000000',lw=0.5, alpha=0.1, linestyles='--')
        #ax2.text(periods1[i] - 0.58, powers1[i] + 0.015, '%5.2f' % periods1[i], horizontalalignment='center', fontsize='x-small', style='normal', family='sans-serif')
        ax2.text(periods1[i], powers1[i] + 0.015, '%5.2f' % periods1[i], horizontalalignment='center', fontsize='x-small', style='normal', family='sans-serif')
    ax2.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True)
    # ax2.legend(loc='upper right')
    #ax2.set_xticks((0,5,10,15,20,25,30,35))
    #ax2.set_xscale('symlog')
    # ax2.set_xticklabels([])


    # -----------------------------
    # ax3 = fig4.add_subplot(223)
    # ax41.set_title(target + '   ' + epic, fontsize=20)
    # ax3.set_ylabel(r'Flux $-$ Model', fontsize=font_size, style='normal', family='sans-serif')
    # ax3.set_xlabel('Time (d)', fontsize=font_size, style='normal', family='sans-serif')
    # ax3.set_xlim([min(time1), max(time1)])
    # ax3.plot(time3, flux3, color='#000000',lw=0.5)
    # ax3.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True)
    #
    # # -------
    # # ax4 = fig4.add_subplot(224)
    # ax4.set_ylabel('Power', fontsize=font_size, style='normal', family='sans-serif')
    # ax4.set_xlabel('Period (d)', fontsize=font_size, style='normal', family='sans-serif')
    # mx = max(periods3)
    # ax4.set_xlim([0, mx])
    # ax4.set_ylim([0, 0.175])
    # ax4.plot(periods3, powers3, color='#000000', lw=0.5)
    # ax4.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True)
    # ax4.set_xticks((0, 5, 10, 15, 20, 25, 30, 35))
    # -----------------------------

    # tight_layout()
    fig4.savefig(savehere + target + '.pdf', format='pdf', bbox_inches='tight')
    # plt.show()
    plt.close()


def Gaussiansmooth(target, listx, listy, listy_err, savehere, degree, PerMaxTest):  # degree=48
    window = degree * 2 - 1
    weight = np.array([1.0] * window)
    weightGauss = []

    for i in range(window):
        i = i - degree + 1
        frac = i / float(window)
        gauss = 1 / (np.exp((4 * (frac)) ** 2))
        weightGauss.append(gauss)

    weight = np.array(weightGauss) * weight

    smoothed_y = [0.0] * (len(listy) - window)
    smoothed_x = [0.0] * (len(listx) - window)

    for thisy in range(len(smoothed_y)):
        smoothed_y[thisy] = np.sum(np.array(listy[thisy:thisy + window]) * weight) / np.sum(weight)
    for thisx in range(len(smoothed_x)):
        smoothed_x[thisx] = np.median(listx[thisx:thisx + window])

    wheresame = []
    for thistime in range(len(smoothed_x)):
        if len(np.where(listx == smoothed_x[thistime])[0]) > 0:
            wheresame.append(np.where(listx == smoothed_x[thistime])[0][0])

    # fig_test = plt.figure(num=None, figsize=(8, 7.5), facecolor='w')
    # ax_test = fig_test.add_subplot(211)
    # # ax_test.set_title('fit = %0.4f+/-%4f * x + %0.4f+/-%4f' % (slope, slope_err, yint, yint_err))
    # ax_test.set_ylabel('Flux', fontsize=20)
    # #ax_test.set_xlabel('Time (d)', fontsize=24)
    # #ax_test.set_xticks([])
    # ax_test.plot(listx[wheresame], listy[wheresame], color='black')
    # ax_test.plot(smoothed_x, smoothed_y, linewidth=1, color='red')
    # ax_test.tick_params(labelsize=16, size=6, width=2, pad=4)
    #
    # ax2 = fig_test.add_subplot(212)
    # ax2.set_ylabel('Original - Smoothed',fontsize=20)
    # ax2.set_xlabel('Time (d)', fontsize=20)
    # ax2.plot(listx[wheresame], listy[wheresame]-smoothed_y, color='black')
    # ax2.tick_params(labelsize=16, size=6, width=2, pad=4)
    #
    # fig_test.tight_layout()
    # fig_test.savefig(savehere + target + '_GaussianSmoothed.pdf')
    # #show()
    # plt.close()

    pers1, pows1 = lombscargle(listx[wheresame], listy[wheresame], listy_err[wheresame], Pmax=PerMaxTest, res=2000)
    pers2, pows2 = lombscargle(smoothed_x, smoothed_y, np.zeros_like(listx[wheresame]) + 1e-8, Pmax=PerMaxTest, res=2000)
    pers3, pows3 = lombscargle(smoothed_x, listy[wheresame] - smoothed_y, listy_err[wheresame], Pmax=PerMaxTest, res=2000)

    # plot_lcper3(target+'_model', smoothed_x, smoothed_y, pers2, pows2, savehere)

    # plot_lcper5(target, listx[wheresame], listy[wheresame], smoothed_x, smoothed_y, smoothed_x, listy[wheresame] - smoothed_y, pers1, pows1, pers2, pows2, pers3, pows3, savehere)
    plot_lcper6(target, listx[wheresame], listy[wheresame], smoothed_x, smoothed_y, smoothed_x, listy[wheresame] - smoothed_y, pers1, pows1, pers2, pows2, pers3, pows3, savehere)

    return smoothed_x, smoothed_y, listy[wheresame] - smoothed_y, listy_err[wheresame]


def variance(values, values_err):
    N = len(values)
    weights = 1.0 / (values_err ** 2)
    top = np.sum(weights * values)
    bottom = np.sum(weights)
    weighted_mean = top / bottom
    weighted_mean_err = 1.0 / np.sum(weights ** 2)

    weighted_mean = weighted_mean * 0
    weighted_mean_err = weighted_mean_err * 0

    var = np.sum((values - weighted_mean) ** 2) / N

    propogation_terms = []
    for n in range(len(values)):
        dvar_dxi = (2. * (values[n] - weighted_mean)) / N
        # dvar_dmean = (2.*(weighted_mean - values[n]))/N
        dvar_dmean = 2. * weighted_mean - 2. * np.sum(values)

        termi1 = dvar_dxi ** 2 * values[n] ** 2
        termi2 = dvar_dmean ** 2 * weighted_mean_err ** 2

        term = termi1 + termi2
        propogation_terms.append(term)

    var_err = np.sqrt(np.sum(propogation_terms))

    # import pdb; pdb.set_trace()

    return var, var_err


def binlc_slide2(xin, yin, yin_err, binwidth, slidestep):
    xin = xin - min(xin)

    x_binned = []
    y_binned = []
    yerr_binned = []
    bin_variance = []
    bin_variance_err = []

    # numbins = int((max(xin) - min(xin))/ binwidth)
    numbins = int((max(xin) - binwidth) / slidestep)

    k = 0
    slidemax = np.max(xin) + 0.5 * binwidth
    current_winmax = k * slidestep + 0.5 * binwidth

    # for k in range(0, numbins):
    while current_winmax <= slidemax:
        # inbin = np.where((xin >= k*slidestep) & (xin <= (k*slidestep) + binwidth))[0]
        slidemax = np.max(xin) + 0.5 * binwidth
        current_winmin = k * slidestep - 0.5 * binwidth
        current_winmax = k * slidestep + 0.5 * binwidth
        inbin = np.where((xin >= current_winmin) & (xin <= current_winmax))[0]

        # if k == 0:
        #     num_inbin = len(inbin)

        # window_variance, window_variance_err = variance(yin[inbin],yin_err[inbin])

        # if len(inbin) >= 0.90*num_inbin:
        if len(inbin) > 1:
            # window_variance = np.nanvar(yin[inbin]*(1./(yin_err[inbin])**2))
            window_variance = np.nanvar(yin[inbin])
            window_variance_err = window_variance * (1 / np.sqrt(len(inbin)))  # 1e-9

            xdiff = np.abs(xin[inbin] - 0.5 * (current_winmax - current_winmin))
            where_xmin = np.where(xdiff == np.nanmin(xdiff))[0]
            if len(where_xmin) > 1:
                where_xmin = where_xmin[0]
            if k > 0:
                if xin[inbin][where_xmin] in x_binned:
                    k += 1
                    continue

            bin_variance.append(window_variance)
            bin_variance_err.append(window_variance_err)
            x_meanbin = np.nanmean(xin[inbin][where_xmin])
            y_meanbin = np.nanmean(yin[inbin] * (1. / yin_err[inbin]))
            x_binned.append(x_meanbin)
            y_binned.append(y_meanbin)
            yerr_binned.append(np.nanstd(yin[inbin] * (1. / yin_err[inbin])) / np.sqrt(len(yin[inbin])))

        k += 1

    return x_binned, y_binned, yerr_binned, bin_variance, bin_variance_err


def plot_2D_LS_Period(Xin, Yin, Z, peaks, target, winds, savehere):
    font_size = 'small'
    font_style = 'normal'
    font_family = 'sans-serif'

    fig = plt.figure(num=None, figsize=(3, 2.85), facecolor='w', dpi=300)

    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(left=0.0, right=0.95, hspace=0.0)
    ax1 = plt.subplot(gs1[0, 0])

    gs2 = gridspec.GridSpec(1, 1)
    gs2.update(left=0.96, right=1.0, hspace=0.0)
    ax2 = plt.subplot(gs2[0, 0])

    # ax1 = fig.add_subplot(111)
    # ax_test.set_title('fit = %0.4f+/-%4f * x + %0.4f+/-%4f' % (slope, slope_err, yint, yint_err))
    ax1.set_ylabel('Window Size (d)', fontsize=font_size, style=font_style, family=font_family)
    ax1.set_xlabel('Period (d)', fontsize=font_size, style=font_style, family=font_family)
    ax1.set_ylim([min(Xin), max(Xin)])
    ax1.set_xlim([min(Yin), max(Yin)])
    # mx = max(Yin)
    # ax1.set_xticks((5,10,15,20,25,30,35))
    X, Y = np.meshgrid(Yin, Xin)
    p = ax1.pcolor(X, Y, Z, cmap=cm.inferno, edgecolors='face', vmin=0,
                   vmax=1.0)  # vmax=abs(Z.max())) #vmax=abs(Z.max()))
    # cbaxes = fig.add_axes([0.8,0.1,0.03,0.8])
    cb = fig.colorbar(p, cax=ax2, use_gridspec=True)  # , ticks=linspace(0,abs(Z).max(),10))
    cb.set_label(label='Power', fontsize=font_size, style=font_style, family=font_family)
    # cb.ax.set_yticklabels(np.arange(0,Z.max(),0.1),style=font_style, family=font_family)
    cb.ax.tick_params(labelsize=font_size)  # , style=font_style, family=font_family)

    # for foo, this_peak in enumerate(peaks):
    #     text_vshift = [0.25]
    #     text_hshift = [0.2]
    #     # text_colors = ['#8000ff', '#00cc00', '#ff9900']
    #     text_colors = ['#ffffff']
    #     line_styles = ['--']
    #     ax1.plot([this_peak, this_peak], [min(Xin), max(Xin)], linestyle=line_styles[foo], lw=0.5,
    #              color=text_colors[foo])
    #     ax1.text(this_peak + text_hshift[foo], max(Xin) - 1.25, '%5.3f' % this_peak, size='x-small',
    #              color=text_colors[foo], style=font_style, family=font_family)
    #     ax1.plot([min(Yin), max(Yin)], [winds[foo], winds[foo]], linestyle=line_styles[foo], lw=0.5,
    #              color=text_colors[foo])
    #     ax1.text(2.0, winds[foo] + text_vshift[foo], '%5.3f' % winds[foo], size='x-small', color=text_colors[foo],
    #              style=font_style, family=font_family)

    ax1.tick_params(axis='both', labelsize=font_size, direction='in', top=True, right=True, color='#ffffff')
    fig.savefig(savehere + target + '_2D-period.pdf', format='pdf', bbox_inches='tight')
    plt.close()


lightcurve_file = '/Users/lbiddle/Dropbox/Dissertation_Figures/lc_accretion_2030-2.csv'
save_here = '/Users/lbiddle/Desktop/Plots_For_Dissertation/Chapter2_Figures/'


targ_name = 'lc_2030-2'

do_2D = False

lightcurve = pd.read_csv(lightcurve_file)

lightcurve = lightcurve[lightcurve['time'] >= 0]

x = lightcurve['time'].values
y = lightcurve['flux'].values
yerr = lightcurve['err'].values


MaxTestPeriod = 20  # int((np.max(x) - np.min(x)) * 0.5)

smooth_degree = 15
minwinsize = 0.05
slide_stepsize = 0.025

smooth_degree = 3
# minwinsize = 0.05
# slide_stepsize = 0.025

resolution = 1000

time_smooth, flux_smooth, flux_residual, flux_residual_err = Gaussiansmooth(targ_name, x, y, yerr, save_here, degree=smooth_degree, PerMaxTest=MaxTestPeriod)  # degree=int(24*0.75)





if (do_2D == True):  # and (targ_name == single_targ_name):

    wherenan_residual = np.where(np.isnan(flux_residual) == False)[0]
    time_smooth = np.array(time_smooth)[wherenan_residual]
    flux_smooth = np.array(flux_smooth)[wherenan_residual]
    flux_residual = np.array(flux_residual)[wherenan_residual]
    flux_residual_err = np.array(flux_residual_err)[wherenan_residual]

    maxwinsize = MaxTestPeriod * (1. / 2.)
    winsizes = np.linspace(minwinsize, maxwinsize, 200)
    Z = np.zeros((len(winsizes), resolution))

    for thiswin in range(0, len(winsizes)):
        print(thiswin)
        winsize = winsizes[thiswin]
        # bin_time, bin_flux, bin_err, bin_variance = binlc_slide(time_smooth, flux_residual, winsize, slidestep=0.10)
        bin_time, bin_flux, bin_err, bin_variance, bin_variance_err = binlc_slide2(time_smooth,
                                                                                   flux_residual,
                                                                                   flux_residual_err,
                                                                                   winsize,
                                                                                   slidestep=slide_stepsize)
        # var_periods, var_powers = lombscargle(bin_time, bin_variance, np.zeros(len(bin_variance)) + 1e-10, Pmax=MaxTestPeriod, res=ls_resolution)
        var_periods, var_powers = lombscargle(bin_time, bin_variance, bin_variance_err, Pmax=MaxTestPeriod,
                                              res=resolution)

        Z[thiswin, :] = var_powers

    wherenan = np.where(np.isnan(Z) == True)
    Z[wherenan] = 0
    wheremax = np.where(Z == Z.max())
    bestwin = winsizes[wheremax[0][0]]
    peak_windows = [bestwin]
    powers_of_bestwin = Z[wheremax[0][0], :]

    peak_period = var_periods[wheremax[1][0]]
    peak_periods = [peak_period]

    # import pdb; pdb.set_trace()

    print('Generating 2D LS In Period Space...')
    plot_2D_LS_Period(winsizes, var_periods, Z, peak_periods, targ_name, peak_windows, save_here)

    # periodogram_results = xo.estimators.lomb_scargle_estimator(x, y, max_peaks=1, min_period=0.1, max_period=20.0, samples_per_peak=50)
    #
    # # peak = periodogram_results["peaks"][0]
    # freq, power = periodogram_results["periodogram"]
    # period = 1.0/freq

    # font_size = 'medium'
    #
    # # fig = plt.figure(1, figsize=(11, 5), facecolor="#ffffff")
    # fig = plt.figure(1, figsize=(14, 5), facecolor="#ffffff")
    # ax = fig.add_subplot(121)
    # ax.plot(x, y, color='#000000', lw=1.5)
    # ax.set_title(targ_name, fontsize=font_size, style='normal', family='sans-serif')
    # ax.set_xlabel('Time', fontsize=font_size, style='normal', family='sans-serif')
    # if name_of_mission == 'K2':
    #     ax.set_ylabel('Normalized K2 Flux', fontsize=font_size, style='normal', family='sans-serif')
    # if name_of_mission == 'TESS':
    #     ax.set_ylabel('Normalized TESS Flux', fontsize=font_size, style='normal', family='sans-serif')
    # ax.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
    #
    # ax2 = fig.add_subplot(122)
    # ax2.plot(pers, pows, color='#000000', lw=1.5)
    # #ax2.set_title(targ_name, fontsize=font_size, style='normal', family='sans-serif')
    # ax2.set_xlabel('Period (d)', fontsize=font_size, style='normal', family='sans-serif')
    # ax2.set_ylabel('Power', fontsize=font_size, style='normal', family='sans-serif')
    # ax2.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
    #
    #
    # plt.tight_layout()
    # if name_of_mission == 'K2':
    #     plt.savefig('K2_lightcurves/' + targ_name + '.pdf', dpi=300)
    # if name_of_mission == 'TESS':
    #     plt.savefig('TESS_lightcurves/' + targ_name + '.pdf', dpi=300)
    # plt.close()
    # # plt.show()


