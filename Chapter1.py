import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import lightkurve as lk
from astroML.time_series import lomb_scargle
import pandas as pd
from scipy.optimize import curve_fit


def get_fwhm(lightcurve, around_periods):

    fwhm_dict = {}

    for period_i, period in enumerate(around_periods):

        if period < 14:
            where_x = np.where((lightcurve['PERS'] >= around_periods[period_i]-1.0) & (lightcurve['PERS'] <= around_periods[period_i]+1.0))
        if period >= 14:
            where_x = np.where((lightcurve['PERS'] >= around_periods[period_i] - 2.0) & (lightcurve['PERS'] <= around_periods[period_i] + 2.0))

        x_in = lightcurve['PERS'][where_x]
        y_in = lightcurve['POWS'][where_x]
        y_max = max(y_in)
        try:
            x_max = x_in[np.where(y_in == y_max)][0]
        except:
            continue

        x_interp = np.linspace(np.min(x_in),np.max(x_in),1000)
        y_interp = np.interp(x_interp, x_in, y_in)

        half = np.max(y_in) / 2.0
        signs = np.sign(np.add(y_interp, -half))
        zero_crossings = (signs[0:-2] != signs[1:-1])
        where_zero_crossings = np.where(zero_crossings)[0]

        try:
            x1 = np.mean(x_interp[where_zero_crossings[0]:where_zero_crossings[0] + 1])
        except:
            continue
        try:
            x2 = np.mean(x_interp[where_zero_crossings[1]:where_zero_crossings[1] + 1])
        except:
            continue

        fwhm = x2 - x1
        hwhm = 0.5*fwhm

        fwhm_dict[str(x_max)] = ['FWHM = ' + str(np.round(fwhm,2)), 'HWHM = ' + str(np.round(hwhm,2))]

    return fwhm_dict
def lombscargle(x, y, yerr, Pmin=0.20, Pmax=20, res=2000):
    periods = np.linspace(Pmin, Pmax, res)
    ang_freqs = 2 * np.pi / periods
    powers = lomb_scargle(x, y, yerr, ang_freqs, generalized=True)
    return periods, powers
def phasefold(x, period):

    x_shifted = x - x[0]
    x_period_divided = x_shifted / float(period)
    x_phased = np.mod(x_period_divided, 1)

    return x_phased
def compute_binned(x, y, yerr, binwidth):

    x_binned = []
    y_binned = []
    yerr_binned = []

    bin_start = 0
    while bin_start < max(x):
        bin_end = bin_start + binwidth
        where_in_bin = np.where((x >= bin_start) & (x < bin_end))

        y_in_bin = y[where_in_bin]
        yerr_in_bin = yerr[where_in_bin]

        x_bin = bin_start + 0.5*binwidth

        weights = 1./(np.array(yerr_in_bin)**2)
        y_bin = (sum(y_in_bin*weights)) / sum(weights)

        yerr_bin = np.sqrt(sum((y_in_bin**2*yerr_in_bin**2)/len(yerr_in_bin)))

        x_binned.append(x_bin)
        y_binned.append(y_bin)
        yerr_binned.append(yerr_bin)

        bin_start += binwidth

    return x_binned, y_binned, yerr_binned
def my_sin(x, amplitude, phase, offset):
    return amplitude*np.sin(x*2*np.pi + phase) + offset
def fit_sinfunc(x, y, y_err):

    popt, pcov = curve_fit(my_sin, x, y, sigma=y_err) #, p0=p0)

    # recreate the fitted curve using the optimized parameters
    x_fit = np.linspace(0, 1.0, 10000)
    y_fit = my_sin(x_fit, *popt)
    popt_err = np.sqrt(np.diag(pcov))

    return x_fit, y_fit, popt, popt_err


def plot_lightcurves(input_data, save_as):

    vpad = 0.025
    hpad = 0.10
    left_subplot_locations = {'left': np.array([0, 0, 0, 0]) + hpad,
                              'bottom': np.array([0.75 - 4*vpad, 0.50 - 2*vpad, 0.25, 0.0 + 0.05]),  # + vpad,
                              'width': np.array([0.57, 0.57, 0.57, 0.57]) - hpad,
                              'height': np.array([0.2, 0.2, 0.2, 0.2]) - vpad,
                              }
    right_subplot_locations = {'left': np.array([0.55, 0.55, 0.55, 0.55]) + hpad,
                               'bottom': np.array([0.75 - 4*vpad, 0.50 - 2*vpad, 0.25, 0.0 + 0.05]),  # + vpad,
                               'width': np.array([0.37, 0.37, 0.37, 0.37]) - hpad,
                               'height': np.array([0.2, 0.2, 0.2, 0.2]) - vpad,
                               }
    font_size = 'medium'
    colors = ['#003366','#80002a','#4d0066','#cc3300']
    fig = plt.figure(1, figsize=(5*1.45, 9), facecolor="#ffffff")  # , dpi=300)

    for parameter_i, parameter in enumerate(input_data.keys()):
        ax1 = fig.add_axes((left_subplot_locations['left'][parameter_i], left_subplot_locations['bottom'][parameter_i],
                           left_subplot_locations['width'][parameter_i], left_subplot_locations['height'][parameter_i]))

        ax1.set_ylabel(parameter + ' Flux (ppt)', fontsize=font_size)
        if parameter_i == len(input_data.keys()) - 1:
            ax1.set_xlabel('Time (BJD - 2454833)', fontsize=font_size)
        if parameter_i != len(input_data.keys()) - 1:
            ax1.set_xticklabels([])
        xplot1 = np.array(input_data[parameter]['TIME'])
        yplot1 = np.array(input_data[parameter]['FLUX'])
        ax1.plot(xplot1, yplot1, color=colors[parameter_i], lw=1.0)
        ax1.set_xlim([min(xplot1), max(xplot1)])
        # maxshift = max([max(abs(yplot+np.array(std_diffs[parameter]))),max(abs(yplot-np.array(std_diffs[parameter])))])
        # ax1.set_ylim([0 - 1.15*maxshift, 0 + 1.15*maxshift])
        ax1.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)

        ax2 = fig.add_axes((right_subplot_locations['left'][parameter_i],
                            right_subplot_locations['bottom'][parameter_i],
                            right_subplot_locations['width'][parameter_i],
                            right_subplot_locations['height'][parameter_i]))

        ax2.set_ylabel('Power', fontsize=font_size)
        if parameter_i == len(input_data.keys()) - 1:
            ax2.set_xlabel('Period (d)', fontsize=font_size)
        if parameter_i != len(input_data.keys()) - 1:
            ax2.set_xticklabels([])

        xplot2 = np.array(input_data[parameter]['PERS'])
        yplot2 = np.array(input_data[parameter]['POWS'])
        ax2.plot(xplot2, yplot2, color=colors[parameter_i], lw=1.5)

        peaks = input_data[parameter]['PEAKS'].keys()
        for peak_i, peak in enumerate(peaks):
            if peak_i <= 1:
                ax2.vlines(float(peak), 0, 0.4, color='#000000', lw=0.5, alpha=1.0, linestyles='--')
            else:
                ax2.vlines(float(peak), 0, 0.4, color='#000000', lw=0.5, alpha=0.25, linestyles='--')
            where_peak = np.where(xplot2 == float(peak))
            if peak_i == 0:
                ax2.text(float(peak)-1.95, yplot2[where_peak]+0.011, '%5.2f' % float(peak),
                         horizontalalignment='center', fontsize='small', style='normal', family='sans-serif',
                         color=colors[parameter_i], weight='bold')
            if peak_i == 1:
                ax2.text(float(peak)+1.55, yplot2[where_peak]+0.011, '%5.2f' % float(peak),
                         horizontalalignment='center', fontsize='small', style='normal', family='sans-serif',
                         color=colors[parameter_i], weight='bold')

        ax2.set_xlim([0, 20])
        ax2.set_ylim([0, 0.4])
        ax2.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)



    plt.tight_layout()
    plt.savefig(save_as,dpi=300)
    plt.close()
    # plt.show()
def plot_halves(input_data, input_first_half_data, input_second_half_data, save_as):

    vpad = 0.025
    hpad = 0.16
    subplot_locations = {'left': np.array([0, 0, 0, 0]) + hpad,
                         'bottom': np.array([0.80 - 1.5*vpad, 0.55 - 1*vpad, 0.30-0.5*vpad, 0.0 + 0.05]),  # + vpad,
                         'width': np.array([0.97, 0.97, 0.97, 0.97]) - hpad,
                         'height': np.array([0.23, 0.23, 0.23, 0.23]) - vpad,
                         }
    font_size = 'medium'
    colors = ['#003366','#80002a','#4d0066','#cc3300']
    colors_first_half = ['#001a33', '#4d0019', '#260033', '#802000']
    colors_second_half = ['#0066cc', '#b3003b', '#730099', '#ff4000']
    fig = plt.figure(1, figsize=(3.75, 9), facecolor="#ffffff")  # , dpi=300)

    for parameter_i, parameter in enumerate(input_data.keys()):
        ax2 = fig.add_axes((subplot_locations['left'][parameter_i],
                            subplot_locations['bottom'][parameter_i],
                            subplot_locations['width'][parameter_i],
                            subplot_locations['height'][parameter_i]))

        ax2.set_title(parameter, fontsize=font_size, pad=3)
        ax2.set_ylabel('Power', fontsize=font_size)
        if parameter_i == len(input_data.keys()) - 1:
            ax2.set_xlabel('Period (d)', fontsize=font_size)
        if parameter_i != len(input_data.keys()) - 1:
            ax2.set_xticklabels([])

        xplot_full = np.array(input_data[parameter]['PERS'])
        yplot_full = np.array(input_data[parameter]['POWS'])
        ax2.plot(xplot_full, yplot_full, color=colors[parameter_i], linestyle='-',
                 lw=1.5,alpha=0.3,label='Full Lightcurve')

        xplot_first_half = np.array(input_first_half_data[parameter]['PERS'])
        yplot_first_half = np.array(input_first_half_data[parameter]['POWS'])
        ax2.plot(xplot_first_half, yplot_first_half, color=colors_first_half[parameter_i], linestyle='--',
                 lw=1.5, label='First Half')

        xplot_second_half = np.array(input_second_half_data[parameter]['PERS'])
        yplot_second_half = np.array(input_second_half_data[parameter]['POWS'])
        ax2.plot(xplot_second_half, yplot_second_half, color=colors_second_half[parameter_i], linestyle=':',
                 lw=1.5, label='Second Half')

        peaks = input_data[parameter]['PEAKS'].keys()
        for peak_i, peak in enumerate(peaks):
            if peak_i <= 1:
                ax2.vlines(float(peak), 0, 0.6, color='#000000', lw=0.5, alpha=1.0, linestyles='--')
            else:
                ax2.vlines(float(peak), 0, 0.6, color='#000000', lw=0.5, alpha=0.20, linestyles='--')
            # where_peak = np.where(xplot_full == float(peak))
            # if peak_i == 0:
            #     ax2.text(float(peak)-1.95, yplot2_full[where_peak]+0.011, '%5.2f' % float(peak),
            #              horizontalalignment='center', fontsize='small', style='normal', family='sans-serif',
            #              color=colors[parameter_i], weight='bold')
            # if peak_i == 1:
            #     ax2.text(float(peak)+1.55, yplot2_full[where_peak]+0.011, '%5.2f' % float(peak),
            #              horizontalalignment='center', fontsize='small', style='normal', family='sans-serif',
            #              color=colors[parameter_i], weight='bold')

        ax2.set_xlim([0, max(xplot_full)])
        ax2.set_ylim([0, 0.6])
        ax2.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)

        ax2.legend(loc='upper left', fontsize='x-small', framealpha=1.0, fancybox=False)

    plt.tight_layout()
    plt.savefig(save_as,dpi=300)
    plt.close()
    # plt.show()
def plot_phasefolded(input_data, period, save_as):

    vpad_base = 0.05
    hpad_base = 0.08
    vpad = 0.0075
    hpad = 0.05
    axis_height = 0.215
    axis_width = 0.42
    left_subplot_locations = {'left': np.array([hpad_base, hpad_base, hpad_base, hpad_base]),
                              'bottom': np.array([vpad_base+3*(axis_height+vpad), vpad_base+2*(axis_height+vpad), vpad_base+axis_height+vpad, vpad_base]),  # + vpad,
                              'width': np.array([axis_width, axis_width, axis_width, axis_width]),
                              'height': np.array([axis_height, axis_height, axis_height, axis_height]),
                              }
    right_subplot_locations = {'left': np.array([hpad_base+axis_width, hpad_base+axis_width, hpad_base+axis_width, hpad_base+axis_width]) + hpad,
                               'bottom': np.array([vpad_base+3*(axis_height+vpad), vpad_base+2*(axis_height+vpad), vpad_base+axis_height+vpad, vpad_base]),  # + vpad,
                               'width': np.array([axis_width, axis_width, axis_width, axis_width]),
                               'height': np.array([axis_height, axis_height, axis_height, axis_height]),
                               }
    font_size = 'medium'
    colors = ['#003366','#80002a','#4d0066','#cc3300']
    fig = plt.figure(1, figsize=(10, 9), facecolor="#ffffff")

    for parameter_i, parameter in enumerate(input_data.keys()):
        ax1 = fig.add_axes((left_subplot_locations['left'][parameter_i],
                            left_subplot_locations['bottom'][parameter_i],
                            left_subplot_locations['width'][parameter_i],
                            left_subplot_locations['height'][parameter_i]))
        ax2 = fig.add_axes((right_subplot_locations['left'][parameter_i],
                            right_subplot_locations['bottom'][parameter_i],
                            right_subplot_locations['width'][parameter_i],
                            right_subplot_locations['height'][parameter_i]))

        x_lightcurve = np.array(input_data[parameter]['TIME'])
        y_lightcurve = np.array(input_data[parameter]['FLUX'])
        yerr_lightcurve = np.array(input_data[parameter]['ERR'])

        peaks = []
        peak_keys = input_data[parameter]['PEAKS'].keys()
        for peak_key in enumerate(peak_keys):
            # import pdb; pdb.set_trace()
            peaks.append(float(peak_key[1]))

        which_period = np.where(abs(np.array(peaks) - period) == min(abs(np.array(peaks) - period)))[0][0]
        # import pdb; pdb.set_trace()
        peak = peaks[which_period]
        x_phased1 = phasefold(x_lightcurve, period=peak)

        ax1.scatter(x_phased1, y_lightcurve, color=colors[parameter_i], s=np.pi*(1.5)**2,
                    edgecolor='None', alpha=0.35, rasterized=True)
        ax2.scatter(x_phased1, y_lightcurve, color=colors[parameter_i], s=np.pi*(1.5)**2,
                    edgecolor='None', alpha=0.50, rasterized=True)

        x_binned1, y_binned1, yerr_binned1 = compute_binned(x_phased1, y_lightcurve, yerr_lightcurve, binwidth=0.1)

        ax1.scatter(x_binned1, y_binned1, color='#000000', s=np.pi*(2.5)**2, edgecolor='None')
        ax1.errorbar(x=x_binned1, y=y_binned1, yerr=yerr_binned1, ecolor='#000000', elinewidth=1.0,
                     capsize=2, capthick=1, linestyle='None')
        ax2.scatter(x_binned1, y_binned1, color='#000000', s=np.pi*(3.5)**2, edgecolor='None')
        ax2.errorbar(x=x_binned1, y=y_binned1, yerr=yerr_binned1, ecolor='#000000', elinewidth=2.0,
                     capsize=2, capthick=1, linestyle='None')


        x_fit, y_fit, y_fitpars, y_fitpars_err = fit_sinfunc(x=x_binned1, y=y_binned1, y_err=yerr_binned1)
        print(' ')
        print(parameter)
        print(y_fitpars)
        print(y_fitpars_err)

        ax1.plot(x_fit, y_fit, color='#000000', lw=1.5)
        ax2.plot(x_fit, y_fit, color='#000000', lw=2)

        ax1.set_xlim([0, 1])
        ax2.set_xlim([0, 1])
        ax1.set_ylim([-1.0*max(abs(np.array(y_lightcurve))), 1.0*max(abs(np.array(y_lightcurve)))])
        fitmax = np.max([max(abs(np.array(y_fit))),max(abs(np.array(y_binned1))+abs(np.array(yerr_binned1)))])
        ax2.set_ylim([-1.025*fitmax, 1.025*fitmax])

        ax1.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax2.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        if parameter_i == 0:
            ax1.set_title('Full', fontsize='large', pad=2)
            ax2.set_title('Zoomed', fontsize='large', pad=2)
        ax1.set_ylabel(parameter + ' Flux (ppt)', fontsize=font_size)
        if parameter_i == len(input_data.keys()) - 1:
            ax1.set_xlabel('Phase', fontsize=font_size)
            ax2.set_xlabel('Phase', fontsize=font_size)
            ax1.set_xticklabels([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax2.set_xticklabels([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        if parameter_i != len(input_data.keys()) - 1:
            ax1.set_xticklabels([])
            ax2.set_xticklabels([])
        ax1.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
        ax2.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)

    if which_period == 0:
        peak_text = '6.6'
    if which_period == 1:
        peak_text = '9.0'
    fig.suptitle(r'$\sim$' + peak_text + ' d Period Signature', fontsize='x-large')
    plt.tight_layout()
    plt.savefig(save_as,dpi=300,rasterized=True)
    plt.close()
    # plt.show()





MAST_filename = '/Users/lbiddle/Desktop/CITau_Lightcurves/CITau_MAST.fits'
K2SC_filename = '/Users/lbiddle/Desktop/CITau_Lightcurves/CITau_K2SC.fits'
EVEREST_filename = '/Users/lbiddle/Desktop/CITau_Lightcurves/CITau_EVEREST.fits'
K2SFF_filename = '/Users/lbiddle/Desktop/CITau_Lightcurves/CITau_K2SFF.fits'


lightcurve_dict = {}
periogogram_max = 30
#----------------------------------#
#-------------- MAST --------------#
#----------------------------------#

def get_MAST():
    # search_result_tpf = lk.search_targetpixelfile('CI Tau', mission='K2', campaign=13)
    # tpf = search_result_tpf.download()
    #
    # first_cadence = tpf[0]
    # first_cadence.plot(frame=0)
    #
    # import pdb; pdb.set_trace()

    search_result = lk.search_lightcurvefile('CI Tau', mission='K2', campaign=13)

    lcf = search_result[0].download(quality_bitmask='default')
    qual = np.isnan(lcf.PDCSAP_FLUX.flux) == False
    x = lcf.PDCSAP_FLUX.time
    y = lcf.PDCSAP_FLUX.flux
    yerr = lcf.PDCSAP_FLUX.flux_err

    x = np.ascontiguousarray(x[qual], dtype=np.float64)
    y = np.ascontiguousarray(y[qual], dtype=np.float64)
    yerr = np.ascontiguousarray(yerr[qual], dtype=np.float64)
    mu = np.mean(y)

    y = (y / mu - 1) * 1e3
    yerr = yerr / mu * 1e3

    wherenan_lightcurve = np.where(np.isnan(y) == False)[0]
    PDCSAP_time = x[wherenan_lightcurve]
    PDCSAP_flux = y[wherenan_lightcurve]
    PDCSAP_err = yerr[wherenan_lightcurve]

    return PDCSAP_time, PDCSAP_flux, PDCSAP_err


PDCSAP_time, PDCSAP_flux, PDCSAP_err = get_MAST()

PDCSAP_pers, PDCSAP_pows = lombscargle(PDCSAP_time, PDCSAP_flux, PDCSAP_err, Pmax=periogogram_max, res=2000)

PDCSAP_dict = {}
PDCSAP_dict['TIME'] = PDCSAP_time
PDCSAP_dict['FLUX'] = PDCSAP_flux
PDCSAP_dict['ERR'] = PDCSAP_err

PDCSAP_dict['PERS'] = PDCSAP_pers
PDCSAP_dict['POWS'] = PDCSAP_pows

PDCSAP_peaks = get_fwhm(lightcurve=PDCSAP_dict, around_periods=[6.6,9.0,11.5,14.3])
PDCSAP_peaks_df = pd.DataFrame.from_dict(PDCSAP_peaks)
PDCSAP_peaks_df.to_csv('/Users/lbiddle/Desktop/CITau_Lightcurves/PDCSAP_Peaks.csv', index=False)

PDCSAP_dict['PEAKS'] = PDCSAP_peaks

lightcurve_dict['PDCSAP'] = PDCSAP_dict


# ---------------------------------- #
# -------------- K2SC -------------- #
# ---------------------------------- #

def get_K2SC(filename):

    K2SC_fits = fits.open(filename)
    #K2SC_fits.info()

    K2SC_hdr = K2SC_fits[0].header

    K2SC_cols = K2SC_fits[1].columns
    #K2SC_cols.info()
    #K2SC_cols.names

    K2SC_data = K2SC_fits[1].data


    qual = np.isnan(K2SC_data['PDCSAP_FLUX']) == False
    x = K2SC_data['TIME']
    y = K2SC_data['PDCSAP_FLUX']
    yerr = K2SC_data['PDCSAP_FLUX_ERR']

    x = np.ascontiguousarray(x[qual], dtype=np.float64)
    y = np.ascontiguousarray(y[qual], dtype=np.float64)
    yerr = np.ascontiguousarray(yerr[qual], dtype=np.float64)
    mu = np.mean(y)

    y = (y / mu - 1) * 1e3
    yerr = yerr / mu * 1e3

    wherenan_lightcurve = np.where(np.isnan(y) == False)[0]
    K2SC_time = x[wherenan_lightcurve]
    K2SC_flux = y[wherenan_lightcurve]
    K2SC_err = yerr[wherenan_lightcurve]

    return K2SC_time, K2SC_flux, K2SC_err


K2SC_time, K2SC_flux, K2SC_err = get_K2SC(filename=K2SC_filename)

K2SC_pers, K2SC_pows = lombscargle(K2SC_time, K2SC_flux, K2SC_err, Pmax=periogogram_max, res=2000)

K2SC_dict = {}
K2SC_dict['TIME'] = K2SC_time
K2SC_dict['FLUX'] = K2SC_flux
K2SC_dict['ERR'] = K2SC_err

K2SC_dict['PERS'] = K2SC_pers
K2SC_dict['POWS'] = K2SC_pows

K2SC_peaks = get_fwhm(lightcurve=K2SC_dict, around_periods=[6.6,9.0,11.5,14.3])
K2SC_peaks_df = pd.DataFrame.from_dict(K2SC_peaks)
K2SC_peaks_df.to_csv('/Users/lbiddle/Desktop/CITau_Lightcurves/K2SC_Peaks.csv', index=False)

K2SC_dict['PEAKS'] = K2SC_peaks

lightcurve_dict['K2SC'] = K2SC_dict


# ---------------------------------- #
# ------------ EVEREST ------------- #
# ---------------------------------- #

def get_EVEREST(filename):
    EVEREST_fits = fits.open(filename)
    # EVEREST_fits.info()

    EVEREST_hdr = EVEREST_fits[0].header

    EVEREST_cols = EVEREST_fits[1].columns
    # EVEREST_cols.info()
    # EVEREST_cols.names

    EVEREST_data = EVEREST_fits[1].data

    qual = np.isnan(EVEREST_data['FCOR']) == False
    x = EVEREST_data['TIME']
    y = EVEREST_data['FCOR']
    yerr = EVEREST_data['FRAW_ERR']

    x = np.ascontiguousarray(x[qual], dtype=np.float64)
    y = np.ascontiguousarray(y[qual], dtype=np.float64)
    yerr = np.ascontiguousarray(yerr[qual], dtype=np.float64)

    mu = np.mean(y)

    # import pdb; pdb.set_trace()

    y = (y / mu - 1) * 1e3
    yerr = yerr / mu * 1e3

    wherenan_lightcurve = np.where(np.isnan(y) == False)[0]
    EVEREST_time = x[wherenan_lightcurve]
    EVEREST_flux = y[wherenan_lightcurve]
    EVEREST_err = yerr[wherenan_lightcurve]

    return EVEREST_time, EVEREST_flux, EVEREST_err


EVEREST_time, EVEREST_flux, EVEREST_err = get_EVEREST(filename=EVEREST_filename)

EVEREST_pers, EVEREST_pows = lombscargle(EVEREST_time, EVEREST_flux, EVEREST_err, Pmax=periogogram_max, res=2000)

EVEREST_dict = {}
EVEREST_dict['TIME'] = EVEREST_time
EVEREST_dict['FLUX'] = EVEREST_flux
EVEREST_dict['ERR'] = EVEREST_err

EVEREST_dict['PERS'] = EVEREST_pers
EVEREST_dict['POWS'] = EVEREST_pows

EVEREST_peaks = get_fwhm(lightcurve=EVEREST_dict, around_periods=[6.6,9.0,11.5,14.3])
EVEREST_peaks_df = pd.DataFrame.from_dict(EVEREST_peaks)
EVEREST_peaks_df.to_csv('/Users/lbiddle/Desktop/CITau_Lightcurves/EVEREST_Peaks.csv', index=False)

EVEREST_dict['PEAKS'] = EVEREST_peaks

lightcurve_dict['EVEREST'] = EVEREST_dict


# ---------------------------------- #
# -------------- K2SFF ------------- #
# ---------------------------------- #

def get_K2SFF(filename):
    K2SFF_fits = fits.open(filename)
    # K2SFF_fits.info()

    K2SFF_hdr = K2SFF_fits[0].header

    K2SFF_cols = K2SFF_fits[1].columns
    # K2SFF_cols.info()
    # K2SFF_cols.names

    K2SFF_data = K2SFF_fits[1].data

    qual = np.isnan(K2SFF_data['FCOR']) == False
    x = K2SFF_data['T']
    y = K2SFF_data['FCOR']

    x = np.ascontiguousarray(x[qual], dtype=np.float64)
    y = np.ascontiguousarray(y[qual], dtype=np.float64)

    mu = np.mean(y)

    # import pdb; pdb.set_trace()

    y = (y / mu - 1) * 1e3

    wherenan_lightcurve = np.where(np.isnan(y) == False)[0]
    K2SFF_time = x[wherenan_lightcurve]
    K2SFF_flux = y[wherenan_lightcurve]

    return K2SFF_time, K2SFF_flux


K2SFF_time, K2SFF_flux = get_K2SFF(filename=K2SFF_filename)
K2SFF_err = np.zeros(len(K2SFF_time)) + np.mean(PDCSAP_err)

K2SFF_pers, K2SFF_pows = lombscargle(K2SFF_time, K2SFF_flux, K2SFF_err, Pmax=periogogram_max, res=2000)

K2SFF_dict = {}
K2SFF_dict['TIME'] = K2SFF_time
K2SFF_dict['FLUX'] = K2SFF_flux
K2SFF_dict['ERR'] = K2SFF_err

K2SFF_dict['PERS'] = K2SFF_pers
K2SFF_dict['POWS'] = K2SFF_pows

K2SFF_peaks = get_fwhm(lightcurve=K2SFF_dict, around_periods=[6.6,9.0,11,14.3,17.5])
K2SFF_peaks_df = pd.DataFrame.from_dict(K2SFF_peaks)
K2SFF_peaks_df.to_csv('/Users/lbiddle/Desktop/CITau_Lightcurves/K2SFF_Peaks.csv', index=False)

K2SFF_dict['PEAKS'] = K2SFF_peaks

lightcurve_dict['K2SFF'] = K2SFF_dict


# ---------------------------------- #
# --------------- PLOT ------------- #
# ---------------------------------- #
save_figure_as = '/Users/lbiddle/Desktop/Plots_For_Dissertation/Chapter1_Figures/Lightcurves.pdf'
plot_lightcurves(input_data=lightcurve_dict, save_as=save_figure_as)

per1 = 6.6
save_figure_as = '/Users/lbiddle/Desktop/Plots_For_Dissertation/Chapter1_Figures/CITau_Phasefolded_' + str(np.floor(per1)) + '.pdf'
plot_phasefolded(input_data=lightcurve_dict, period=per1, save_as=save_figure_as)

per2 = 9.0
save_figure_as = '/Users/lbiddle/Desktop/Plots_For_Dissertation/Chapter1_Figures/CITau_Phasefolded_' + str(np.floor(per2)) + '.pdf'
plot_phasefolded(input_data=lightcurve_dict, period=per2, save_as=save_figure_as)


# -------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------ CALCULATE HALVES -------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
halftime = min(PDCSAP_time) + 0.5*(max(PDCSAP_time) - min(PDCSAP_time))

lightcurve_first_half_dict = {}
lightcurve_second_half_dict = {}

# ---------------------------------- #
# -------------- MAST -------------- #
# ---------------------------------- #
PDCSAP_where_first_half = np.where(PDCSAP_time <= halftime)[0]
PDCSAP_where_second_half = np.where(PDCSAP_time > halftime)[0]

PDCSAP_first_half_dict = {}
PDCSAP_first_half_dict['TIME'] = PDCSAP_time[PDCSAP_where_first_half]
PDCSAP_first_half_dict['FLUX'] = PDCSAP_flux[PDCSAP_where_first_half]
PDCSAP_first_half_dict['ERR'] = PDCSAP_err[PDCSAP_where_first_half]

PDCSAP_pers_first_half, PDCSAP_pows_first_half = lombscargle(PDCSAP_time[PDCSAP_where_first_half],
                                                             PDCSAP_flux[PDCSAP_where_first_half],
                                                             PDCSAP_err[PDCSAP_where_first_half],
                                                             Pmax=periogogram_max, res=2000)

PDCSAP_first_half_dict['PERS'] = PDCSAP_pers_first_half
PDCSAP_first_half_dict['POWS'] = PDCSAP_pows_first_half

PDCSAP_first_half_peaks = get_fwhm(lightcurve=PDCSAP_first_half_dict, around_periods=[6.6,9.0,11.5,14.3])
PDCSAP_first_half_peaks_df = pd.DataFrame.from_dict(PDCSAP_first_half_peaks)
PDCSAP_first_half_peaks_df.to_csv('/Users/lbiddle/Desktop/CITau_Lightcurves/PDCSAP_first_half_Peaks.csv', index=False)

PDCSAP_first_half_dict['PEAKS'] = PDCSAP_first_half_peaks

lightcurve_first_half_dict['PDCSAP'] = PDCSAP_first_half_dict

# ---------------------------------- #


PDCSAP_second_half_dict = {}
PDCSAP_second_half_dict['TIME'] = PDCSAP_time[PDCSAP_where_second_half]
PDCSAP_second_half_dict['FLUX'] = PDCSAP_flux[PDCSAP_where_second_half]
PDCSAP_second_half_dict['ERR'] = PDCSAP_err[PDCSAP_where_second_half]

PDCSAP_pers_second_half, PDCSAP_pows_second_half = lombscargle(PDCSAP_time[PDCSAP_where_second_half],
                                                             PDCSAP_flux[PDCSAP_where_second_half],
                                                             PDCSAP_err[PDCSAP_where_second_half],
                                                             Pmax=periogogram_max, res=2000)

PDCSAP_second_half_dict['PERS'] = PDCSAP_pers_second_half
PDCSAP_second_half_dict['POWS'] = PDCSAP_pows_second_half

PDCSAP_second_half_peaks = get_fwhm(lightcurve=PDCSAP_second_half_dict, around_periods=[6.6,9.0,11.5,14.3])
PDCSAP_second_half_peaks_df = pd.DataFrame.from_dict(PDCSAP_second_half_peaks)
PDCSAP_second_half_peaks_df.to_csv('/Users/lbiddle/Desktop/CITau_Lightcurves/PDCSAP_second_half_Peaks.csv', index=False)

PDCSAP_second_half_dict['PEAKS'] = PDCSAP_second_half_peaks

lightcurve_second_half_dict['PDCSAP'] = PDCSAP_second_half_dict


# ---------------------------------- #
# -------------- K2SC -------------- #
# ---------------------------------- #
K2SC_where_first_half = np.where(K2SC_time <= halftime)[0]
K2SC_where_second_half = np.where(K2SC_time > halftime)[0]

K2SC_first_half_dict = {}
K2SC_first_half_dict['TIME'] = K2SC_time[K2SC_where_first_half]
K2SC_first_half_dict['FLUX'] = K2SC_flux[K2SC_where_first_half]
K2SC_first_half_dict['ERR'] = K2SC_err[K2SC_where_first_half]

K2SC_pers_first_half, K2SC_pows_first_half = lombscargle(K2SC_time[K2SC_where_first_half],
                                                         K2SC_flux[K2SC_where_first_half],
                                                         K2SC_err[K2SC_where_first_half],
                                                         Pmax=periogogram_max, res=2000)

K2SC_first_half_dict['PERS'] = K2SC_pers_first_half
K2SC_first_half_dict['POWS'] = K2SC_pows_first_half

K2SC_first_half_peaks = get_fwhm(lightcurve=K2SC_first_half_dict, around_periods=[6.6,9.0,11.5,14.3])
K2SC_first_half_peaks_df = pd.DataFrame.from_dict(K2SC_first_half_peaks)
K2SC_first_half_peaks_df.to_csv('/Users/lbiddle/Desktop/CITau_Lightcurves/K2SC_first_half_Peaks.csv', index=False)

K2SC_first_half_dict['PEAKS'] = K2SC_first_half_peaks

lightcurve_first_half_dict['K2SC'] = K2SC_first_half_dict

# ---------------------------------- #


K2SC_second_half_dict = {}
K2SC_second_half_dict['TIME'] = K2SC_time[K2SC_where_second_half]
K2SC_second_half_dict['FLUX'] = K2SC_flux[K2SC_where_second_half]
K2SC_second_half_dict['ERR'] = K2SC_err[K2SC_where_second_half]

K2SC_pers_second_half, K2SC_pows_second_half = lombscargle(K2SC_time[K2SC_where_second_half],
                                                           K2SC_flux[K2SC_where_second_half],
                                                           K2SC_err[K2SC_where_second_half],
                                                           Pmax=periogogram_max, res=2000)

K2SC_second_half_dict['PERS'] = K2SC_pers_second_half
K2SC_second_half_dict['POWS'] = K2SC_pows_second_half

K2SC_second_half_peaks = get_fwhm(lightcurve=K2SC_second_half_dict, around_periods=[6.6,9.0,11.5,14.3])
K2SC_second_half_peaks_df = pd.DataFrame.from_dict(K2SC_second_half_peaks)
K2SC_second_half_peaks_df.to_csv('/Users/lbiddle/Desktop/CITau_Lightcurves/K2SC_second_half_Peaks.csv', index=False)

K2SC_second_half_dict['PEAKS'] = K2SC_second_half_peaks

lightcurve_second_half_dict['K2SC'] = K2SC_second_half_dict


# ---------------------------------- #
# ------------- EVEREST ------------ #
# ---------------------------------- #
EVEREST_where_first_half = np.where(EVEREST_time <= halftime)[0]
EVEREST_where_second_half = np.where(EVEREST_time > halftime)[0]

EVEREST_first_half_dict = {}
EVEREST_first_half_dict['TIME'] = EVEREST_time[EVEREST_where_first_half]
EVEREST_first_half_dict['FLUX'] = EVEREST_flux[EVEREST_where_first_half]
EVEREST_first_half_dict['ERR'] = EVEREST_err[EVEREST_where_first_half]

EVEREST_pers_first_half, EVEREST_pows_first_half = lombscargle(EVEREST_time[EVEREST_where_first_half],
                                                               EVEREST_flux[EVEREST_where_first_half],
                                                               EVEREST_err[EVEREST_where_first_half],
                                                               Pmax=periogogram_max, res=2000)

EVEREST_first_half_dict['PERS'] = EVEREST_pers_first_half
EVEREST_first_half_dict['POWS'] = EVEREST_pows_first_half

EVEREST_first_half_peaks = get_fwhm(lightcurve=EVEREST_first_half_dict, around_periods=[6.6,9.0,11.5,14.3])
EVEREST_first_half_peaks_df = pd.DataFrame.from_dict(EVEREST_first_half_peaks)
EVEREST_first_half_peaks_df.to_csv('/Users/lbiddle/Desktop/CITau_Lightcurves/EVEREST_first_half_Peaks.csv', index=False)

EVEREST_first_half_dict['PEAKS'] = EVEREST_first_half_peaks

lightcurve_first_half_dict['EVEREST'] = EVEREST_first_half_dict

# ---------------------------------- #


EVEREST_second_half_dict = {}
EVEREST_second_half_dict['TIME'] = EVEREST_time[EVEREST_where_second_half]
EVEREST_second_half_dict['FLUX'] = EVEREST_flux[EVEREST_where_second_half]
EVEREST_second_half_dict['ERR'] = EVEREST_err[EVEREST_where_second_half]

EVEREST_pers_second_half, EVEREST_pows_second_half = lombscargle(EVEREST_time[EVEREST_where_second_half],
                                                                 EVEREST_flux[EVEREST_where_second_half],
                                                                 EVEREST_err[EVEREST_where_second_half],
                                                                 Pmax=periogogram_max, res=2000)

EVEREST_second_half_dict['PERS'] = EVEREST_pers_second_half
EVEREST_second_half_dict['POWS'] = EVEREST_pows_second_half

EVEREST_second_half_peaks = get_fwhm(lightcurve=EVEREST_second_half_dict, around_periods=[6.6,9.0,11.5,14.3])
EVEREST_second_half_peaks_df = pd.DataFrame.from_dict(EVEREST_second_half_peaks)
EVEREST_second_half_peaks_df.to_csv('/Users/lbiddle/Desktop/CITau_Lightcurves/EVEREST_second_half_Peaks.csv', index=False)

EVEREST_second_half_dict['PEAKS'] = EVEREST_second_half_peaks

lightcurve_second_half_dict['EVEREST'] = EVEREST_second_half_dict


# ---------------------------------- #
# -------------- K2SFF ------------- #
# ---------------------------------- #
K2SFF_where_first_half = np.where(K2SFF_time <= halftime)[0]
K2SFF_where_second_half = np.where(K2SFF_time > halftime)[0]

K2SFF_first_half_dict = {}
K2SFF_first_half_dict['TIME'] = K2SFF_time[K2SFF_where_first_half]
K2SFF_first_half_dict['FLUX'] = K2SFF_flux[K2SFF_where_first_half]
K2SFF_first_half_dict['ERR'] = K2SFF_err[K2SFF_where_first_half]

K2SFF_pers_first_half, K2SFF_pows_first_half = lombscargle(K2SFF_time[K2SFF_where_first_half],
                                                           K2SFF_flux[K2SFF_where_first_half],
                                                           K2SFF_err[K2SFF_where_first_half],
                                                           Pmax=periogogram_max, res=2000)

K2SFF_first_half_dict['PERS'] = K2SFF_pers_first_half
K2SFF_first_half_dict['POWS'] = K2SFF_pows_first_half

K2SFF_first_half_peaks = get_fwhm(lightcurve=K2SFF_first_half_dict, around_periods=[6.6,9.0,11.5,14.3])
K2SFF_first_half_peaks_df = pd.DataFrame.from_dict(K2SFF_first_half_peaks)
K2SFF_first_half_peaks_df.to_csv('/Users/lbiddle/Desktop/CITau_Lightcurves/K2SFF_first_half_Peaks.csv', index=False)

K2SFF_first_half_dict['PEAKS'] = K2SFF_first_half_peaks

lightcurve_first_half_dict['K2SFF'] = K2SFF_first_half_dict

# ---------------------------------- #


K2SFF_second_half_dict = {}
K2SFF_second_half_dict['TIME'] = K2SFF_time[K2SFF_where_second_half]
K2SFF_second_half_dict['FLUX'] = K2SFF_flux[K2SFF_where_second_half]
K2SFF_second_half_dict['ERR'] = K2SFF_err[K2SFF_where_second_half]

K2SFF_pers_second_half, K2SFF_pows_second_half = lombscargle(K2SFF_time[K2SFF_where_second_half],
                                                             K2SFF_flux[K2SFF_where_second_half],
                                                             K2SFF_err[K2SFF_where_second_half],
                                                             Pmax=periogogram_max, res=2000)

K2SFF_second_half_dict['PERS'] = K2SFF_pers_second_half
K2SFF_second_half_dict['POWS'] = K2SFF_pows_second_half

K2SFF_second_half_peaks = get_fwhm(lightcurve=K2SFF_second_half_dict, around_periods=[6.6,9.0,11.5,14.3])
K2SFF_second_half_peaks_df = pd.DataFrame.from_dict(K2SFF_second_half_peaks)
K2SFF_second_half_peaks_df.to_csv('/Users/lbiddle/Desktop/CITau_Lightcurves/K2SFF_second_half_Peaks.csv', index=False)

K2SFF_second_half_dict['PEAKS'] = K2SFF_second_half_peaks

lightcurve_second_half_dict['K2SFF'] = K2SFF_second_half_dict


# ---------------------------------- #
# ------------ PLOT HALVES --------- #
# ---------------------------------- #
save_figure_as = '/Users/lbiddle/Desktop/Plots_For_Dissertation/Chapter1_Figures/Half_Lightcurve_Analysis.pdf'
plot_halves(input_data=lightcurve_dict, input_first_half_data=lightcurve_first_half_dict,
            input_second_half_data=lightcurve_second_half_dict, save_as=save_figure_as)


# plt.plot(PDCSAP_first_half_dict['TIME'],PDCSAP_first_half_dict['FLUX'])
# plt.plot(PDCSAP_second_half_dict['TIME'],PDCSAP_second_half_dict['FLUX'])
# plt.show()








