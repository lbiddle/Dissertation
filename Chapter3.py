import numpy as np
import matplotlib.pyplot as plt
import copy
import astropy.units as u



class Flares:
    def __init__(self, template, cadence, tpeak, fwhm, ampl, downsample, GA):
        self.template = template
        self.cadence = cadence
        self.tpeak = tpeak
        self.fwhm = fwhm
        self.ampl = ampl
        self.downsample = downsample
        self.GA = GA

    def get_fwhm(self, x_in, y_in):

        x_interp = np.linspace(np.min(x_in), np.max(x_in), 100000)
        y_interp = np.interp(x_interp, x_in, y_in)

        half = np.max(y_in) / 2.0
        signs = np.sign(np.add(y_interp, -half))
        zero_crossings = (signs[0:-2] != signs[1:-1])
        where_zero_crossings = np.where(zero_crossings)[0]

        try:
            x1 = np.mean(x_interp[where_zero_crossings[0]:where_zero_crossings[0] + 1])
        except:
            print('issue with fwhm determination interp get_fwhm: x1')
            # plt.plot(x_in, y_in)
            # plt.show()
            # import pdb; pdb.set_trace()

        try:
            x2 = np.mean(x_interp[where_zero_crossings[1]:where_zero_crossings[1] + 1])
        except:
            print('issue with fwhm determination interp get_fwhm: x2')
            # plt.plot(x_in, y_in)
            # plt.show()
            # import pdb; pdb.set_trace()

        return x2 - x1
    def get_fwhm2(self, x_in, y_in, y_err, do_interp=True):
        if do_interp == True:
            x_interp = np.linspace(np.min(x_in), np.max(x_in), 100000)
            y_interp = np.interp(x_interp, x_in, y_in)

            half = np.max(y_in) / 2.0
            signs = np.sign(np.add(y_interp, -half))
            zero_crossings = (signs[0:-2] != signs[1:-1])
            where_zero_crossings = np.where(zero_crossings)[0]

            x1 = np.mean(x_interp[where_zero_crossings[0]:where_zero_crossings[0] + 1])
            x2 = np.mean(x_interp[where_zero_crossings[1]:where_zero_crossings[1] + 1])

            xdiff = x2 - x1

            average_uncertainty = np.mean(y_err)

            upper = half + average_uncertainty
            signs_upper = np.sign(np.add(y_interp, -upper))
            zero_crossings_upper = (signs_upper[0:-2] != signs_upper[1:-1])
            where_zero_crossings_upper = np.where(zero_crossings_upper)[0]

            x1_upper = np.mean(x_interp[where_zero_crossings_upper[0]:where_zero_crossings_upper[0] + 1])
            x2_upper = np.mean(x_interp[where_zero_crossings_upper[1]:where_zero_crossings_upper[1] + 1])

            xdiff_upper = x2_upper - x1_upper

            lower = half - average_uncertainty
            signs_lower = np.sign(np.add(y_interp, -lower))
            zero_crossings_lower = (signs_lower[0:-2] != signs_lower[1:-1])
            where_zero_crossings_lower = np.where(zero_crossings_lower)[0]

            x1_lower = np.mean(x_interp[where_zero_crossings_lower[0]:where_zero_crossings_lower[0] + 1])
            x2_lower = np.mean(x_interp[where_zero_crossings_lower[1]:where_zero_crossings_lower[1] + 1])

            xdiff_lower = x2_lower - x1_lower

            xdiff_plus = xdiff_lower - xdiff
            xdiff_minus = xdiff - xdiff_upper

            # print(xdiff*24*60, xdiff_plus*24*60, xdiff_minus*24*60)

            # import pdb; pdb.set_trace()


        else:
            half = np.max(y_in) / 2.0
            signs = np.sign(np.add(y_in, -half))
            zero_crossings = (signs[0:-2] != signs[1:-1])
            where_zero_crossings = np.where(zero_crossings)[0]

            x1 = np.mean(x_in[where_zero_crossings[0]:where_zero_crossings[0] + 1])
            x2 = np.mean(x_in[where_zero_crossings[1]:where_zero_crossings[1] + 1])

            xdiff = x2 - x1

            average_uncertainty = np.mean(y_err)

            upper = half + average_uncertainty
            signs_upper = np.sign(np.add(y_in, -upper))
            zero_crossings_upper = (signs_upper[0:-2] != signs_upper[1:-1])
            where_zero_crossings_upper = np.where(zero_crossings_upper)[0]

            x1_upper = np.mean(x_in[where_zero_crossings_upper[0]:where_zero_crossings_upper[0] + 1])
            x2_upper = np.mean(x_in[where_zero_crossings_upper[1]:where_zero_crossings_upper[1] + 1])

            xdiff_upper = x2_upper - x1_upper

            lower = half - average_uncertainty
            signs_lower = np.sign(np.add(y_in, -lower))
            zero_crossings_lower = (signs_lower[0:-2] != signs_lower[1:-1])
            where_zero_crossings_lower = np.where(zero_crossings_lower)[0]

            x1_lower = np.mean(x_in[where_zero_crossings_lower[0]:where_zero_crossings_lower[0] + 1])
            x2_lower = np.mean(x_in[where_zero_crossings_lower[1]:where_zero_crossings_lower[1] + 1])

            xdiff_lower = x2_lower - x1_lower

            xdiff_plus = xdiff_lower - xdiff
            xdiff_minus = xdiff - xdiff_upper

        return xdiff, xdiff_plus, xdiff_minus
    def aflare1(self, t, tpeak_val, fwhm_val, ampl_val):
        '''
        The Analytic Flare Model evaluated for a single-peak (classical).
        Reference Davenport et al. (2014) http://arxiv.org/abs/1411.3723
        Use this function for fitting classical flares with most curve_fit
        tools.
        Note: this model assumes the flux before the flare is zero centered
        Parameters
        ----------
        t : 1-d array
            The time array to evaluate the flare over
        tpeak : float
            The time of the flare peak
        fwhm : float
            The "Full Width at Half Maximum", timescale of the flare
        ampl : float
            The amplitude of the flare
        Returns
        -------
        flare : 1-d array
            The flux of the flare model evaluated at each time
        '''
        _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
        _fd = [0.689008, -1.60053, 0.302963, -0.278318]

        flare = np.piecewise(t, [(t <= tpeak_val) * (t - tpeak_val) / fwhm_val > -1.,
                                 (t > tpeak_val)],
                             [lambda x: (_fr[0] +  # 0th order
                                         _fr[1] * ((x - tpeak_val) / fwhm_val) +  # 1st order
                                         _fr[2] * ((x - tpeak_val) / fwhm_val) ** 2. +  # 2nd order
                                         _fr[3] * ((x - tpeak_val) / fwhm_val) ** 3. +  # 3rd order
                                         _fr[4] * ((x - tpeak_val) / fwhm_val) ** 4.),  # 4th order
                              lambda x: (_fd[0] * np.exp(((x - tpeak_val) / fwhm_val) * _fd[1]) +
                                         _fd[2] * np.exp(((x - tpeak_val) / fwhm_val) * _fd[3]))]
                             ) * np.abs(ampl_val)  # amplitude

        return flare
    def jflare1(self, x, tpeak_val, j_fwhm, dav_fwhm, gauss_ampl, decay_ampl):
        _fd = [0.689008, -1.60053, 0.302963, -0.278318]
        g_profile = np.abs(gauss_ampl) * np.exp(-((x - tpeak_val) ** 2) / (j_fwhm ** 2))

        # decay_start = tpeak - tdiff
        decay_start = x[np.min(np.where(np.abs(g_profile - decay_ampl) == np.min(np.abs(g_profile - decay_ampl)))[0])]
        if decay_start > tpeak_val:
            decay_start = tpeak_val - (decay_start - tpeak_val)

        d_profile = (_fd[0] * np.exp(((x - decay_start) / dav_fwhm) * _fd[1]) +
                     _fd[2] * np.exp(((x - decay_start) / dav_fwhm) * _fd[3])) * np.abs(decay_ampl)

        # set padding
        g_profile[x < 0.1] = 0
        g_profile[x > 0.9] = 0
        d_profile[x > 0.9] = 0
        d_profile[x < decay_start] = 0

        c_profile = np.convolve(g_profile, d_profile, 'same')

        # print(x[np.where(g_profile == np.max(g_profile))[0]])
        # print(x[np.where(c_profile == np.max(c_profile))[0]])

        return g_profile, d_profile, c_profile, decay_start
    def create_single_synthetic(self, tpeak_val, fwhm_val, ampl_val):

        fwhm_val = fwhm_val * (1. / 60.) * (1. / 24.)  # minutes converted to days

        fine_sampling = 1. / 24. / 60. / 60.  # 1 second converted to days
        x_synth = np.arange(0, 3, fine_sampling)  # cadence in days
        y_synth = self.aflare1(x_synth, tpeak_val, fwhm_val, ampl_val)

        # wheremax = np.where(y_synth == np.max(y_synth))[0][0]
        # import pdb; pdb.set_trace()
        exptime = self.cadence
        exptime_sec = exptime * 60.
        synth_flare_scatter = []
        synth_flare_err = []
        for point_i in range(len(y_synth)):
            Sig = y_synth[point_i] + 1.
            SNR_sig = (Sig * np.sqrt(exptime_sec)) / np.sqrt(Sig)
            err_sig = (1. / SNR_sig) / 10
            Sig_scatter = np.random.normal(Sig, err_sig, 1)[0]
            synth_flare_scatter.append(Sig_scatter - 1.)
            synth_flare_err.append(err_sig)
        # print(' ')
        # # print('err:' + str(synth_flare_err))
        # print('min err: ' + str(np.min(synth_flare_err)))
        # print('max err: ' + str(np.max(synth_flare_err)))
        # # print('errs around max: ' + str(synth_flare_err[wheremax - 10:wheremax + 10]))
        # print(' ')

        synth_flare_scatter = np.array(synth_flare_scatter)
        synth_flare_err = np.array(synth_flare_err)

        # import pdb; pdb.set_trace()

        # return x_synth, synth_flare_scatter, synth_flare_err, y_synth, flare_properties
        return x_synth, synth_flare_scatter, synth_flare_err, y_synth
    def create_single_synthetic_jflare1(self, tpeak_val, fwhm_val, ampl_val):
        # np.random.seed()

        fine_sampling = 1. / 24. / 60. / 60.  # 1 second converted to days

        flag = 0
        while flag == 0:
            window = 1.  # days
            x_synth = np.arange(0, window, fine_sampling)  # cadence in days

            gauss_tpeak = tpeak_val  # peak time
            # gauss_tpeak = gauss_tpeak + np.random.uniform(-0.05, 0.05, 1)[0]

            gauss_ampl_j = 1.0
            decay_ampl_j = 0.68 * gauss_ampl_j

            # decay_fwhm_j = 0.80 * fwhm_val * (1. / 60.) * (1. / 24.)  # minutes to days
            # gauss_fwhm_j = 0.26 * decay_fwhm_j

            gauss_fwhm_j = 0.32 * fwhm_val * (1. / 60.) * (1. / 24.)  # minutes to days
            decay_fwhm_j = 1.75 * gauss_fwhm_j

            # compute the convolution
            g_profile, d_profile, synth_flare, decay_start = self.jflare1(x_synth, gauss_tpeak, gauss_fwhm_j,
                                                                          decay_fwhm_j, gauss_ampl_j, decay_ampl_j)

            # normalize the flare
            synth_flare /= np.max(synth_flare)


            # set flare amplitude after normalization
            synth_flare *= ampl_val

            exptime = self.cadence
            exptime_sec = exptime * 60.
            synth_flare_scatter = []
            synth_flare_err = []
            for point_i in range(len(synth_flare)):
                Sig = synth_flare[point_i] + 1.
                SNR_sig = (Sig * np.sqrt(exptime_sec)) / np.sqrt(Sig)
                err_sig = (1. / SNR_sig) / 10
                Sig_scatter = np.random.normal(Sig, err_sig, 1)[0]
                synth_flare_scatter.append(Sig_scatter - 1.)
                synth_flare_err.append(err_sig)
            # print(' ')
            # #print('err:' + str(synth_flare_err))
            # print('max err: ' + str(np.max(synth_flare_err)))
            # print('min err: ' + str(np.min(synth_flare_err)))
            # # print('errs around max: ' + str(synth_flare_err[wheremax - 10:wheremax + 10]))
            # print(' ')

            synth_flare_scatter = np.array(synth_flare_scatter)
            synth_flare_err = np.array(synth_flare_err)

            # determine flare fwhm
            # fwhm = get_fwhm(x_synth, synth_flare, do_interp=False)
            # try:
            #     fwhm_calc, fwhm_plus, fwhm_minus = self.get_fwhm2(x_synth, synth_flare, synth_flare_err, do_interp=True)
            try:
                fwhm_calc = self.get_fwhm(x_synth, synth_flare)
            except:
                flag = 0
                # plt.plot(x_synth, synth_flare)
                # plt.show()
                # import pdb; pdb.set_trace()
            else:
                flag = 1

        if np.isscalar(fwhm_calc) == False:
            fwhm_convolved = fwhm_calc[0]*24*60 # days converted to minutes
        else:
            fwhm_convolved = fwhm_calc*24*60 # days converted to minutes

        tpeak_convolved = x_synth[np.where(synth_flare == max(synth_flare))[0][0]]

        return x_synth, synth_flare_scatter, synth_flare_err, synth_flare, fwhm_convolved, tpeak_convolved

    def generate_flare(self):
        if self.template == 'Davenport':
            self.x_synth, y_synth_scatter, y_synth_err, self.y_synth_noscatter = self.create_single_synthetic(
                tpeak_val=self.tpeak, fwhm_val=self.fwhm, ampl_val=self.ampl)
        if self.template == 'Jackman':
            self.x_synth, y_synth_scatter, y_synth_err, self.y_synth_noscatter, fwhm_conv, tpeak_conv = self.create_single_synthetic_jflare1(
                tpeak_val=self.tpeak, fwhm_val=self.fwhm, ampl_val=self.ampl)

            self.fwhm_conv = fwhm_conv
            self.tpeak_conv = tpeak_conv

        if self.downsample == True:
            cadence_bench = (self.cadence) * 60  # to put in terms of seconds because finest sampling done with 1 sec cadence

            if self.GA == True:
                where_start = int(np.floor(np.random.uniform(0, cadence_bench + 1, 1)))
            else:
                where_start = 0
            self.where_start = where_start

            self.x_flare = self.x_synth[where_start::int(cadence_bench)]
            self.y_flare_scatter = y_synth_scatter[where_start::int(cadence_bench)]
            self.y_flare_err = y_synth_err[where_start::int(cadence_bench)]
            self.y_flare_noscatter = self.y_synth_noscatter[where_start::int(cadence_bench)]
        if self.downsample == False:
            self.x_flare = copy(self.x_synth)  # [0::1]
            self.y_flare_scatter = y_synth_scatter  # [0::1]
            self.y_flare_err = y_synth_err  # [0::1]
            self.y_flare_noscatter = copy(self.y_synth_noscatter)  # [0::1]

        eqdur_noscatter = np.trapz(self.y_flare_noscatter, x=self.x_flare)
        self.eqdur_noscatter = eqdur_noscatter * (24 * 60 * 60)  # convert days to seconds

        eqdur_scatter = np.trapz(self.y_flare_scatter, x=self.x_flare)
        self.eqdur_scatter = eqdur_scatter * (24 * 60 * 60)  # convert days to seconds
    def generate_candidate(self, guess_tpeak, guess_fwhm, guess_ampl):
        if self.template == 'Davenport':
            x_synth_candidate, y_synth_scatter_candidate, y_synth_err_candidate, y_synth_noscatter_candidate = self.create_single_synthetic(
                tpeak_val=guess_tpeak, fwhm_val=guess_fwhm, ampl_val=guess_ampl)
        if self.template == 'Jackman':
            x_synth_candidate, y_synth_scatter_candidate, y_synth_err_candidate, y_synth_noscatter_candidate,\
            fwhm_conv_candidate, tpeak_conv_candidate = self.create_single_synthetic_jflare1(
                tpeak_val=guess_tpeak, fwhm_val=guess_fwhm, ampl_val=guess_ampl)

            self.fwhm_conv_candidate = fwhm_conv_candidate
            self.tpeak_conv_candidate = tpeak_conv_candidate
            self.x_synth_candidate = x_synth_candidate
            self.y_synth_noscatter_candidate = y_synth_noscatter_candidate


        cadence_bench = (self.cadence) * 60  # to put in terms of seconds because finest sampling done with 1 sec cadence

        self.x_flare_candidate = x_synth_candidate[self.where_start::int(cadence_bench)]
        self.y_flare_scatter_candidate = y_synth_scatter_candidate[self.where_start::int(cadence_bench)]
        self.y_flare_err_candidate = y_synth_err_candidate[self.where_start::int(cadence_bench)]
        self.y_flare_noscatter_candidate = y_synth_noscatter_candidate[self.where_start::int(cadence_bench)]


        eqdur_noscatter_candidate = np.trapz(self.y_flare_noscatter_candidate, x=self.x_flare_candidate)
        self.eqdur_noscatter_candidate = eqdur_noscatter_candidate * (24 * 60 * 60)  # convert days to seconds

        eqdur_scatter_candidate = np.trapz(self.y_flare_scatter_candidate, x=self.x_flare_candidate)
        self.eqdur_scatter_candidate = eqdur_scatter_candidate * (24 * 60 * 60)  # convert days to seconds

    def quick_test_plot(self, any_x, any_y, label_x, label_y, plot_type_y, plot_alpha_y, y_axis_label, x_axis_range, y_axis_range, plot_title, save_as):
        font_size = 'large'

        # mycolormap = choose_cmap()
        # colors = mycolormap(range(len(any_x)))
        # colors = ['#b30047', '#ff3300', '#00cc99', '#3366ff']
        if len(any_x) > 1:
           #  evenly_spaced_interval = np.linspace(0, 1, len(any_x))
            # colors = [cm.rainbow(smoosh) for smoosh in evenly_spaced_interval]
            # colors = [cm.Spectral(smoosh) for smoosh in evenly_spaced_interval]
            colors = ['#005580', '#ff9900', '#cc0052']
            # colors = [mycolormap(smoosh) for smoosh in evenly_spaced_interval]
        if len(any_x) == 1:
            colors = ['#000000']

        # import pdb; pdb.set_trace()

        fig = plt.figure(1, figsize=(6, 5.5), facecolor="#ffffff")  # , dpi=300)
        ax = fig.add_subplot(111)
        ax.set_title(plot_title, fontsize=font_size, style='normal', family='sans-serif')

        for v in range(len(any_x)):

            if plot_type_y[v] == 'hist':
                ax.set_xlabel(label_y[v], fontsize=font_size, style='normal', family='sans-serif')
                ax.set_ylabel('Counts', fontsize=font_size, style='normal', family='sans-serif')
                if len(any_y) == 1:
                    colors = ['#ccffcc']
                y_hist, bin_edges = np.histogram(any_y[v], bins='auto')
                ax.bar(bin_edges[:-1] + np.diff(bin_edges) / 2, y_hist, np.diff(bin_edges), color=colors[v],
                       edgecolor='#000000', alpha=plot_alpha_y[v], label=label_y[v])
                ax.set_ylim([0, 1.1 * np.max(y_hist)])

            if plot_type_y[v] == 'line':
                ax.plot(any_x[v], any_y[v], color=colors[v], lw=2, alpha=plot_alpha_y[v], label=label_y[v], zorder=0)

            if plot_type_y[v] == 'scatter':
                if np.shape(any_y[v])[0] == 2:
                    # print('Plotting Errorbars In QuickTestPlot')
                    ax.scatter(any_x[v], any_y[v][0], color=colors[v], s=np.pi * (3) ** 2, alpha=plot_alpha_y[v],
                               label=label_y[v], zorder=1)
                    ax.errorbar(any_x[v], any_y[v][0], yerr=any_y[v][1], fmt='None', ecolor=colors[v], elinewidth=2,
                                capsize=2, capthick=2, alpha=plot_alpha_y[v], zorder=1)
                else:
                    ax.scatter(any_x[v], any_y[v], color=colors[v], s=np.pi * (3) ** 2, alpha=plot_alpha_y[v],
                               label=label_y[v], zorder=1)

            if plot_type_y[v] == 'fill_between_x':
                ax.fill_betweenx(any_y[v], any_x[v][1], any_x[v][0], color='#ccccff', alpha=plot_alpha_y[v])

            if plot_type_y[v] == 'fill_between_y':
                ax.fill_between(any_x[v], any_y[v][1], any_y[v][0], color='#ccccff', alpha=plot_alpha_y[v])

        if ('line' in plot_type_y) or (
                'scatter' in plot_type_y):  # (plot_type_y[v] == 'line') or (plot_type_y[v] == 'scatter'):
            if 'hist' not in plot_type_y:
                ax.set_xlim(x_axis_range)
                ax.set_ylim(y_axis_range)
                ax.set_xlabel(label_x, fontsize=font_size, style='normal', family='sans-serif')
                ax.set_ylabel(y_axis_label, fontsize=font_size, style='normal', family='sans-serif')
        ax.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
        ax.legend(loc='upper right', fontsize='large', framealpha=1.0, fancybox=False, frameon=True)
        plt.tight_layout()
        plt.savefig(save_as, dpi=300)
        plt.close()
        # plt.show()

    def track_genes(self, global_gene_tracker, global_gene_stddev_tracker, tracker, stddev_tracker):
        tpeaks_tracker = tracker[0]
        fwhm_tracker = tracker[1]
        ampl_tracker = tracker[2]

        tpeaks_stddev_tracker = stddev_tracker[0]
        fwhm_stddev_tracker = stddev_tracker[1]
        ampl_stddev_tracker = stddev_tracker[2]

        global_gene_tracker['tpeak'].append(tpeaks_tracker)
        global_gene_tracker['fwhm'].append(fwhm_tracker)
        global_gene_tracker['ampl'].append(ampl_tracker)

        global_gene_stddev_tracker['tpeak'].append(tpeaks_stddev_tracker)
        global_gene_stddev_tracker['fwhm'].append(fwhm_stddev_tracker)
        global_gene_stddev_tracker['ampl'].append(ampl_stddev_tracker)

        return global_gene_tracker, global_gene_stddev_tracker
    def plot_gene_progression(self, global_gene_tracker, stddev_tracker, save_as):
        tpeak_diffs = np.transpose(global_gene_tracker['tpeak'])
        fwhm_diffs = np.transpose(global_gene_tracker['fwhm'])
        ampl_diffs = np.transpose(global_gene_tracker['ampl'])

        tpeak_stddevs = np.transpose(stddev_tracker['tpeak'])
        fwhm_stddevs = np.transpose(stddev_tracker['fwhm'])
        ampl_stddevs = np.transpose(stddev_tracker['ampl'])

        gene_diffs = {'Difference From True Peak Time (min)': [],
                      'Fractional Diff From True FWHM': [],
                      'Fractional Diff From True Peak Flux': [],
                      }
        std_diffs = {'Difference From True Peak Time (min)': [],
                     'Fractional Diff From True FWHM': [],
                     'Fractional Diff From True Peak Flux': [],
                     }
        for generation in range(len(tpeak_diffs)):
            tpeak_diff = (tpeak_diffs[generation] * u.d).to(u.min).value
            fwhm_diff = fwhm_diffs[generation]
            ampl_diff = ampl_diffs[generation]
            gene_diffs['Difference From True Peak Time (min)'].append(tpeak_diff)
            gene_diffs['Fractional Diff From True FWHM'].append(fwhm_diff)
            gene_diffs['Fractional Diff From True Peak Flux'].append(ampl_diff)

            tpeak_diff_std = (tpeak_stddevs[generation] * u.d).to(u.min).value
            fwhm_diff_std = fwhm_stddevs[generation]
            ampl_diff_std = ampl_stddevs[generation]
            std_diffs['Difference From True Peak Time (min)'].append(tpeak_diff_std)
            std_diffs['Fractional Diff From True FWHM'].append(fwhm_diff_std)
            std_diffs['Fractional Diff From True Peak Flux'].append(ampl_diff_std)

        vpad = 0.05
        hpad = 0.16
        subplot_locations = {'left': np.array([0, 0, 0]) + hpad,
                             'bottom': np.array([0.60, 0.30, 0]) + vpad,
                             'width': np.array([0.97, 0.97, 0.97]) - hpad,
                             'height': np.array([0.34, 0.34, 0.34]) - vpad,
                             }
        font_size = 'medium'
        # colors = ['#0059b3','#ff6600','#990073']
        colors = ['#003366', '#006600', '#b3003b']
        # colors = [plt.cm.jet(color_i) for color_i in np.linspace(0, 1, len(average_diffs.keys()))]
        fig = plt.figure(1, figsize=(4.75, 9), facecolor="#ffffff")  # , dpi=300)
        for parameter_i, parameter in enumerate(gene_diffs.keys()):
            ax = fig.add_axes((subplot_locations['left'][parameter_i], subplot_locations['bottom'][parameter_i],
                               subplot_locations['width'][parameter_i], subplot_locations['height'][parameter_i]))

            ax.set_ylabel(parameter, fontsize=font_size)
            if parameter_i == len(gene_diffs) - 1:
                ax.set_xlabel('Generation Number', fontsize=font_size)
            if parameter_i != len(gene_diffs) - 1:
                ax.set_xticklabels([])

            yplot = np.array(np.concatenate(gene_diffs[parameter]))
            stddevplot = np.array(np.concatenate(std_diffs[parameter]))
            xplot = np.array(np.arange(0, len(yplot), 1)) + 1

            ax.plot([min(xplot), max(xplot)], [0, 0], color='#000000', alpha=0.2, lw=1.5)
            # import pdb; pdb.set_trace()
            ax.fill_between(x=xplot, y1=yplot + stddevplot, y2=yplot - stddevplot, color=colors[parameter_i], alpha=0.2)
            ax.plot(xplot, yplot, color=colors[parameter_i], lw=2)

            ax.set_xlim([min(xplot), max(xplot)])
            maxshift = max([max(abs(yplot + stddevplot)), max(abs(yplot - stddevplot))])
            ax.set_ylim([0 - 1.15 * maxshift, 0 + 1.15 * maxshift])
            ax.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
        # plt.tight_layout()
        plt.savefig(save_as, dpi=300)
        plt.close()
        # plt.show()
    def do_GA(self):

        if self.template == 'Davenport':
            initial_pop_size = 30
            n_pop = 39
            n_iter = 80  # int(80*factor)
            max_age = 3
            the_true_tpeak = self.tpeak
            the_true_fwhm = self.fwhm
            the_true_ampl = self.ampl
        if self.template == 'Jackman':
            initial_pop_size = 10
            n_pop = 10
            n_iter = 1000
            max_age = 10
            the_true_tpeak = self.tpeak_conv
            the_true_fwhm = self.fwhm_conv
            the_true_ampl = self.ampl
        Temperature = 1

        master_population_genes = []
        master_population_pool = []
        master_population_fitnesses = []
        master_population_ages = []

        percent_diff_gene_tracker = []
        gene_tracker = []
        iter_tracker = []
        fitness_tracker = []
        std_tracker = []

        gene_tracker_dict = {'tpeak': [],
                             'fwhm': [],
                             'ampl': [],
                             }
        stddev_gene_tracker_dict = {'tpeak': [],
                                    'fwhm': [],
                                    'ampl': [],
                                    }

        issue = 0

        while len(master_population_pool) < initial_pop_size:

            population_genes = []
            population_pool = []
            population_fitnesses = []
            population_ages = []

            fwhm_guess_calc = self.get_fwhm(self.x_flare, self.y_flare_scatter)

            tpeak_guess = np.abs(self.x_flare[np.where(self.y_flare_scatter == np.max(self.y_flare_scatter))[0][0]] + np.random.uniform(-0.4, 0.20, 1)[0] * self.cadence * (1. / 24.) * (1. / 60.))
            fwhm_guess = np.abs(np.random.uniform(0.50*fwhm_guess_calc, 1.25*fwhm_guess_calc, 1)[0]) * 24. * 60. # (1. / 24.) * (1. / 60.)
            ampl_guess = np.abs(np.random.uniform(0.80*max(self.y_flare_scatter), 1.20*max(self.y_flare_scatter), 1)[0])

            self.generate_candidate(guess_tpeak=tpeak_guess, guess_fwhm=fwhm_guess, guess_ampl=ampl_guess)
            population_candidate = self.y_flare_noscatter_candidate

            residuals = self.y_flare_scatter - population_candidate

            weights = abs(self.y_flare_scatter / self.y_flare_err)
            fitness = (1. / 2.) * np.sum((-np.log(2. * np.pi) - np.log((self.y_flare_err) ** 2) - ((residuals ** 2) / ((self.y_flare_err) ** 2))) * weights)

            if fitness not in population_fitnesses:
                if (np.isinf(fitness) == False) and (np.isnan(fitness) == False):
                    population_genes.append([tpeak_guess, fwhm_guess, ampl_guess])
                    population_pool.append(population_candidate)
                    population_fitnesses.append(fitness)
                    population_ages.append(0)

            sort_inds = np.array(population_fitnesses).argsort()
            population_genes = np.array(population_genes)[sort_inds[::-1]]
            population_pool = np.array(population_pool)[sort_inds[::-1]]
            population_fitnesses = np.array(population_fitnesses)[sort_inds[::-1]]
            population_ages = np.array(population_ages)[sort_inds[::-1]]

            population_genes = list(population_genes)
            population_pool = list(population_pool)
            population_fitnesses = list(population_fitnesses)
            population_ages = list(population_ages)

            master_population_genes.append(population_genes[0])
            master_population_pool.append(population_pool[0])
            master_population_fitnesses.append(population_fitnesses[0])
            master_population_ages.append(population_ages[0])


        population_genes2 = master_population_genes
        population_pool2 = master_population_pool
        population_fitnesses2 = master_population_fitnesses
        population_ages2 = master_population_ages

        for iter_i in range(n_iter):

            print(str(iter_i+1) + '/' + str(n_iter))

            if np.mod(iter_i + 1, int(0.25 * n_iter)) == 0:
                Temperature += 1

            population_ages2 = list(np.array(population_ages2) + 1)

            if issue == 1:
                break

            minpop = np.min(population_fitnesses2)

            gene_pool = copy.deepcopy(population_genes2)

            probability_distribution_temp = (population_fitnesses2 - np.min(population_fitnesses2)) / \
                                            (np.max(population_fitnesses2) - np.min(population_fitnesses2))
            probability_distribution = probability_distribution_temp / np.sum(probability_distribution_temp)
            # print(probability_distribution)

            if len(np.where(np.isnan(probability_distribution) == True)[0]) > 0:
                print(' ')
                print(len(np.where(np.isnan(probability_distribution) == True)))
                import pdb; pdb.set_trace()

            pool_elements = np.linspace(0, len(gene_pool) - 1, len(gene_pool))

            for pop_i in range(n_pop):

                tpeak_pool = np.transpose(gene_pool)[0]
                fwhm_pool = np.transpose(gene_pool)[1]
                ampl_pool = np.transpose(gene_pool)[2]

                parent1 = int(np.random.choice(pool_elements, p=probability_distribution))
                parent2 = int(np.random.choice(pool_elements, p=probability_distribution))
                count = 0
                while (parent2 == parent1):
                    # print(count)
                    parent2 = int(np.random.choice(pool_elements, p=probability_distribution))
                    count += 1
                    if count >= 100:
                        break

                if np.random.choice([True, False]) == True:
                    tpeak_guess = tpeak_pool[parent1]
                else:
                    tpeak_guess = tpeak_pool[parent2]
                if np.random.choice([True, False]) == True:
                    fwhm_guess = fwhm_pool[parent1]
                else:
                    fwhm_guess = fwhm_pool[parent2]
                if np.random.choice([True, False]) == True:
                    ampl_guess = ampl_pool[parent1]
                else:
                    ampl_guess = ampl_pool[parent2]

                made_tpeak_guess = 0
                made_fwhm_guess = 0
                made_ampl_guess = 0
                if np.random.choice([True, False, False, False, False]) == True:
                    if parent1 != parent2:
                        tpeak_guess = np.mean([tpeak_pool[parent1], tpeak_pool[parent2]])
                        made_tpeak_guess = 1

                old_fwhm_guess = np.copy(fwhm_guess)
                if np.random.choice([True, False, False, False, False]) == True:
                    if parent1 != parent2:
                        fwhm_guess = np.mean([fwhm_pool[parent1], fwhm_pool[parent2]])
                        made_fwhm_guess = 1

                if np.random.choice([True, False, False, False, False]) == True:
                    if parent1 != parent2:
                        ampl_guess = np.mean([ampl_pool[parent1], ampl_pool[parent2]])
                        made_ampl_guess = 1

                if made_tpeak_guess != 1:
                    if self.template == 'Davenport':
                        random_draw = np.random.choice([True, False])
                    if self.template == 'Jackman':
                        random_draw = np.random.choice([True, False])
                    if random_draw == True:  # Does it mutate
                        if np.random.choice([True, False, False]) == True:  # Is it a big mutation
                            tpeak_guess = np.abs(np.random.normal(tpeak_guess, 0.50 * self.cadence / 24. / 60., 1)[0])
                        else:
                            tpeak_guess = np.abs(np.random.normal(tpeak_guess, 0.05 * self.cadence / 24. / 60., 1)[0])

                # fwhm_mutated = False
                if made_fwhm_guess != 1:
                    if self.template == 'Davenport':
                        random_draw = np.random.choice([True, False, False, False])
                    if self.template == 'Jackman':
                        random_draw = np.random.choice([True, False, False, False])
                    if random_draw == True:  # Does it mutate
                        if np.random.choice([True, False, False]) == True:  # Is it a big mutation
                            fwhm_guess = np.abs(np.random.normal(old_fwhm_guess, 0.50 * old_fwhm_guess, 1)[0])
                        else:
                            fwhm_guess = np.abs(np.random.normal(old_fwhm_guess, 0.05 * old_fwhm_guess, 1)[0])

                if made_ampl_guess != 1:
                    old_ampl_guess = np.copy(ampl_guess)
                    # reguess = 0
                    if self.template == 'Davenport':
                        random_draw = np.random.choice([True, False])
                    if self.template == 'Jackman':
                        random_draw = np.random.choice([True, False])
                    if random_draw == True:  # Does it mutate
                        if np.random.choice([True, False, False]) == True:  # Is it a big mutation
                            ampl_guess = np.abs(np.random.normal(old_ampl_guess, 0.30 * old_ampl_guess, 1)[0])
                        elif np.random.choice([True, False, False]) == True:
                            ampl_guess = np.abs(np.random.normal(old_ampl_guess, 0.05 * old_ampl_guess, 1)[0])
                        elif np.random.choice([True, False, False]) == True:  # Is it a big mutation
                                ampl_guess = np.abs(np.random.normal(old_ampl_guess + 0.50*old_ampl_guess,
                                                                     0.30 * old_ampl_guess, 1)[0])

                self.generate_candidate(tpeak_guess, fwhm_guess, ampl_guess)
                population_candidate = self.y_flare_noscatter_candidate

                residuals = self.y_flare_scatter - population_candidate

                # fitness = -len(flare_f) * np.log(np.sqrt(2. * np.pi) * (np.sum(residuals_std * weights))) - np.sum((residuals ** 2) / ((residuals_std * weights) ** 2))
                # fitness = -len(flare_f) * np.log(np.sqrt(2. * np.pi) * (np.sum(residuals_std))) - np.sum((residuals ** 2) / ((residuals_std) ** 2))
                # if self.template == 1:
                #     fitness = (1. / 2.) * np.sum(-np.log(2. * np.pi) - np.log(self.y_flare_err ** 2) - ((residuals ** 2) / (self.y_flare_err ** 2)))

                weights = abs(self.y_flare_scatter / self.y_flare_err)
                # fitness = (1. / 2.) * np.sum(-np.log(2. * np.pi) - np.log((flare_f_err*weights) ** 2) - ((residuals ** 2) / ((flare_f_err*weights) ** 2)))
                fitness = (1. / 2.) * np.sum((-np.log(2. * np.pi) - np.log((self.y_flare_err) ** 2) - ((residuals ** 2) / ((self.y_flare_err) ** 2))) * weights)
                # fitness = np.round(fitness, 1)
                # print(fitness)
                # import pdb; pdb.set_trace()

                if fitness not in population_fitnesses2:
                    if (np.isinf(fitness) == False) and (np.isnan(fitness) == False):
                        population_genes2.append([tpeak_guess, fwhm_guess, ampl_guess])
                        population_pool2.append(population_candidate)
                        population_fitnesses2.append(fitness)
                        population_ages2.append(0)

                        if fitness > minpop:
                            new_member_flag = 1
                else:
                    new_member_flag = 0

            # import pdb; pdb.set_trace()

            if len(population_pool2) > n_pop:

                where_young = np.where(np.array(population_ages2) < max_age)[0]
                if len(where_young) < int(0.5 * n_pop):
                    where_old = np.where(np.array(population_ages2) >= max_age)[0]
                    makeup_diff = int(0.75 * n_pop) - len(where_young)

                    # print('pausing where we let old live')
                    # import pdb; pdb.set_trace()

                    where_old_live = np.random.choice(where_old, makeup_diff)

                    where_carry = np.concatenate((where_young, where_old_live))
                    population_genes2 = np.array(population_genes2)[where_carry]
                    population_pool2 = np.array(population_pool2)[where_carry]
                    population_fitnesses2 = np.array(population_fitnesses2)[where_carry]
                    population_ages2 = np.array(population_ages2)[where_carry]

                    sort_inds = np.array(population_fitnesses2).argsort()
                    population_genes3 = np.array(population_genes2)[sort_inds[::-1]]
                    population_pool3 = np.array(population_pool2)[sort_inds[::-1]]
                    population_fitnesses3 = np.array(population_fitnesses2)[sort_inds[::-1]]
                    population_ages3 = np.array(population_ages2)[sort_inds[::-1]]

                    population_genes4 = list(population_genes3)
                    population_pool4 = list(population_pool3)
                    population_fitnesses4 = list(population_fitnesses3)
                    population_ages4 = list(population_ages3)
                else:
                    population_genes2 = np.array(population_genes2)[where_young]
                    population_pool2 = np.array(population_pool2)[where_young]
                    population_fitnesses2 = np.array(population_fitnesses2)[where_young]
                    population_ages2 = np.array(population_ages2)[where_young]

                    sort_inds = population_fitnesses2.argsort()
                    population_genes3 = population_genes2[sort_inds[::-1]]
                    population_pool3 = population_pool2[sort_inds[::-1]]
                    population_fitnesses3 = population_fitnesses2[sort_inds[::-1]]
                    population_ages3 = population_ages2[sort_inds[::-1]]

                    population_genes4 = list(population_genes3[0:n_pop])
                    population_pool4 = list(population_pool3[0:n_pop])
                    population_fitnesses4 = list(population_fitnesses3[0:n_pop])
                    population_ages4 = list(population_ages3[0:n_pop])

            else:
                sort_inds = np.array(population_fitnesses2).argsort()
                population_genes3 = np.array(population_genes2)[sort_inds[::-1]]
                population_pool3 = np.array(population_pool2)[sort_inds[::-1]]
                population_fitnesses3 = np.array(population_fitnesses2)[sort_inds[::-1]]
                population_ages3 = np.array(population_ages2)[sort_inds[::-1]]

                population_genes4 = list(population_genes3)  # [0:n_pop])
                population_pool4 = list(population_pool3)  # [0:n_pop])
                population_fitnesses4 = list(population_fitnesses3)  # [0:n_pop])
                population_ages4 = list(population_ages3)  # [0:n_pop])

            population_genes2 = copy.deepcopy(population_genes4)
            population_pool2 = copy.deepcopy(population_pool4)
            population_fitnesses2 = copy.deepcopy(population_fitnesses4)
            population_ages2 = copy.deepcopy(population_ages4)




            # -------------- track gene progression ----------------#

            population_genes2_track = copy.deepcopy(population_genes2)

            population_genes2_temp = np.transpose(population_genes2_track)

            fitness_tracker.append(population_fitnesses2[0])
            # if iter_i > 0:
            #     print(fitness_tracker[-1], fitness_tracker[-2])


            gene_tracker.append([population_genes2_temp[0][0], population_genes2_temp[1][0], population_genes2_temp[2][0]])


            diff_tpeak = population_genes2_temp[0][0] - the_true_tpeak
            percent_diff_fwhm = (population_genes2_temp[1][0] / the_true_fwhm) - 1.
            percent_diff_ampl = (population_genes2_temp[2][0] / the_true_ampl) - 1.

            stddev_tpeak = np.std(population_genes2_temp[0])
            stddev_fwhm = np.std(population_genes2_temp[1])
            stddev_ampl = np.std(population_genes2_temp[2])

            percent_diff_gene_tracker.append([diff_tpeak, percent_diff_fwhm, percent_diff_ampl])
            std_tracker.append([stddev_tpeak, stddev_fwhm, stddev_ampl])


            iter_tracker.append(iter_i + 1)




            plot_tpeak = population_genes2_temp[0][0]
            plot_fwhm = population_genes2_temp[1][0]
            plot_ampl = population_genes2_temp[2][0]

            save_movie_plots = True
            if save_movie_plots == True:
                save_every = 10

                if iter_i == 0:

                    self.generate_candidate(plot_tpeak, plot_fwhm, plot_ampl)

                    plot_x = [self.x_synth, self.x_flare, self.x_synth_candidate]
                    plot_y = [self.y_synth_noscatter, [self.y_flare_scatter, self.y_flare_err],
                              self.y_synth_noscatter_candidate]
                    plot_types_y = ['line', 'scatter', 'line']
                    plot_alphas_y = [1.0, 1.0, 1.0]

                    # thres_y = [0, 1.1 * np.max([np.max(self.y_synth_noscatter), np.max(self.y_flare_scatter), np.max(self.y_synth_noscatter_candidate)])]
                    thres_y = [0, 1.2]
                    x_range = [0.45, 0.65]

                    label_y = ['Truth', 'Simulated Observations', 'Best Fit']
                    pl_title = 'FWHM: ' + str(np.round(plot_fwhm,2)) + ' min   Amplitude: ' + str(np.round(plot_ampl,2))

                    save_test_individual = '/Users/lbiddle/Desktop/Plots_For_Dissertation/Chapter3_Figures/Chapter3_Movie/' + \
                                           str(iter_i+1) + '_Flare_Fit_' + self.template + '_' + str(self.cadence) + '.pdf'
                    self.quick_test_plot(any_x=plot_x, any_y=plot_y, label_x='Time (d)', label_y=label_y,
                                         plot_type_y=plot_types_y,
                                         plot_alpha_y=plot_alphas_y, y_axis_label='Flare Flux', x_axis_range=x_range,
                                         y_axis_range=thres_y, plot_title=pl_title,save_as=save_test_individual)

            if (iter_i > 0) and ((fitness_tracker[-2] < fitness_tracker[-1]) or (np.mod(iter_i + 1, save_every) == 0)):

                self.generate_candidate(plot_tpeak, plot_fwhm, plot_ampl)

                plot_x = [self.x_synth, self.x_flare, self.x_synth_candidate]
                plot_y = [self.y_synth_noscatter, [self.y_flare_scatter, self.y_flare_err],
                          self.y_synth_noscatter_candidate]
                plot_types_y = ['line', 'scatter', 'line']
                plot_alphas_y = [1.0, 1.0, 1.0]

                thres_y = [0, 1.2]
                x_range = [0.475, 0.60]

                label_y = ['Truth', 'Simulated Observations', 'Best Fit']
                pl_title = 'FWHM: ' + str(np.round(plot_fwhm, 2)) + ' min   Amplitude: ' + str(np.round(plot_ampl, 2))

                save_test_individual = '/Users/lbiddle/Desktop/Plots_For_Dissertation/Chapter3_Figures/Chapter3_Movie/' + \
                                       str(iter_i + 1) + '_Flare_Fit_' + self.template + '_' + str(self.cadence) + '.pdf'
                self.quick_test_plot(any_x=plot_x, any_y=plot_y, label_x='Time (d)', label_y=label_y,
                                     plot_type_y=plot_types_y,
                                     plot_alpha_y=plot_alphas_y, y_axis_label='Flare Flux', x_axis_range=x_range,
                                     y_axis_range=thres_y, plot_title=pl_title, save_as=save_test_individual)


                percent_diff_gene_tracker2 = np.transpose(percent_diff_gene_tracker)
                std_tracker2 = np.transpose(std_tracker)
                plot_gene_tracker, plot_stddev_tracker = self.track_genes(global_gene_tracker=gene_tracker_dict, global_gene_stddev_tracker=stddev_gene_tracker_dict, tracker=percent_diff_gene_tracker2, stddev_tracker=std_tracker2)
                save_as_tracker = '/Users/lbiddle/Desktop/Plots_For_Dissertation/Chapter3_Figures/Chapter3_Movie/gene_progression_' + self.template + '_' + str(self.cadence) + '.pdf'
                self.plot_gene_progression(plot_gene_tracker, plot_stddev_tracker, save_as=save_as_tracker)


            if iter_i == n_iter - 1:

                self.generate_candidate(plot_tpeak, plot_fwhm, plot_ampl)

                plot_x = [self.x_synth, self.x_flare, self.x_synth_candidate]
                plot_y = [self.y_synth_noscatter, [self.y_flare_scatter, self.y_flare_err], self.y_synth_noscatter_candidate]
                plot_types_y = ['line', 'scatter', 'line']
                plot_alphas_y = [1.0, 1.0, 1.0]

                # thres_y = [0, 1.1 * np.max([np.max(self.y_synth_noscatter), np.max(self.y_flare_scatter), np.max(self.y_synth_noscatter_candidate)])]
                thres_y = [0, 1.2]
                x_range = [0.475, 0.60]

                label_y = ['Truth', 'Simulated Observations', 'Best Fit']
                pl_title = 'FWHM: ' + str(np.round(plot_fwhm, 2)) + ' min   Amplitude: ' + str(np.round(plot_ampl, 2))

                save_test_individual = '/Users/lbiddle/Desktop/Plots_For_Dissertation/Chapter3_Figures/Flare_Fit_' + \
                                       self.template + '_' + str(self.cadence) + '.pdf'
                self.quick_test_plot(any_x=plot_x, any_y=plot_y, label_x='Time (d)', label_y=label_y,
                                     plot_type_y=plot_types_y,
                                     plot_alpha_y=plot_alphas_y, y_axis_label='Flare Flux', x_axis_range=x_range,
                                     y_axis_range=thres_y, plot_title=pl_title,save_as=save_test_individual)

                percent_diff_gene_tracker2 = np.transpose(percent_diff_gene_tracker)
                std_tracker2 = np.transpose(std_tracker)
                plot_gene_tracker, plot_stddev_tracker = self.track_genes(global_gene_tracker=gene_tracker_dict, global_gene_stddev_tracker=stddev_gene_tracker_dict, tracker=percent_diff_gene_tracker2, stddev_tracker=std_tracker2)
                save_as_tracker = '/Users/lbiddle/Desktop/Plots_For_Dissertation/Chapter3_Figures/Chapter3_Movie/gene_progression_' + self.template + '_' + str(self.cadence) + '.pdf'
                self.plot_gene_progression(plot_gene_tracker, plot_stddev_tracker, save_as=save_as_tracker)

        gene_tracker2 = np.transpose(gene_tracker)

        best_population_genes = []
        population_gene_uncertainties = []

        for tracker_i in range(3):
            mean_tracker = np.average(gene_tracker2[tracker_i])  # ,weights=1./std_tracker2[tracker_i])
            spread_tracker = np.std(gene_tracker2[tracker_i])  # - mean_tracker, weights=1./std_tracker2[tracker_i])
            # spread_tracker = np.var(gene_tracker2[tracker_i])

            best_population_genes.append(mean_tracker)
            population_gene_uncertainties.append(3. * spread_tracker)

        # import pdb; pdb.set_trace()

        print('Best Population Genes: ' + str(best_population_genes))
        print('Uncertainties: ' + str(population_gene_uncertainties))
        # print(np.array(population_gene_uncertainties) / np.array(best_population_genes))

        return best_population_genes, population_gene_uncertainties


    def fit_flare(self):
        self.popt, self.perr = self.do_GA()

lightcurve_cadence = 30.0 # minutes
stddev = 0.001 * (1. / 50)

template_list = ['Davenport', 'Jackman']
amplitudes = [0.2, 0.4, 0.6, 0.8, 1.0]
widths = [1, 2, 4, 8, 16, 32]

ampl_colors = [plt.cm.viridis_r(color_i) for color_i in np.linspace(0, 1, len(amplitudes))]
fwhm_colors = [plt.cm.viridis_r(color_i) for color_i in np.linspace(0, 1, len(widths))]
font_size = 'medium'
plt.close()

axes1 = [0.08,0.12,0.40,0.85]
axes2 = [0.58,0.12,0.40,0.85]

do_GA_fit = True
if do_GA_fit == True:

    set_tpeak = 0.5 # days
    set_fwhm = 10.0 # min
    set_ampl = 1.0

    flare_template = 'Jackman'

    F = Flares(template=flare_template, cadence=lightcurve_cadence, tpeak=set_tpeak, fwhm=set_fwhm, ampl=set_ampl,
               downsample=True, GA=True)
    F.generate_flare()
    F.fit_flare()

do_compare_modelpars = False
if do_compare_modelpars == True:
    for template_i, flare_template in enumerate(template_list):

        print('\nCalculating Amplitudes For Template: ' + flare_template)

        fig = plt.figure(1, figsize=(9.5, 4.0), facecolor="#ffffff")
        ax1 = fig.add_axes(axes1)
        ax2 = fig.add_axes(axes2)

        for ampl_i, set_ampl in enumerate(amplitudes):

            set_tpeak = 0.5 # days
            set_fwhm = 15 # minutes
            # set_ampl = 1

            F = Flares(template=flare_template, cadence=lightcurve_cadence, tpeak=set_tpeak, fwhm=set_fwhm,
                       ampl=set_ampl, downsample=True, GA=False)
            F.generate_flare()


            ax1.plot([0,1], [0,0], color='#000000', alpha=0.15)

            x_plot = F.x_flare
            y_plot = F.y_flare_noscatter
            ax1.plot(x_plot, y_plot, color=ampl_colors[ampl_i], label='ampl = ' + str(np.round(set_ampl,1)))

            ax2.scatter([set_ampl], [F.eqdur_noscatter], color=ampl_colors[ampl_i], s=np.pi*(4)**2)

        # ax.errorbar(x=[star_mass], y=[star_empirical_radius],
        #             yerr=[[star_empirical_radius_lower], [star_empirical_radius_upper]], ecolor=Empirical_color,
        #             elinewidth=1.0, capsize=2.5, capthick=1, linestyle='None', marker='None')
        # ax.scatter([star_mass], [star_empirical_radius], color=Empirical_color, marker='*', s=80)
        ax1.set_xlim(0.45, 0.75)
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_xlabel('Time (d)')
        ax1.set_ylabel('Flare Flux')
        ax1.legend(loc='upper right', fontsize='large', framealpha=1.0, fancybox=False, frameon=False)
        ax1.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)


        ax2.set_xlim(0.1,1.1)
        # ax2.set_xticklabels([0.2, 0.4, 0.6, 0.8, 1.0])
        ax2.set_ylim(0, 2500)
        ax2.set_xlabel('Amplitude')
        ax2.set_ylabel('Equivalent Duration (s)')
        ax2.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)

        # plt.tight_layout()
        plt.savefig('/Users/lbiddle/Desktop/Plots_For_Dissertation/Chapter3_Figures/amplitudes_' + flare_template +'.pdf', dpi=300)
        plt.close()


    for template_i, flare_template in enumerate(template_list):

        print('\nCalculating Widths For Template: ' + flare_template)

        fig = plt.figure(1, figsize=(9.5, 4.0), facecolor="#ffffff")
        ax1 = fig.add_axes(axes1)
        ax2 = fig.add_axes(axes2)

        for width_i, set_fwhm in enumerate(widths):

            set_tpeak = 0.5 # days
            #set_fwhm = 15 # minutes
            set_ampl = 1

            F = Flares(template=flare_template, cadence=lightcurve_cadence, tpeak=set_tpeak, fwhm=set_fwhm,
                       ampl=set_ampl, downsample=True, GA=False)
            F.generate_flare()

            ax1.plot([0,1], [0,0], color='#000000', alpha=0.15)

            if flare_template == 'Jackman':
                fwhm_plot = F.fwhm_conv
            if flare_template == 'Davenport':
                fwhm_plot = set_fwhm

            x_plot = F.x_flare
            y_plot = F.y_flare_noscatter
            ax1.plot(x_plot, y_plot, color=fwhm_colors[width_i], label='FWHM = ' + str(int(np.floor(fwhm_plot))) + ' min')

            ax2.scatter([fwhm_plot], [F.eqdur_noscatter], color=fwhm_colors[width_i], s=np.pi*(4)**2)

        # ax.errorbar(x=[star_mass], y=[star_empirical_radius],
        #             yerr=[[star_empirical_radius_lower], [star_empirical_radius_upper]], ecolor=Empirical_color,
        #             elinewidth=1.0, capsize=2.5, capthick=1, linestyle='None', marker='None')
        # ax.scatter([star_mass], [star_empirical_radius], color=Empirical_color, marker='*', s=80)
        ax1.set_xlim(0.45, 0.75)
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_xlabel('Time (d)')
        ax1.set_ylabel('Flare Flux')
        ax1.legend(loc='upper right', fontsize='large', framealpha=1.0, fancybox=False, frameon=False)
        ax1.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)


        # ax2.set_xlim(0.1,1.1)
        # ax2.set_xticklabels([0.2, 0.4, 0.6, 0.8, 1.0])
        ax2.set_ylim(0, 3600)
        ax2.set_xlabel('FWHM (min)')
        ax2.set_ylabel('Equivalent Duration (s)')
        ax2.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)

        # plt.tight_layout()
        plt.savefig('/Users/lbiddle/Desktop/Plots_For_Dissertation/Chapter3_Figures/fwhms_' + flare_template +'.pdf', dpi=300)
        plt.close()


