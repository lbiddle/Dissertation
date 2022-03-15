import numpy as np
import astropy.units as u
import pandas as pd
from astropy.constants import G, m_p, M_sun, R_sun, k_B, sigma_sb, L_sun  # These are in Meters Kilograms Seconds
import matplotlib.pyplot as plt


Mstar = 0.90 * u.Msun
Mstar_err = 0.02 * u.Msun
Teff = 4277 * u.K # 4060 * u.K
Teff_err = 50 * u.K
Luminosity = 0.93 * u.Lsun
Luminosity_err = 0.35 * u.Lsun
vsini = (10.08 * u.km) / (1. * u.s)
vsini_err = (0.31 * u.km) / (1. * u.s)
Prot = (6.70 * u.d).to(u.s)
Prot_err = (0.23 * u.d).to(u.s)
Porb = (22.2 * u.d).to(u.s) # (9.08 * u.d).to(u.s)
Porb_err = (2.2 * u.d).to(u.s) # (0.45 * u.d).to(u.s)
i = (49.99 * u.deg).to(u.rad)
i_err = (0.12 * u.deg).to(u.rad)
Age = 2.5 * u.Myr
Age_err = 0.5 * u.Myr
Mplanet = 11.6 * u.Mjup
Mplanet_err = 2.9 * u.Mjup
ecc = 0.25
ecc_err = 0.16

class Calculations:
    def __init__(self, Mstar, Mstar_err, Teff, Teff_err, Luminosity, Luminosity_err,
                 vsini, vsini_err, Prot, Prot_err, Porb, Porb_err, i, i_err, Age, Age_err,
                 Mplanet, Mplanet_err, ecc, ecc_err):
        self.Mstar = Mstar.value
        self.Mstar_err = Mstar_err.value
        self.Teff = Teff.value
        self.Teff_err = Teff_err.value
        self.Luminosity = Luminosity.value
        self.Luminosity_err = Luminosity_err.value
        self.vsini = vsini.value
        self.vsini_err = vsini_err.value
        self.Prot = Prot.value
        self.Prot_err = Prot_err.value
        self.Porb = Porb.value
        self.Porb_err = Porb_err.value
        self.i = i.value
        self.i_err = i_err.value
        self.Age = Age.value
        self.Age_err = Age_err.value
        self.Mplanet = Mplanet.value
        self.Mplanet_err = Mplanet_err.value
        self.ecc = ecc
        self.ecc_err = ecc_err
        self.Baraffe = pd.read_csv('/Users/lbiddle/Desktop/Plots_For_Dissertation/Baraffe2015.csv')

    def calc_Rstar_with_Period(self):
        const = 1./(2.0 * np.pi)
        self.Rstar_calculated_rot = ((const * (self.Prot * self.vsini) / np.sin(self.i))*u.km).to(u.Rsun).value

        self.Rstar_calculated_rot_err = ((np.sqrt((const * self.vsini / np.sin(self.i))**2 * self.Prot_err**2 +
                                       (const * self.Prot / np.sin(self.i))**2 * self.vsini_err**2 +
                                       (2 * const * self.Prot * self.vsini * np.cos(self.i) / (np.cos(2*self.i)-1.0))**2 * self.i_err**2))*u.km).to(u.Rsun).value

        self.Rstar_calculated_orb = ((const * (self.Porb * self.vsini) / np.sin(self.i))*u.km).to(u.Rsun).value

        self.Rstar_calculated_orb_err = ((np.sqrt((const * self.vsini / np.sin(self.i))**2 * self.Porb_err**2 +
                                       (const * self.Porb / np.sin(self.i))**2 * self.vsini_err**2 +
                                       (2 * const * self.Porb * self.vsini * np.cos(self.i) / (np.cos(2*self.i)-1.0))**2 * self.i_err**2))*u.km).to(u.Rsun).value

        print('Rotation Radius: ' + str(np.round(self.Rstar_calculated_rot, 3)) +
              '  +/-  ' + str(np.round(self.Rstar_calculated_rot_err, 3)) +
              '  Rsun')
        print('Orbit Radius: ' + str(np.round(self.Rstar_calculated_orb, 3)) +
              '  +/-  ' + str(np.round(self.Rstar_calculated_orb_err, 3)) +
              '  Rsun')

    def calc_Rstar_empirical(self):
        const = (1./ (4.0 * np.pi * sigma_sb)).value
        L = (self.Luminosity * L_sun).value
        L_err = (self.Luminosity_err * L_sun).value

        # import pdb; pdb.set_trace()
        self.Rstar_empirical = ((np.sqrt(const * L/(self.Teff**4)))* u.m).to(u.Rsun).value

        self.Rstar_empirical_err = ((np.sqrt( ((np.sqrt(const * L/(self.Teff**4)))/(2*L))**2 * L_err**2 +
                                            (-2*(np.sqrt(const * L/(self.Teff**4)))/self.Teff)**2 * self.Teff_err**2 ))* u.m).to(u.Rsun).value

        print('Empirical Radius: ' + str(np.round(self.Rstar_empirical, 3)) +
              '  +/-  ' +
              str(np.round(self.Rstar_empirical_err, 3)) +
              '  Rsun')

    def get_Baraffe_Model(self):
        if self.Mstar < 0.1:
            Baraffe_Models = self.Baraffe[self.Baraffe['Mstar (Msun)'] == np.round(self.Mstar,2)]
        if self.Mstar >= 0.1:
            Baraffe_Models = self.Baraffe[self.Baraffe['Mstar (Msun)'] == np.round(self.Mstar,1)]
        Age_difference = abs(Baraffe_Models['Age (Myr)'].values - self.Age)
        Age_difference_upper = abs(Baraffe_Models['Age (Myr)'].values - (self.Age + self.Age_err))
        Age_difference_lower = abs(Baraffe_Models['Age (Myr)'].values - (self.Age - self.Age_err))
        where_min_Age_difference = np.where(Age_difference == min(Age_difference))[0][0]
        where_min_Age_difference_upper = np.where(Age_difference_upper == min(Age_difference_upper))[0][0]
        where_min_Age_difference_lower = np.where(Age_difference_lower == min(Age_difference_lower))[0][0]
        Star_Model = Baraffe_Models[Baraffe_Models['Age (Myr)'] == Baraffe_Models['Age (Myr)'].values[where_min_Age_difference]]
        Star_Model_upper = Baraffe_Models[Baraffe_Models['Age (Myr)'] == Baraffe_Models['Age (Myr)'].values[where_min_Age_difference_upper]]
        Star_Model_lower = Baraffe_Models[Baraffe_Models['Age (Myr)'] == Baraffe_Models['Age (Myr)'].values[where_min_Age_difference_lower]]

        Baraffe_Radius_upper = Star_Model_upper['Rstar (Rsun)'].values[0]
        Baraffe_Radius_lower = Star_Model_lower['Rstar (Rsun)'].values[0]

        Baraffe_logg_upper = Star_Model_upper['logg (g/cm^3)'].values[0]
        Baraffe_logg_lower = Star_Model_lower['logg (g/cm^3)'].values[0]

        self.Baraffe_Radius = Star_Model['Rstar (Rsun)'].values[0]
        self.Baraffe_Radius_upper = abs(self.Baraffe_Radius - Baraffe_Radius_upper)
        self.Baraffe_Radius_lower = abs(self.Baraffe_Radius - Baraffe_Radius_lower)

        self.Baraffe_logg = Star_Model['logg (g/cm^3)'].values[0]
        self.Baraffe_logg_upper = abs(self.Baraffe_logg - Baraffe_logg_upper)
        self.Baraffe_logg_lower = abs(self.Baraffe_logg - Baraffe_logg_lower)

        print('Baraffe Radius: ' + str(np.round(self.Baraffe_Radius, 3)) +
              '  +  ' + str(np.round(self.Baraffe_Radius_upper, 3)) +
              '  -  ' + str(np.round(self.Baraffe_Radius_lower, 3)) +
              '  Rsun')
        print('Baraffe logg: ' + str(np.round(self.Baraffe_logg, 3)) +
              '  +  ' + str(np.round(self.Baraffe_logg_upper, 3)) +
              '  -  ' + str(np.round(self.Baraffe_logg_lower, 3)) +
              '  g/cm^3')



    def calc_Corotation_Radius(self):
        const = (G / (4.0*np.pi**2)).value
        M = (self.Mstar * M_sun).value
        M_err = (self.Mstar_err * M_sun).value

        self.Rcor_au = (((const * self.Prot**2 * M)**(1./3.)) * u.m).to(u.au).value

        self.Rcor_au_err = ((np.sqrt(((2./3.)*((const*M)/self.Prot)**(1./3.))**2 * self.Prot_err**2 +
                                 ((1./3.)*((const*self.Prot**2)/(M**2))**(1./3.))**2 * M_err**2)) * u.m).to(u.au).value

        print('Co-rotation Radius: ' + str(np.round(self.Rcor_au, 3)) +
              '  +/-  ' + str(np.round(self.Rcor_au_err, 3)) +
              '  au')


    def calc_Orbit_Distance(self):
        const = (G / (4.0*np.pi**2)).value
        M = (self.Mstar * M_sun).value
        M_err = (self.Mstar_err * M_sun).value
        Mpl = (self.Mplanet * u.Mjup).to(u.kg).value
        Mpl_err = (self.Mplanet_err * u.Mjup).to(u.kg).value

        # self.Rorb_au = (((const * self.Porb ** 2 * M) ** (1. / 3.)) * u.m).to(u.au).value
        #
        # self.Rorb_au_err = ((np.sqrt(((2. / 3.) * ((const * M) / self.Porb) ** (1. / 3.)) ** 2 * self.Porb_err ** 2 +
        #                              ((1. / 3.) * ((const * self.Prot ** 2) / (M ** 2)) ** (
        #                                          1. / 3.)) ** 2 * M_err ** 2)) * u.m).to(u.au).value

        self.Rorb_au = (((const * self.Porb**2 * (M + Mpl))**(1./3.)) * u.m).to(u.au).value

        self.Rorb_au_err = (np.sqrt(((2*(const*(Mpl + M)*self.Porb**2)**(1/3))/(3*self.Porb))**2 * self.Porb_err**2 +
                                   (((const*self.Porb**2)/(Mpl + M)**2)**(1/3)/3)**2 * M_err**2 +
                                   (((const*self.Porb**2)/(Mpl + M)**2)**(1/3)/3)**2 * Mpl_err**2) * u.m).to(u.au).value

        print('Planet Orbit Distance: ' + str(np.round(self.Rorb_au, 3)) +
              '  +/-  ' + str(np.round(self.Rorb_au_err, 3)) +
              '  au')

    def calc_Hill_Radius(self):
        const = (G / (4.0 * np.pi ** 2)).value
        Mst = (self.Mstar * M_sun).value
        Mst_err = (self.Mstar_err * M_sun).value
        Mpl = (self.Mplanet*u.Mjup).to(u.kg).value
        Mpl_err = (self.Mplanet_err * u.Mjup).to(u.kg).value
        a = (self.Rorb_au*u.au).to(u.m).value
        a_err = (self.Rorb_au_err * u.au).to(u.m).value

        self.Rhill_au = ((a * (1. - self.ecc) * (Mpl / (3. * Mst))**(1./3.)) * u.m).to(u.au).value

        self.Rhill_au_err = (np.sqrt( (((1.-self.ecc)*(Mpl/Mst)**(1/3))/(3**(1/3)))**2 * a_err**2 +
                                     (-(a*(-1+self.ecc)*(Mpl/Mst)**(1/3))/(3*3**(1/3)*Mpl))**2 * Mpl_err**2 +
                                     ((a*(-1+self.ecc)*(Mpl/Mst)**(1/3))/(3*3**(1/3)*Mst))**2 * Mst_err**2 +
                                     (-((a*(Mpl/Mst)**(1/3))/(3**(1/3))))**2 * self.ecc_err**2)*u.m).to(u.au).value

        print('Hill Radius: ' + str(np.round(self.Rhill_au, 3)) +
              '  +/-  ' + str(np.round(self.Rhill_au_err, 3)) +
              '  au')

        # self.Rhill_au_err =


    def plot_Baraffe_Isochrones(self):

        Baraffe_masses = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07,
                          0.072, 0.075, 0.08, 0.09, 0.1, 0.11, 0.13, 0.15,
                          0.17, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                          1.1, 1.2, 1.3, 1.4]

        # ages = [0.75, 1.25, 2.0, 3.0, 4.5, 6.5, 9.0]
        # age_range = [0.25, 0.25, 0.5, 0.5, 1.0, 1.0, 1.5]

        ages = [0.75, 1.5, 2.5, 3.5, 5.0, 7.0, 9.0]
        age_range = [0.25, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0]


        radii_upper_plot = []
        radii_lower_plot = []

        for age_i, age in enumerate(ages):

            Baraffe_Age_Models = self.Baraffe[self.Baraffe['Age (Myr)'] >= age - age_range[age_i]]
            Baraffe_Age_Models = Baraffe_Age_Models[Baraffe_Age_Models['Age (Myr)'] <= age + age_range[age_i]]

            radii_upper_per_mass = []
            radii_lower_per_mass = []

            for mass_i, mass in enumerate(Baraffe_masses):

                Baraffe_mass_models = Baraffe_Age_Models[Baraffe_Age_Models['Mstar (Msun)'] == mass]

                min_age = min(Baraffe_mass_models['Age (Myr)'].values)
                max_age = max(Baraffe_mass_models['Age (Myr)'].values)

                min_age_model = Baraffe_mass_models[Baraffe_mass_models['Age (Myr)'] == min_age]
                max_age_model = Baraffe_mass_models[Baraffe_mass_models['Age (Myr)'] == max_age]

                radius_min_age = min_age_model['Rstar (Rsun)'].values[0]
                radius_max_age = max_age_model['Rstar (Rsun)'].values[0]

                radii_upper_per_mass.append(radius_max_age)
                radii_lower_per_mass.append(radius_min_age)

            radii_upper_plot.append(radii_upper_per_mass)
            radii_lower_plot.append(radii_lower_per_mass)

        font_size = 'medium'
        #colors = ['green', 'blue', 'red', 'purple', 'orange', 'black']
        colors = [plt.cm.viridis_r(color_i) for color_i in np.linspace(0, 1, len(ages))]
        fig = plt.figure(1, figsize=(5, 6), facecolor="#ffffff")  # , dpi=300)
        ax = fig.add_subplot(111)

        for age_i, age in enumerate(ages):

            ax.fill_between(Baraffe_masses, y1=radii_upper_plot[age_i], y2=radii_lower_plot[age_i],
                            color=colors[age_i], alpha=1.0, label=str(np.round(age - age_range[age_i], 2)) + ' - ' + str(np.round(age + age_range[age_i], 2)) + ' Myr')
            star_mass = self.Mstar
            star_model_radius_rot = self.Rstar_calculated_rot
            star_model_radius_rot_upper = self.Rstar_calculated_rot_err
            star_model_radius_rot_lower = self.Rstar_calculated_rot_err

            star_model_radius_orb = self.Rstar_calculated_orb
            star_model_radius_orb_upper = self.Rstar_calculated_orb_err
            star_model_radius_orb_lower = self.Rstar_calculated_orb_err

            star_empirical_radius = self.Rstar_empirical
            star_empirical_radius_upper = self.Rstar_empirical_err
            star_empirical_radius_lower = self.Rstar_empirical_err
            # ax.vlines(star_mass, ymin=0, ymax=star_model_radius, color='#000000', linestyle='--', lw=0.75, alpha=0.10)
            # ax.hlines(star_model_radius, xmin=0, xmax=star_mass, color='#000000', linestyle='--', lw=0.75, alpha=0.10)

            Empirical_color = '#ffffff'
            Rrot_color = '#000000'

            ax.errorbar(x=[star_mass], y=[star_empirical_radius],
                        yerr=[[star_empirical_radius_lower], [star_empirical_radius_upper]], ecolor=Empirical_color,
                        elinewidth=1.0, capsize=2.5, capthick=1, linestyle='None', marker='None')
            ax.scatter([star_mass], [star_empirical_radius], color=Empirical_color, marker='*', s=80)
            ax.errorbar(x=[star_mass], y=[star_model_radius_rot], yerr=[[star_model_radius_rot_lower],[star_model_radius_rot_upper]], ecolor=Rrot_color, elinewidth=1.0,
                         capsize=2.5, capthick=1, linestyle='None', marker='None')
            ax.scatter([star_mass], [star_model_radius_rot], color=Rrot_color, marker='*', s=80, zorder=1)
            ax.errorbar(x=[star_mass], y=[star_model_radius_orb],
                        yerr=[[star_model_radius_orb_lower], [star_model_radius_orb_upper]], ecolor='red',
                        elinewidth=1.0, capsize=2.5, capthick=1, linestyle='None', marker='None')
            ax.scatter([star_mass], [star_model_radius_orb], color='red', marker='*', s=80)

        ax.scatter([], [], color=Empirical_color, edgecolor='#000000', linewidth=0.3, marker='*', s=110, label=r'Empirical R$_{\star}$')
        ax.scatter([], [], color=Rrot_color, marker='*', s=80, label=r'P$_{rot}$ ~ 6.6 d')
        ax.scatter([], [], color='red', marker='*', s=80, label=r'P$_{rot}$ ~ 9.0 d')
        ax.set_xlim(0, max(Baraffe_masses))
        ax.set_ylim(0.05, 3.75)
        ax.set_xlabel(r'Model Mass (M$_{\odot}$)')
        ax.set_ylabel(r'Model Radius (R$_{\odot}$)')
        ax.legend(loc='upper left', fontsize='small', framealpha=1.0, fancybox=False, frameon=False)
        ax.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
        plt.tight_layout()
        plt.savefig('/Users/lbiddle/Desktop/CITau_Lightcurves/Baraffe_Isochrones.pdf', dpi=300)
        plt.close()

        return


Calc = Calculations(Mstar, Mstar_err, Teff, Teff_err, Luminosity, Luminosity_err,
                    vsini, vsini_err, Prot, Prot_err, Porb, Porb_err, i, i_err, Age,
                    Age_err, Mplanet, Mplanet_err, ecc, ecc_err)

Calc.calc_Rstar_with_Period()
Calc.get_Baraffe_Model()
Calc.calc_Rstar_empirical()
Calc.calc_Corotation_Radius()
Calc.calc_Orbit_Distance()
Calc.calc_Hill_Radius()
Calc.plot_Baraffe_Isochrones()



# def calc_Tshock():
#     M_star = 0.9 * M_sun
#     R_star = 1.8 * R_sun
#     R_inner = 10. * R_star
#     gamma = 5. / 3.
#     v_ff_squared = np.abs((2. * G * M_star) / R_star) * np.abs(1. - R_star / R_inner)
#
#     v_ff = np.sqrt(np.abs((2. * G * M_star) / R_star))  # * np.abs(1. - R_star/R_inner))
#
#     T_shock1 = (3. / 16.) * ((u * 1.00784) / k_B) * v_ff_squared
#
#     T_shock2 = ((2. * m_p) / k_B) * ((gamma - 1.) / ((gamma + 1.) ** 2)) * v_ff_squared
