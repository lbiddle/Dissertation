import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataframe_ribas = pd.read_csv('/Users/lbiddle/Desktop/Plots_For_Dissertation/DiskFractions_Ribas.csv')
dataframe_hernandez = pd.read_csv('/Users/lbiddle/Desktop/Plots_For_Dissertation/DiskFractions_Hernandez.csv')

def plot_data(input_dataframe1, input_dataframe2):

    cluster1 = input_dataframe1['Cluster Name'].values
    Age1 = input_dataframe1['Age (Myr)'].values
    Age_err1 = input_dataframe1['Age (Myr) Err'].values
    Disk_frac1 = input_dataframe1['Disk Fraction'].values
    Disk_frac_err1 = input_dataframe1['Disk Fraction Err'].values

    cluster2 = input_dataframe2['Cluster Name'].values
    Age2 = input_dataframe2['Age (Myr)'].values
    Age_err2 = input_dataframe2['Age (Myr) Err'].values
    Disk_frac2 = input_dataframe2['Disk Fraction'].values
    Disk_frac_err2 = input_dataframe2['Disk Fraction Err'].values

    font_size = 'large'
    # colors = ['green', 'blue', 'red', 'purple', 'orange', 'black']
    colors = [plt.cm.rainbow(color_i) for color_i in np.linspace(0, 1, len(cluster1))]
    plt.close()
    fig = plt.figure(1, figsize=(5, 5), facecolor="#ffffff")  # , dpi=300)
    ax = fig.add_subplot(111)

    for cl_i, cl in enumerate(cluster1):
        ax.errorbar(x=Age1[cl_i], y=Disk_frac1[cl_i],
                    yerr=Disk_frac_err1[cl_i], xerr=Age_err1[cl_i], ecolor=colors[cl_i], elinewidth=2.5, capsize=3.0,
                    capthick=2, linestyle='None', marker='None', zorder=0)
        ax.scatter(Age1[cl_i], Disk_frac1[cl_i], color=colors[cl_i], edgecolor='#000000', s=np.pi*(5)**2, zorder=1)
        ax.text(12.5, 91.0-(cl_i*5.5), cl, fontsize='x-large', weight='bold', color=colors[cl_i],
                horizontalalignment='left')

    for clh_i, clh in enumerate(cluster2):
        color_element = np.where(cluster1 == clh)[0][0]
        color = colors[color_element]

        ax.errorbar(x=Age2[clh_i], y=Disk_frac2[clh_i],
                    yerr=Disk_frac_err2[clh_i], xerr=Age_err2[clh_i], ecolor=color, elinewidth=2.5, capsize=3.0,
                    capthick=2, linestyle='None', marker='None', zorder=0)
        ax.scatter(Age2[clh_i], Disk_frac2[clh_i], color=color, edgecolor='#000000', marker='s',  s=100, zorder=1)


    ax.set_xticks([0,2,4,6,8,10,12,14,16,18])
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Age (Myr)', fontsize=font_size)
    ax.set_ylabel('Disk Fraction (%)', fontsize=font_size)
    # ax.legend(loc='upper right', fontsize='small', framealpha=1.0, fancybox=False, frameon=False)
    ax.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
    plt.tight_layout()
    plt.savefig('/Users/lbiddle/Desktop/Plots_For_Dissertation/Chapter2_Figures/Disk_Fractions.pdf', dpi=300)
    plt.close()


plot_data(input_dataframe1=dataframe_ribas, input_dataframe2=dataframe_hernandez)