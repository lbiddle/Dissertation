import numpy as np
import matplotlib.pyplot as plt
import copy
import astropy.units as u
import pandas as pd
import os


def unique(input_list):
    unique_list = []
    for val in input_list:
        if val not in unique_list:
            unique_list.append(val)
    return unique_list
def clean_df(dataframe, type):
    if type == 'Target':
        target_dict = {}
        lats = np.degrees(dataframe['spotlats'].values)
        lons = np.degrees(dataframe['spotlons'].values)
        rads = np.degrees(dataframe['spotrads'].values)
        nspots = [len(lats)] * len(lats)
        areas = []
        for rad in rads:
            area = np.pi*rad**2
            areas.append(area)
        spot_area = [sum(areas)] * len(lats)
        # incl = np.degrees(dataframe['stellar rotation inclination'].values)
        incl = [60.0] * len(lats)

        target_dict['spotlats'] = lats
        target_dict['spotlons'] = lons
        target_dict['spotrads'] = rads
        target_dict['Nspots'] = nspots
        target_dict['total_spot_area'] = spot_area
        target_dict['incl'] = incl

        dataframe_out = pd.DataFrame(data=target_dict)

    if type == 'Candidate':
        candidate_dict = {}
        gens = dataframe['generation'].values
        lats = np.degrees(dataframe['coolspot lats'].values)
        lons = np.degrees(dataframe['coolspot lons'].values)
        rads = np.degrees(dataframe['coolspot radii'].values)

        nspots = []
        spot_area = []
        unique_gens = unique(gens)
        for gen_i, gen in enumerate(unique_gens):
            where_gen = np.where(gens == gen)[0]
            rads_i = np.array(rads)[where_gen]
            areas = np.pi*rads_i**2
            area = sum(areas)
            spotnum = len(where_gen)

            for row_i in range(len(where_gen)):
                spot_area.append(area)
                nspots.append(spotnum)

        incls = dataframe['stellar rotation inclination'].values
        # fitnesses = dataframe['fitness score'].values


        candidate_dict['generation'] = gens
        candidate_dict['spotlats'] = lats
        candidate_dict['spotlons'] = lons
        candidate_dict['spotrads'] = rads
        candidate_dict['Nspots'] = nspots
        candidate_dict['total_spot_area'] = spot_area
        candidate_dict['incl'] = incls
        #candidate_dict['fitness'] = fitnesses

        dataframe_out = pd.DataFrame(data=candidate_dict)


    return dataframe_out
def plot_fit_results(target_df,candidates_df, save):

    plot_text = {'generation': 'Generation',
                 'spotlats': 'Spot Latitude (deg)',
                 'spotlons': 'Spot Longitude (deg)',
                 'spotrads': 'Spot Radius (deg)',
                 'Nspots': 'Number of Spots',
                 'total_spot_area': r'Total Spot Area (deg$^{2}$)',
                 'incl': 'Inclination (deg)',
                 'fitness': 'Fitness Score',
                 }

    independent_variables = ['generation']  # , 'tpeak frac', 'S/N at peak']
    plot_subjects = ['incl', 'total_spot_area', 'spotrads', 'spotlons', 'spotlats', 'Nspots']

    num_plots = int(len(plot_subjects) * len(independent_variables))

    num_cols = len(independent_variables)
    num_rows = len(plot_subjects)

    vpad_base = 0.05
    vpad = vpad_base / (num_rows / 2.)
    hpad_base = 0.15
    hpad = 0.025

    ax_height = 1. / num_rows - vpad
    ax_width = (0.97 - (hpad_base + hpad)) / num_cols

    left_array = []
    bottom_array = []
    width_array = []
    height_array = []

    for column in range(len(independent_variables)):
        for row in range(len(plot_subjects)):
            left_array.append(column * (ax_width + hpad) + hpad_base)
            bottom_array.append(np.mod(row, len(plot_subjects)) * (ax_height + vpad / 2) + vpad_base)
            width_array.append(ax_width)
            height_array.append(ax_height)


    # import pdb; pdb.set_trace()

    plt.close()
    font_size = 'x-large'
    # colors = [plt.cm.Spectral(color_i) for color_i in np.linspace(0, 1, num_plots)]
    colors = [plt.cm.rainbow(color_i) for color_i in np.linspace(0, 1, num_plots)]
    fig = plt.figure(1, figsize=(6 * num_cols, 3 * num_rows), facecolor="#ffffff")  # , dpi=300)

    ith_term = 0
    for x_parameter_i, x_parameter in enumerate(independent_variables):
        for y_parameter_i, y_parameter in enumerate(plot_subjects):
            ax = fig.add_axes((left_array[ith_term], bottom_array[ith_term],
                               width_array[ith_term], height_array[ith_term]))

            ax.set_ylabel(plot_text[y_parameter],fontsize=font_size)
            if y_parameter_i == 0:
                ax.set_xlabel(plot_text[x_parameter], fontsize=font_size)
            if y_parameter_i != 0:
                ax.set_xticks([])

            xplot_candidate = candidates_df[x_parameter].values
            yplot_candidate = candidates_df[y_parameter].values

            ymin_candidate = min(yplot_candidate)
            ymin_target = min(target_df[y_parameter].values)
            ymax_candidate = max(yplot_candidate)
            ymax_target = max(target_df[y_parameter].values)

            verymin = min([ymin_target, ymin_candidate])
            verymax = max([ymax_target, ymax_candidate])

            ydiff = abs(verymax - verymin)
            ymin = verymin - 0.1 * ydiff
            ymax = verymax + 0.1 * ydiff

            if y_parameter == 'Nspots':
                ymin = 0

            for targ_value_i, targ_value in enumerate(target_df[y_parameter].values):
                ax.plot([min(xplot_candidate)-2, max(xplot_candidate)+2],
                        [target_df[y_parameter].values[targ_value_i], target_df[y_parameter].values[targ_value_i]],
                        color='#000000', lw=2, zorder=0)


            if (y_parameter == 'incl') or (y_parameter == 'total_spot_area') or (y_parameter == 'Nspots'):
                plot_type = 'line'
            else:
                plot_type = 'scatter'


            if plot_type == 'line':
                # for gen_i, gen in enumerate(unique(candidates_df[x_parameter].values)):
                #     for targ_value_i, targ_value in enumerate(target_df[y_parameter].values):
                #         ax.scatter(gen, targ_value, color='#000000', s=np.pi * (3) ** 2, zorder=0)
                #
                # gens = unique(candidates_df[x_parameter].values)
                # ax.plot(gens, target_df[y_parameter].values[0])

                # for targ_value_i, targ_value in enumerate(target_df[y_parameter].values):
                #     ax.plot([min(xplot_candidate), max(xplot_candidate)],
                #             [target_df[y_parameter].values[targ_value_i],target_df[y_parameter].values[targ_value_i]],
                #             color='#000000', lw=3)
                ax.plot(xplot_candidate, yplot_candidate, color=colors[y_parameter_i], lw=3, zorder=1)


            if plot_type == 'scatter':
                # for gen_i, gen in enumerate(unique(candidates_df[x_parameter].values)):
                #     for targ_value_i, targ_value in enumerate(target_df[y_parameter].values):
                #         ax.scatter(gen, targ_value, color='#000000', s=np.pi * (3) ** 2, zorder=0)
                ax.scatter(xplot_candidate, yplot_candidate, color=colors[y_parameter_i], edgecolors='#000000',
                           linewidths=1, s=np.pi * (5) ** 2, zorder=1)




            for gen_i, gen in enumerate(unique(candidates_df[x_parameter].values)):
                ax.plot([gen, gen], [ymin, ymax], color='#000000', alpha=0.15, lw=1, zorder=0)

            ax.set_ylim([ymin, ymax])
            ax.set_xlim([min(xplot_candidate)-1, max(xplot_candidate)+1])

            ax.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)

            ith_term += 1

    plt.savefig(save + 'candidate_progression.pdf', dpi=300)
    plt.close()

def arclen_dist(lat1, lon1, lat2, lon2):
    sphere_radius = 1

    dlon = abs(lon2 - lon1)
    dlat = abs(lat2 - lat1)

    x = np.sin(dlat/2.)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.)**2
    dist = 2.* sphere_radius * np.arcsin(np.sqrt(x))

    # dist = dist / (2.*np.pi*sphere_radius)

    dist = np.degrees(dist)

    return dist
def spot_area(radius):
    sphere_radius = 1

    lat1 = 0
    lat2 = 0
    lon1 = 0
    lon2 = radius

    dlon = abs(lon2 - lon1)
    dlat = abs(lat2 - lat1)

    x = np.sin(dlat/2.)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.)**2
    r_dist = 2.* sphere_radius * np.arcsin(np.sqrt(x))

    r_dist = np.degrees(r_dist)

    area = np.pi * r_dist**2

    # area = area / (4.*np.pi*sphere_radius**2)

    return area
def find_overlapping_spots(inlats, inlons, inrads):

    goodlats = []
    goodlons = []
    goodrads = []

    accounted_for = []

    for lat1_i in range(len(inlats)):
        lat1 = inlats[lat1_i]
        lon1 = inlons[lat1_i]
        rad1 = inrads[lat1_i]

        merged_spots = False
        for lat2_i in range(len(inlats)):
            lat2 = inlats[lat2_i]
            lon2 = inlons[lat2_i]
            rad2 = inrads[lat2_i]

            if (lat2_i != lat1_i) and (lat1 not in accounted_for) and (lat2 not in accounted_for):
                spot_dist = arclen_dist(lat1, lon1, lat2, lon2)
                if spot_dist < np.degrees(rad1) + np.degrees(rad2):
                    goodlats.append(np.mean([lat1, lat2]))
                    goodlons.append(np.mean([lon1, lon2]))
                    goodrads.append(np.mean([rad1, rad2]))

                    print('Merged Spots!')
                    merged_spots = True

                    accounted_for.append(lat1)
                    accounted_for.append(lat2)

        if merged_spots == False:
            goodlats.append(lat1)
            goodlons.append(lon1)
            goodrads.append(rad1)

    return goodlats, goodlons, goodrads

def get_data_for_MetaAnalysis(filedir, how_many_spots):

    print('\nCollecting Data For ' + str(int(how_many_spots)) + 'spot\n')

    files_in_dir = [f for f in os.listdir(filedir) if os.path.isfile(os.path.join(filedir, f))]

    relevant_string = str(int(how_many_spots)) + 'spot'

    relevant_candidate_files = []
    for c_file in files_in_dir:
        if (relevant_string in c_file) and ('candidate' in c_file):
            relevant_candidate_files.append(c_file)

    relevant_target_files = []
    for t_file in files_in_dir:
        if (relevant_string in t_file) and ('target' in t_file):
            relevant_target_files.append(t_file)


    top_candidates = {}

    ids = []
    area_differences = []
    spot_distances = []
    incl_differences = []
    lonely_spots = []
    lonely_spots_area = []

    lat_differences = []
    lon_differences = []

    erroneous_spots = []
    erroneous_spots_area = []


    for file_i, thisfile in enumerate(relevant_candidate_files):
        ids.append(int(thisfile[-5]))
        associated_targetnum = thisfile[-5]

        candidate_file_df = pd.read_csv(filedir + thisfile)
        top_candidate_df = candidate_file_df[candidate_file_df['generation'] == max(candidate_file_df['generation'])]

        for targfile in relevant_target_files:
            if targfile[-5] == associated_targetnum:
                target_file_df = pd.read_csv(filedir + targfile)

        t_lats = target_file_df['spotlats'].values
        t_lons = target_file_df['spotlons'].values
        t_rads = target_file_df['spotrads'].values
        c_lats = top_candidate_df['coolspot lats'].values
        c_lons = top_candidate_df['coolspot lons'].values
        c_rads = top_candidate_df['coolspot radii'].values


        for clat_i, clat in enumerate(c_lats):
            if isinstance(clat, str) == True:
                c_lat_str = clat.strip('[]')
                c_lats[clat_i] = float(c_lat_str)

            if isinstance(c_lons[clat_i], str) == True:
                c_lon_str = c_lons[clat_i].strip('[]')
                c_lons[clat_i] = float(c_lon_str)

            if isinstance(c_rads[clat_i], str) == True:
                c_rad_str = c_rads[clat_i].strip('[]')
                c_rads[clat_i] = float(c_rad_str)

        if len(c_lats) > 1:
            c_lats, c_lons, c_rads = find_overlapping_spots(inlats=c_lats, inlons=c_lons, inrads=c_rads)


        try:
            t_incl = target_file_df['incl'].values[0]
        except:
            t_incl = 60.0
        c_incl = top_candidate_df['stellar rotation inclination'].values[0]

        incl_differences.append(abs(t_incl - c_incl))


        c_spot_accounted_for = []
        t_spot_accounted_for = []


        t_lonely_spots = {'lat': [],
                          'lon': [],
                          'area': [],
                          }
        c_erroneous_spots = {'lat': [],
                             'lon': [],
                             'area': [],
                             }

        # import pdb; pdb.set_trace()

        for t_lat_i1, t_lat1 in enumerate(t_lats):

            if len(t_spot_accounted_for) >= len(c_lats):
                t_lonely_spots['lat'].append(t_lat1)
                t_lonely_spots['lon'].append(t_lons[t_lat_i1])
                t_lonely_spots['area'].append(spot_area(t_rads[t_lat_i1]))

            # too_many_iters = 0
            # while (t_lat_i1 not in t_spot_accounted_for) and (t_lat1 not in t_lonely_spots['lat']) and (len(t_spot_accounted_for) < len(c_lats)):
            #
            #     if too_many_iters > 10:
            #         print('too many iters')
            #         import pdb; pdb.set_trace()
            #     too_many_iters += 1

            distance_tracker = 1000000

            for t_lat_i, t_lat in enumerate(t_lats):
                if t_lat_i not in t_spot_accounted_for:

                    for c_lat_i, c_lat in enumerate(c_lats):
                        if c_lat_i not in c_spot_accounted_for:

                            radial_dist_nosignflip = arclen_dist(lat1=t_lat, lon1=t_lons[t_lat_i], lat2=c_lat, lon2=c_lons[c_lat_i])
                            radial_dist_signflip = arclen_dist(lat1=t_lat, lon1=t_lons[t_lat_i], lat2=-c_lat, lon2=c_lons[c_lat_i])

                            if (radial_dist_signflip <= np.radians(10.)) and (abs(t_lons[t_lat_i] - c_lons[c_lat_i]) <= np.radians(10)):
                                radial_dist = radial_dist_signflip
                                c_lats[c_lat_i] = -c_lat
                            else:
                                radial_dist = radial_dist_nosignflip

                            if (radial_dist < distance_tracker) and (radial_dist not in spot_distances):
                                distance_tracker = radial_dist
                                closest_spot_c = c_lat_i
                                closest_spot_t = t_lat_i

                        else:
                            continue
                else:
                    continue

            # import pdb; pdb.set_trace()

            if abs(t_lons[closest_spot_t] - c_lons[closest_spot_c]) >= np.pi/2:
                t_lonely_spots['lat'].append(t_lats[closest_spot_t])
                t_lonely_spots['lon'].append(t_lons[closest_spot_t])
                t_lonely_spots['area'].append(spot_area(t_rads[closest_spot_t]))
            else:
                spot_distances.append(distance_tracker)

                t_area = spot_area(t_rads[closest_spot_t])
                c_area = spot_area(c_rads[closest_spot_c])
                area_difference = abs(t_area - c_area)
                area_differences.append(area_difference)

                lat_differences.append(np.degrees(abs(t_lats[closest_spot_t] - c_lats[closest_spot_c])))
                lon_differences.append(np.degrees(abs(t_lons[closest_spot_t] - c_lons[closest_spot_c])))

                c_spot_accounted_for.append(closest_spot_c)
                t_spot_accounted_for.append(closest_spot_t)



        for t_lat_i, t_lat in enumerate(t_lats):
            if t_lat_i not in t_spot_accounted_for:
                t_lonely_spots['lat'].append(t_lats[t_lat_i])
                t_lonely_spots['lon'].append(t_lons[t_lat_i])
                t_lonely_spots['area'].append(spot_area(t_rads[t_lat_i]))

        for c_lat_i, c_lat in enumerate(c_lats):
            if c_lat_i not in c_spot_accounted_for:
                c_erroneous_spots['lat'].append(c_lats[c_lat_i])
                c_erroneous_spots['lon'].append(c_lons[c_lat_i])
                c_erroneous_spots['area'].append(spot_area(c_rads[c_lat_i]))

        if len(t_lonely_spots['lat']) > 0:
            lonely_spots.append(len(t_lonely_spots['lat']))
            lonely_spots_area.append(sum(t_lonely_spots['area']))
        else:
            lonely_spots.append(float('nan'))
            lonely_spots_area.append(float('nan'))

        if len(c_erroneous_spots['lat']) > 0:
            erroneous_spots.append(len(c_erroneous_spots['lat']))
            erroneous_spots_area.append(sum(c_erroneous_spots['area']))
        else:
            erroneous_spots.append(float('nan'))
            erroneous_spots_area.append(float('nan'))

        # import pdb; pdb.set_trace()

    top_candidates['lat differences'] = lat_differences
    top_candidates['lon differences'] = lon_differences
    top_candidates['area differences'] = area_differences
    top_candidates['spot distances'] = spot_distances
    top_candidates['incl differences'] = incl_differences
    top_candidates['erroneous spots'] = erroneous_spots
    top_candidates['erroneous spot area'] = erroneous_spots_area
    top_candidates['lonely spots'] = lonely_spots
    top_candidates['lonely spot area'] = lonely_spots_area

    # import pdb; pdb.set_trace()


    return top_candidates
def plot_Meta(meta, save):

    plot_text = {'lat differences': 'Latidude Diff (deg)',
                 'lon differences': 'Longitude Diff (deg)',
                 'area differences': r'Spot Area Diff (deg$^{2}$)',
                 'spot distances': 'Spot Distance (deg)',
                 'incl differences': 'Inclination (deg)',
                 'lonely spots': '# Lonely Spots',
                 'lonely spot area': r'Lonely Spot Area (deg$^{2}$)',
                 'erroneous spots': '# Erroneous Spots',
                 'erroneous spot area': r'Erroneous Spot Area (deg$^{2}$)',
                 }

    nbins = 12

    for key in meta[0].keys():

        print(key)

        full_key = []

        plt.close()
        font_size = 'large'
        # colors = [plt.cm.Spectral(color_i) for color_i in np.linspace(0, 1, num_plots)]
        # colors = [plt.cm.rainbow(color_i) for color_i in np.linspace(0, 1, num_plots)]
        # colors = [plt.cm.viridis(color_i) for color_i in np.linspace(0, 1, len(meta))]
        colors = ['#99d25c', '#3e6d8b', '#b30086']  #  '#461f6d']
        fig = plt.figure(1, figsize=(3.5*len(meta),4), facecolor="#ffffff")  # , dpi=300)

        for meta_i in range(len(meta)):

            ax = fig.add_subplot(1,len(meta),meta_i+1)
            ax.set_title(str(meta_i+1) + ' Spot', fontsize=font_size)

            if meta_i == 0:
                ax.set_ylabel('Counts',fontsize=font_size)
            ax.set_xlabel(plot_text[key], fontsize=font_size)
            ax.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)


            if np.isnan(np.nanmax(meta[meta_i][key])) == False:

                if (key == 'lonely spots') or (key == 'erroneous spots'):
                    bins = np.arange(0,np.nanmax(meta[meta_i][key])+2,1) - 0.5
                    ax.hist(meta[meta_i][key], bins=bins, color=colors[meta_i], edgecolor='#000000', linewidth=2)
                else:
                    ax.hist(meta[meta_i][key], bins=nbins, align='mid', color=colors[meta_i], edgecolor='#000000', linewidth=2)


                for value in meta[meta_i][key]:
                    full_key.append(value)



        plt.tight_layout()
        plt.savefig(save + 'SpotsMeta/' + key + '.pdf', dpi=300)
        plt.close()




        fig = plt.figure(1, figsize=(4, 4), facecolor="#ffffff")
        ax = fig.add_subplot(111)
        ax.set_ylabel('Counts', fontsize=font_size)
        ax.set_xlabel(plot_text[key], fontsize=font_size)
        ax.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)


        if (key == 'lonely spots') or (key == 'erroneous spots'):
            bins = np.arange(0, np.nanmax(full_key) + 2, 1) - 0.5
            ax.hist(full_key, bins=bins, color=colors[0], edgecolor='#000000', linewidth=2)
        else:
            ax.hist(full_key, bins=nbins, align='mid', color=colors[0], edgecolor='#000000', linewidth=2)

        plt.tight_layout()
        plt.savefig(save + 'FullMeta/' + key + '_full.pdf', dpi=300)
        plt.close()



file_directory = '/Users/lbiddle/Dropbox/Dissertation_Figures/GAStuff/'

meta1 = get_data_for_MetaAnalysis(filedir=file_directory, how_many_spots=1)
meta2 = get_data_for_MetaAnalysis(filedir=file_directory, how_many_spots=2)
meta3 = get_data_for_MetaAnalysis(filedir=file_directory, how_many_spots=3)


save_here = '/Users/lbiddle/Desktop/Plots_For_Dissertation/Chapter4_Figures/'

plot_Meta(meta=[meta1,meta2,meta3], save=save_here)




do_single_run_analysis = False
if do_single_run_analysis == True:
    target_file = '/Users/lbiddle/Dropbox/Dissertation_Figures/GAStuff/spotlocs_target_1spot-1.csv'
    candidate_file = '/Users/lbiddle/Dropbox/Dissertation_Figures/GAStuff/top_candidate_pars_1spot-1.csv'

    target_input = pd.read_csv(target_file)
    candidate_input = pd.read_csv(candidate_file)

    target = clean_df(dataframe=target_input, type='Target')
    candidates = clean_df(dataframe=candidate_input, type='Candidate')

    plot_fit_results(target_df=target,candidates_df=candidates)








