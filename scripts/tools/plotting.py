# -*- coding: utf-8 -*-
"""Plotting helpers: the reordered tab20 colormap, the PID-weight heatmap, the
`mpl` data/MC plotting class, and fit-projection plots.

Split out of utilities.py. Imports only go one way: this module imports from
utilities.py, never the reverse.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from uncertainties import ufloat
import uncertainties.unumpy as unp

from utilities import (
    analysis_variables,
    apply_eventByEvent_weight,
    draw_mc_band,
    excess_variance,
    get_weights,
    mismodeling_label,
    pid_toy_histograms,
    poisson_error,
    round_uarray,
)

##################################### Plotting #################################
import matplotlib.colors as mcolors

######## define my colormap ########
# Original tab20 colors
original_colors = plt.cm.tab20.colors
# New order for the colors
new_order_indices = [0,1,2,3,16,5,10,4,15,17,19,8,18,7,6,12,13]
# Create a new ordered list of colors
reordered_colors = [original_colors[i] for i in new_order_indices]
# Add the rest of the colors that are not explicitly ordered
remaining_indices = [i for i in range(len(original_colors)) if i not in new_order_indices]
reordered_colors.extend([original_colors[i] for i in remaining_indices])
# Create a new colormap
my_cmap = mcolors.ListedColormap(reordered_colors, name='reordered_tab20')


def plot_PID_weight_heatmap_mpl(
    save_path: str = '',
    variable: str = 'electronIDNN',
    threshold: float = 0.9,
    query: str = 'data_MC_ratio',
    widthFactor: float = 1,
    totErr: bool = True,
    fontsize: int = 10,
    csv_path: str = '/home/belle/zhangboy/inclusive_R_D/MC16_sys_tables/MC16_pid_tables/e_efficiency_MC16rd_run1_TwophotonEe.csv',
    charge: int | None = None,
    plot_graph: bool = False,
    max_zlim: int = 10,
):
    """
    Matplotlib-only version of PID correction heatmap.

    Parameters
    ----------
    charge : None, 1, or -1
        If None, plot both charges.
        If 1, plot only positive charge.
        If -1, plot only negative charge.

    plot_graph : bool
        If False, only plot the table heatmap.
        If True, also plot a simple graph view of heatmap_data.
    """

    df = pd.read_csv(csv_path)

    df_filtered = df[
        (df['variable'] == variable) &
        (df['threshold'] == threshold)
    ].copy()

    if charge is not None:
        if charge not in [-1, 1]:
            raise ValueError("charge must be None, 1, or -1")

        if charge == 1:
            df_filtered = df_filtered[df_filtered['charge_min'] == 0].copy()
            charge_bins = [0, 2]
            charge_labels = ['+']
        else:
            df_filtered = df_filtered[df_filtered['charge_min'] == -2].copy()
            charge_bins = [-2, 0]
            charge_labels = ['-']
    else:
        charge_bins = [-2, 0, 2]
        charge_labels = ['-', '+']

    p_bins = sorted(
        np.unique(
            np.concatenate([
                df_filtered['p_min'].dropna().unique(),
                df_filtered['p_max'].dropna().unique()
            ])
        )
    )

    cosTheta_bins = sorted(
        np.unique(
            np.concatenate([
                df_filtered['cosTheta_min'].dropna().unique(),
                df_filtered['cosTheta_max'].dropna().unique()
            ])
        )
    )

    n_charge = len(charge_bins) - 1
    n_p = len(p_bins) - 1
    n_cosTheta = len(cosTheta_bins) - 1

    print(f'Number of p bins: {n_p}, {p_bins}')
    print(f'Number of cosTheta bins: {n_cosTheta}, {cosTheta_bins}')

    if n_p < 1 or n_cosTheta < 1:
        raise ValueError(
            'Not enough unique p or cosTheta bin edges. '
            'Check p_min, p_max, cosTheta_min, cosTheta_max values.'
        )

    heatmap_data = np.full((n_p, n_cosTheta * n_charge), np.nan)
    annotation_data = np.full((n_p, n_cosTheta * n_charge), '', dtype=object)

    for _, row in df_filtered.iterrows():
        p_idx = p_bins.index(row['p_min'])
        cosTheta_idx = cosTheta_bins.index(row['cosTheta_min'])

        if charge is None:
            charge_idx = charge_bins.index(row['charge_min'])
        else:
            charge_idx = 0

        col_idx = cosTheta_idx * n_charge + charge_idx

        value = row[query]
        heatmap_data[p_idx, col_idx] = value

        if totErr:
            stat_err = row.get('data_MC_uncertainty_stat_up', 0.0)
            sys_err = row.get('data_MC_uncertainty_sys_up', 0.0)
            total_err = np.sqrt(stat_err**2 + sys_err**2)
            annotation_data[p_idx, col_idx] = f'{value:.2f}\n± {total_err:.2f}'
        else:
            annotation_data[p_idx, col_idx] = f'{value:.2f}'

    # -------------------------
    # Plot the table heatmap
    # -------------------------
    fig, ax = plt.subplots(figsize=(14 * widthFactor, 12))
    im = ax.imshow(
        heatmap_data,
        aspect='auto',
        origin='lower',
        cmap='coolwarm'
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(query, fontsize=fontsize + 10)

    ax.set_xticks(np.arange(-0.5, heatmap_data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, heatmap_data.shape[0], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            if not np.isnan(heatmap_data[i, j]):
                ax.text(
                    j, i,
                    annotation_data[i, j],
                    ha='center',
                    va='center',
                    fontsize=fontsize
                )

    cosTheta_sub = charge_labels * n_cosTheta

    ax.set_xticks(np.arange(n_cosTheta * n_charge))
    ax.set_xticklabels(cosTheta_sub, rotation=0, fontsize=fontsize)

    p_edge_positions = np.arange(len(p_bins)) - 0.5
    ax.set_yticks(p_edge_positions)
    ax.set_yticklabels([f'{x:.2g}' for x in p_bins], rotation=0, fontsize=fontsize)
    ax.set_ylim(-0.5, n_p - 0.5)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())

    cosTheta_edge_positions = np.arange(len(cosTheta_bins)) * n_charge - 0.5
    cosTheta_edge_labels = [f'{x:.3g}' for x in cosTheta_bins]

    ax2.set_xticks(cosTheta_edge_positions)
    ax2.set_xticklabels(cosTheta_edge_labels, rotation=0, fontsize=fontsize)
    ax2.spines['bottom'].set_position(('outward', 35))

    ax2.tick_params(
        axis='x',
        bottom=True,
        labelbottom=True,
        top=False,
        labeltop=False,
        labelsize=fontsize,
        direction='in',
        length=4,
        pad=2
    )

    tableType = 'PID'
    table_filename = csv_path.split('/')[-1]
    if 'eff' in table_filename:
        tableType = 'Efficiency'
    elif 'fake' in table_filename:
        true_pid = table_filename.split('_')[0]
        tableType = f'{true_pid} Fake Rate'

    if 'run1' in table_filename:
        dataset = 'Run1'
    elif 'run2' in table_filename:
        dataset = 'Run2'
    else:
        dataset = 'Run1 + Run2'

    charge_title = ''
    if charge == 1:
        charge_title = ', charge +'
    elif charge == -1:
        charge_title = ', charge -'

    ax.set_title(
        f'{dataset} {tableType} Table ({variable} > {threshold}{charge_title})',
        fontsize=fontsize + 10
    )
    ax.set_ylabel('p [GeV]', labelpad=14, fontsize=fontsize + 10)
    ax.set_xlabel(r'$\cos\theta$', labelpad=50, fontsize=fontsize + 10)

    # plt.tight_layout()

    if save_path != '':
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()
    plt.close()

    # -------------------------
    # Optional 2D/3D bar plot
    # -------------------------
    if plot_graph:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        if charge is None:
            charges_to_plot = [(-2, '-'), (0, '+')]
        elif charge == -1:
            charges_to_plot = [(-2, '-')]
        elif charge == 1:
            charges_to_plot = [(0, '+')]

        for this_charge_min, this_charge_label in charges_to_plot:
            graph_data = np.full((n_p, n_cosTheta), np.nan)

            df_charge = df_filtered[df_filtered['charge_min'] == this_charge_min]

            for _, row in df_charge.iterrows():
                p_idx = p_bins.index(row['p_min'])
                cosTheta_idx = cosTheta_bins.index(row['cosTheta_min'])
                graph_data[p_idx, cosTheta_idx] = row[query]

            # Bin centers and widths
            p_centers = 0.5 * (np.array(p_bins[:-1]) + np.array(p_bins[1:]))
            cos_centers = 0.5 * (np.array(cosTheta_bins[:-1]) + np.array(cosTheta_bins[1:]))

            p_widths = np.diff(p_bins)
            cos_widths = np.diff(cosTheta_bins)

            xpos, ypos = np.meshgrid(cos_centers, p_centers)
            dx, dy = np.meshgrid(cos_widths, p_widths)

            zpos = np.zeros_like(xpos)
            dz = graph_data

            valid = ~np.isnan(dz)

            fig = plt.figure(figsize=(14 * widthFactor, 8))
            ax = fig.add_subplot(111, projection='3d')

            ax.bar3d(
                xpos[valid] - dx[valid] / 2,
                ypos[valid] - dy[valid] / 2,
                zpos[valid],
                dx[valid],
                dy[valid],
                dz[valid],
                shade=True,
                alpha=0.85
            )

            ax.set_xlabel(r'$\cos\theta$', labelpad=10, fontsize=fontsize + 2)
            ax.set_ylabel('p [GeV]', labelpad=10, fontsize=fontsize + 2)
            ax.set_zlabel(query, labelpad=10, fontsize=fontsize + 2)
            ax.set_zlim(0, max_zlim)

            ax.set_title(
                f'{dataset} {tableType} 2D Bar Plot, charge {this_charge_label} ({variable} > {threshold})',
                fontsize=fontsize + 5,y=1,
            )

            # ax.view_init(elev=25, azim=-55)

            # plt.tight_layout()
            plt.show()
            plt.close()


    

class mpl:
    def __init__(self, mc_samples, data=None, background_hatches=None):
        """Create a plotting helper.

        Parameters
        ----------
        mc_samples : dict
            Mapping from component names to their data frames.
        data : pandas.DataFrame, optional
            The observed data sample.
        background_hatches : sequence of str or dict, optional
            Hatch patterns for components whose names start with ``"bkg_"``.
            A dictionary can be used to select patterns by component name; a
            sequence is applied in plotting order.  By default, each background
            component receives a different pattern.  Non-background components
            are left solid.
        """
        self.samples = mc_samples
        self.data = data
        self.colors = my_cmap.colors*2
        # sort the components to plot in order of fitted templates_project size
        self.sorted_order = ['bkg_fakeD',                        'bkg_continuum',    
                             'bkg_combinatorial',                'bkg_hadronicB_secondaryL',
                             'bkg_fakeL',                        'bkg_fakeTracks',    'bkg_fakeL_Tracks',
                             r'$D\ell\nu$_gap',
                             r'$D^{\ast\ast}\ell\nu$_narrow',    r'$D^{\ast\ast}\ell\nu$_broad', r'$D^{\ast\ast}\tau\nu$',
                             r'$D^\ast\ell\nu$',                 r'$D\ell\nu$',
                             r'$D^\ast\tau\nu$',                 r'$D\tau\nu$',
#                              'DSemiB_ellPri',  'DSemiB_ellSec',  'DHad1Charm_ellPri',
#                              'DHad1Charm_ellSec', 'DHad2Charm_ellPri', 'DHad2Charm_ellSec',
                             'SemileptonicB2D_PrimaryLepton', 'HadronicB2D_SecondaryLepton',
                             'SemileptonicB2D_SecondaryLepton + HadronicB2D_PrimaryLepton',
                             'BBbar_measured_hadronic',          'BBbar_semileptonic', 
                             'BBbar_unmeasured:2-body',          'BBbar_unmeasured:3-body',
                             'BBbar_unmeasured:4-body',          'BBbar_unmeasured:5-body', 
                             'BBbar_unmeasured:6-body',          'BBbar_unmeasured:7-body',
                             'BBbar_unmeasured:5+-body',
                             'nMC_KL:0', 'nMC_KL:1', 'nMC_KL:2', 'nMC_KL:2+']

        self.bkg = self.sorted_order[:6]
        self.norm = [r'$D\ell\nu$_gap', r'$D^{\ast\ast}\ell\nu$_narrow', r'$D^{\ast\ast}\ell\nu$_broad',
                     r'$D^\ast\ell\nu$',r'$D\ell\nu$']
        self.sig = [r'$D^{\ast\ast}\tau\nu$',r'$D^\ast\tau\nu$',r'$D\tau\nu$']

        self.var_name_dictionary = {'B0_recMissM2': '$M_{miss}^2$    [$GeV^2/c^4$]',
                                    'p_D_l': r'$|p_D| + |p_{\ell}|$    [GeV/c]',}

        background_names = [name for name in self.sorted_order if name.startswith('bkg_')]
        if background_hatches is None:
            background_hatches = ('///', '\\\\', 'xxx', '---', '+++', 'ooo', '...')
        if isinstance(background_hatches, dict):
            self.background_hatches = background_hatches.copy()
        else:
            if not background_hatches:
                raise ValueError('background_hatches must contain at least one pattern')
            self.background_hatches = {
                name: background_hatches[i % len(background_hatches)]
                for i, name in enumerate(background_names)
            }

    def _hist_style(self, component):
        """Return the fill style for an MC component histogram."""
        hatch = self.background_hatches.get(component)
        if hatch is None:
            return {}
        return {'hatch': hatch, 'edgecolor': 'black', 'linewidth': 0.8}
       
    
    def statistics(self, df=None, hist=None, count_only=False):
        if df is not None:
            counts = df.count()
            mean = df.mean()
            std = df.std()
        
        if hist is not None:
            bin_counts, bin_edges = hist
            
            if sum(bin_counts)==0:
                counts = 0
                mean = 0
                std = 0
            else: 
                counts = np.sum(bin_counts).round(0).astype(int)
                
                # Step 1: Calculate bin centers
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                
                # Step 2: Calculate the weighted mean
                mean = np.average(bin_centers, weights=bin_counts)
                
                # Step 3: Calculate the weighted variance
                variance = np.average((bin_centers - mean)**2, weights=bin_counts)
                
                # Step 4: Use the uncertainties package's sqrt if variance has uncertainties
                std = poisson_error(variance)
        if count_only:
            return f'{counts=:d}'
        else:
            return f'''{counts=:d} \n{mean=:.3f} \n{std=:.3f}'''
    
    def plot_pie(self, cut='1.855<D_M<1.885'):
        # Plotting the pie chart
        fakeL_Tracks = pd.concat([self.samples['bkg_fakeL'], self.samples['bkg_fakeTracks']], ignore_index=True)
        samples_to_plot = {**self.samples, 'bkg_fakeL_Tracks': fakeL_Tracks}
        samples_to_plot.pop('bkg_fakeL')
        samples_to_plot.pop('bkg_fakeTracks')
        sizes1 = [len(self.samples[comp].query(cut)) for comp in self.sorted_order if comp in self.samples.keys()]
        sizes2 = [len(samples_to_plot[comp].query(cut)) for comp in self.sorted_order if comp in samples_to_plot.keys()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        
        labels = [name for name in self.sorted_order if name in samples_to_plot.keys()]
        indices = [self.sorted_order.index(name) for name in labels]
        colors = [self.colors[i] for i in indices]
        
        ax1.pie(sizes2, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
        ax1.set_title(f'All components in the region {cut[:15]=}')
        ax2.pie(sizes1[:6], labels=self.sorted_order[:6], autopct='%1.1f%%', startangle=140, colors=self.colors)
        ax2.set_title(f'BKG components in the region {cut[:15]=}')
        plt.tight_layout()
        plt.show()
        
    
    def plot_data_1d(self, bins, ax, hist=None, sub_df=None, variable=None, cut=None,
                 sig_mask=False, scale=1, name='Data', density=False,
                 apply_eventByEvent_correction=False,
                 eventByEvent_weight_col='total_PID_weight'):
        # Provide either variable or hist
        if variable: # requires sub_df
            # Apply signal mask if requested
            if sig_mask:
                s_mask = 'D_M<1.855 or D_M>1.885'
                data = sub_df.query(s_mask) if sub_df is not None else self.data.query(s_mask)
            else:
                data = sub_df if sub_df is not None else self.data

            selected_data = data.query(cut) if cut else data
            var_col = selected_data[variable]
            # Apply scaling if requested
            weight = get_weights(selected_data, scale)
            if apply_eventByEvent_correction:
                weight = apply_eventByEvent_weight(
                    selected_data, weight, eventByEvent_weight_col
                )

            # Compute histogram with weights
            counts, _ = np.histogram(var_col, bins=bins, weights=weight)
            staterr_squared, _ = np.histogram(var_col, bins=bins,weights=weight**2)
            staterror = poisson_error(staterr_squared)

            # Normalize to density if requested
            if density:
                bin_widths = np.diff(bins)
                integral = np.sum(counts * bin_widths)
                if integral > 0:
                    factor = 1.0 / integral
                    # np.histogram returns integer counts when the input weights
                    # are integer-valued.  Do not normalize in place: NumPy
                    # cannot cast the resulting floating-point density back to
                    # that integer array.
                    counts = np.asarray(counts, dtype=float) * factor
                    staterror = np.asarray(staterror, dtype=float) * factor

            label = f'{name} \n{self.statistics(df=var_col)}\n cut_eff={(len(var_col)/len(data)):.3f}'
            data_counts = unp.uarray(counts, staterror)

        else:
            # If hist is provided directly (data_counts as unp.uarray), we handle it similarly
            data_counts = hist
            counts = unp.nominal_values(data_counts)

            if density and hist is not None:
                # Normalize the provided histogram to density if needed
                staterror = unp.std_devs(data_counts)
                bin_widths = np.diff(bins)
                integral = np.sum(counts * bin_widths)
                if integral > 0:
                    factor = 1.0 / integral
                    counts *= factor
                    staterror *= factor
                    data_counts = unp.uarray(counts, staterror)

            label = f'{name} \n{self.statistics(hist=[counts,bins])}'

        bin_centers = (bins[:-1] + bins[1:]) / 2
        data_val = unp.nominal_values(data_counts)
        data_err = unp.std_devs(data_counts)

        if ax is not None:
            # Plot using errorbar to show data with uncertainties
            ax.errorbar(x=bin_centers, y=data_val, yerr=data_err, fmt='ok',label=label)
        
        return data_counts


    def plot_mc_1d(self, bins, ax, sub_df=None, sub_name=None, variable=None, cut=None,
                  weights={}, mask=[], legend='full', density=False,
                  apply_eventByEvent_correction=False, eventByEvent_weight_col='total_PID_weight',
                  mc_uncertainty=False):
        """mc_uncertainty: add the PID-correction uncertainty (spread over the variation
        columns of eventByEvent_weight_col, see apply_pid_corrections) to the MC
        statistical error of the returned histogram, and draw the total MC uncertainty
        (stat + PID) as a band. The MC statistical error is always included."""
        pid_toys = None   # (n_bins, n_toys): toy histograms of the whole stack

        def add_pid_toys(sample, base_weight, scale=1.0):
            # accumulate toy histograms; the same toy index is used for every component,
            # so the PID correlations between components are kept
            nonlocal pid_toys
            if not mc_uncertainty or not apply_eventByEvent_correction or len(sample) == 0:
                return
            toys = pid_toy_histograms(sample, variable, bins, base_weight, eventByEvent_weight_col)
            if toys is None:
                return
            toys = toys * scale
            if pid_toys is not None and pid_toys.shape != toys.shape:
                raise ValueError('MC components have different numbers of PID variations')
            pid_toys = toys if pid_toys is None else pid_toys + toys

        def normalize_to_density(counts, staterror, bins, integral=None):
            # Scale both values and errors so the uncertainty array stays
            # consistent with the density shown in the plot.
            if density:
                bin_widths = np.diff(bins)
                if integral is None:
                    integral = np.sum(counts * bin_widths)
                if integral > 0:
                    counts = counts / integral
                    staterror = staterror / integral
            return counts, staterror


        if sub_df is not None:
            sample = sub_df.query(cut) if cut else sub_df

            if sub_name in weights:
                weight = get_weights(sample, weights.get(sub_name, 1) )
            elif 'all_mc' in weights:
                weight = get_weights(sample, weights.get('all_mc', 1) )
            else:
                weight = get_weights(sample, weights.get(sub_name, 1) )

            base_weight = weight  # without the PID correction
            if apply_eventByEvent_correction: # update the weight
                weight = apply_eventByEvent_weight(sample, weight, eventByEvent_weight_col)

            (counts, _) = np.histogram(sample[variable], bins=bins,weights=weight)
            (staterr_squared, _) = np.histogram(sample[variable], bins=bins,weights=weight**2)
            staterror = poisson_error(staterr_squared)

            raw_integral = np.sum(counts * np.diff(bins))
            counts, staterror = normalize_to_density(counts, staterror, bins)
            add_pid_toys(sample, base_weight,
                         1.0 / raw_integral if density and raw_integral > 0 else 1.0)

            if ax is not None:
                if legend== 'simple_color':
                    label = sub_name
                elif legend == 'count':
                    label = (f'{sub_name} \n{self.statistics(df=sample[variable],count_only=True)} '
                           f'\n cut_eff={(len(sample)/len(sub_df)):.3f}')
                elif legend=='full':
                    label = (f'{sub_name} \n{self.statistics(df=sample[variable],count_only=False)} '
                           f'\n cut_eff={(len(sample)/len(sub_df)):.3f}')
                
                color_index = (
                    self.sorted_order.index(sub_name)
                    if sub_name in self.sorted_order else 0
                )
                ax.hist(bins[:-1], bins, weights=counts, color=self.colors[color_index],
                        label=label, **self._hist_style(sub_name))

            sample_counts = unp.uarray(counts, staterror)
            bottom = sample_counts

        else:
            bottom = unp.uarray(np.zeros(len(bins)-1), np.zeros(len(bins)-1))
            density_integral = None
            if density:
                density_integral = 0.0
                for name in self.sorted_order:
                    if name not in self.samples or name in mask:
                        continue
                    sample = self.samples[name].query(cut) if cut else self.samples[name]
                    if len(sample) == 0:
                        continue
                    scale = weights.get(name, weights.get('all_mc', 1))
                    weight = get_weights(sample, scale)
                    if apply_eventByEvent_correction:
                        weight = apply_eventByEvent_weight(
                            sample, weight, eventByEvent_weight_col
                        )
                    raw_counts, _ = np.histogram(
                        sample[variable], bins=bins, weights=weight
                    )
                    density_integral += np.sum(raw_counts * np.diff(bins))

            for i, name in enumerate(self.sorted_order):
                if name not in self.samples.keys():
                    continue
                
                sample = self.samples[name].query(cut) if cut else self.samples[name]

                if len(sample) == 0 or name in mask:
                    continue
                
                if name in weights:
                    weight = get_weights(sample, weights.get(name, 1) )
                elif 'all_mc' in weights:
                    weight = get_weights(sample, weights.get('all_mc', 1) )
                else:
                    weight = get_weights(sample, weights.get(name, 1) )

                base_weight = weight  # without the PID correction
                if apply_eventByEvent_correction: # update the weight
                    weight = apply_eventByEvent_weight(sample, weight, eventByEvent_weight_col)

                (counts, _) = np.histogram(sample[variable], bins=bins,weights=weight)
                (staterr_squared, _) = np.histogram(sample[variable], bins=bins,weights=weight**2)
                staterror = poisson_error(staterr_squared)
                

                # Normalize if density=True
                counts, staterror = normalize_to_density(
                    counts, staterror, bins, density_integral
                )
                add_pid_toys(sample, base_weight,
                             1.0 / density_integral if density and density_integral else 1.0)
                b = unp.nominal_values(bottom)
                
                if ax is not None:
                    if legend== 'simple_color':
                        label = name
                    elif legend == 'count':
                        label = (f'{name} \n{self.statistics(df=sample[variable],count_only=True)} '
                               f'\n cut_eff={(len(sample)/len(self.samples[name])):.3f}')
                    elif legend=='full':
                        label = (f'{name} \n{self.statistics(df=sample[variable],count_only=False)} '
                               f'\n cut_eff={(len(sample)/len(self.samples[name])):.3f}')
                    ax.hist(bins[:-1], bins, weights=counts, bottom=b, color=self.colors[i],
                            label=label, **self._hist_style(name))

                sample_counts = unp.uarray(counts, staterror)
                bottom += sample_counts

        # full MC covariance of this stack (MC stat; + PID below) for the chi2
        self.mc_model_cov = np.diag(unp.std_devs(bottom) ** 2)
        if mc_uncertainty:
            nominal, stat_err = unp.nominal_values(bottom), unp.std_devs(bottom)
            if pid_toys is None:
                print(f"Warning: no PID variations of '{eventByEvent_weight_col}' found "
                      f"(needs apply_eventByEvent_correction=True and apply_pid_corrections "
                      f"with n_variations > 0); showing the MC statistical uncertainty only")
                pid_err = np.zeros_like(nominal)
                self.mc_pid_cov = None
            else:
                pid_err = pid_toys.std(axis=1, ddof=1)
                self.mc_pid_cov = np.atleast_2d(np.cov(pid_toys))  # bin-to-bin PID covariance
                self.mc_model_cov = self.mc_model_cov + self.mc_pid_cov
            bottom = unp.uarray(nominal, np.hypot(stat_err, pid_err))
            if ax is not None:
                draw_mc_band(ax, bins, nominal, unp.std_devs(bottom),
                             label='MC unc.\n(stat + PID)' if pid_toys is not None else 'MC unc. (stat)')

        return bottom
    
    
    def plot_mc_1d_overlaid(self,variable,bins,cut=None,mask=[],show_only=None,density=False, errorbars=False, 
                            weights={},figsize=(8,5),legend_nc=3,text_fs=14,title=None):
        if show_only is not None:
            # this will overwrite the mask argument
            if show_only == 'sig_and_gap':
                mask = self.bkg + [r'$D^\ast\ell\nu$',r'$D\ell\nu$']
            elif show_only == 'sig':
                mask = self.bkg + self.norm
            elif show_only == 'norm':
                mask = self.bkg + self.sig
            elif show_only == 'bkg':
                mask = self.norm + self.sig
            elif isinstance(show_only, list):
                mask = [x for x in self.sorted_order if x not in show_only]
            else:
                print('Warning: show_only accepts sig, norm, bkg or a list')
            
        fig,axs =plt.subplots(sharex=True, sharey=False,figsize=figsize)
        for i, name in enumerate(self.sorted_order):
            if name not in self.samples.keys():
                continue
            sample = self.samples[name].query(cut) if cut else self.samples[name]
            if len(sample) == 0 or name in mask:
                continue
                
            weight = get_weights(sample,weights.get(name, 1) )
            (counts, _) = np.histogram(sample[variable], bins=bins,weights=weight)

            axs.hist(bins[:-1], bins, weights=counts, density=density,histtype='step',lw=2,color=self.colors[i],
                    label=f'''{name} \n{self.statistics(sample[variable])} \n cut_eff={(len(sample)/len(self.samples[name])):.3f}''')
            if errorbars:
                bin_centers = (bins[:-1]+bins[1:])/2
                axs.errorbar(x=bin_centers, y=counts, yerr=poisson_error(counts), fmt='.',
                        color=self.colors[i],markeredgecolor='white',markeredgewidth=0.5)

        # restrict title length
        print('cut=',cut)
        if title is None:
            axs.set_title(f'MC distribution', fontsize=text_fs)
        else: 
            axs.set_title(title, fontsize=text_fs)

        axs.set_xlabel(f'{variable}', fontsize=text_fs)
        ylabel = 'density' if density else f'# of events per bin {(bins[1]-bins[0]):.3f} GeV'
        axs.set_ylabel(ylabel, fontsize=text_fs)
        axs.grid()
        plt.legend(bbox_to_anchor=(1,1),ncol=legend_nc, fancybox=True, shadow=True,labelspacing=1.5)

    
    def plot_2d(self, bins,fig, ax,title_name, weights={},mask=[],
                variables=None,sub_df=None,sub_name=None,hist=None,cut=None):
        # Compute 2d hist
        if hist is None:
            counts_err = 0
            counts_tot = 0
            if sub_df is None:
                for i, name in enumerate(self.sorted_order):
                    if name not in self.samples.keys():
                        continue
                    sample = self.samples[name].query(cut) if cut else self.samples[name]
                    if len(sample) == 0 or name in mask:
                        continue

                    weight = get_weights(sample, weights.get(name, 1) )

                    (counts, xedges, yedges) = np.histogram2d(sample[variables[0]],sample[variables[1]],bins=bins,weights=weight)

                    (staterr_squared, _, _) = np.histogram2d(sample[variables[0]],sample[variables[1]],bins=bins,weights=weight**2)
                    staterror = poisson_error(staterr_squared)

                    sub_tot = unp.uarray(counts.round(0), staterror.round(0))
                    counts_tot += counts
                    counts_err += sub_tot
            else:
                sample = sub_df.query(cut) if cut else sub_df

                weight = get_weights(sample, weights.get(sub_name, 1) )

                (counts, xedges, yedges) = np.histogram2d(sample[variables[0]],sample[variables[1]],bins=bins,weights=weight)

                (staterr_squared, _, _) = np.histogram2d(sample[variables[0]],sample[variables[1]],bins=bins,weights=weight**2)
                staterror = poisson_error(staterr_squared)

                sub_tot = unp.uarray(counts.round(0), staterror.round(0))
                counts_tot += counts
                counts_err += sub_tot
        else:
            xedges, yedges = bins
            counts_tot = hist.round(0)
            counts_err = counts_tot

        if fig is not None and ax is not None:
            # 2D Histogram
            im = ax.imshow(counts_tot.T.round(0), origin='lower', aspect='auto', 
                             cmap='rainbow', norm=mcolors.LogNorm(),
                             extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
            fig.colorbar(im, ax=ax)
            if variables is None:
                ax.set_xlabel('$M_{miss}^2$')
                ax.set_ylabel(r'$|p_D| + |p_{\ell}|$')
            else:
                ax.set_xlabel(variables[0])
                ax.set_ylabel(variables[1])
            ax.set_title(title_name)
            ax.grid()
        
        return counts_err

    def plot_residuals(self, bins, data, model, ax, fig=None, n_par=0, model_cov=None, min_count=10):
        """Residual panel (1D: normalized residuals; 2D: |data - model| map) with the
        mismodeling legend: data/MC = R +- sigma_R, shape delta (or upper limit), chi2/ndf.
        n_par: parameters floated to obtain the model (post-fit plots). After a fit
        R is ~1 by construction, so only the shape line is meaningful there.
        1D: bins with < min_count expected entries are merged with neighbours for the
        chi2 (not in the plot) and drawn as hollow markers. model_cov: full MC
        covariance (stat + PID), used for the merged bins' errors."""
        if len(bins)>2: # 1d residual
            # Compute residuals (Data - Model) and their errors
            # at bins where data is not 0
            bin_centers = (bins[:-1] + bins[1:]) /2

            residuals = data - model
            residuals[unp.nominal_values(data) == 0] = 0  # Mask bins with 0 data

            res_val = unp.nominal_values(residuals)
            res_err = unp.std_devs(residuals)
            normalized_residuals = np.zeros_like(residuals)

            # Create a mask to exclude points where residual_errors are zero
            mask = res_err != 0
            # compute the normalized residuals
            normalized_residuals[mask] = res_val[mask] / res_err[mask]
            # legend: normalization, shape (excess variance) and chi2/ndf
            if model_cov is not None and np.shape(model_cov) != (len(bin_centers),) * 2:
                model_cov = None
            res = excess_variance(unp.nominal_values(data), unp.nominal_values(model),
                                  model_err=unp.std_devs(model), model_cov=model_cov,
                                  normalize=True, n_par=n_par, min_count=min_count)
            label = mismodeling_label(res)
            low = res['low_count_bins']

            # Plot the residuals in ax: solid = own bin in the chi2, hollow = merged
            # ax.errorbar(x=bin_centers, y=res_val, yerr=res_err, fmt='ok',label=label) # absolute residuals
            ax.plot(bin_centers[~low], normalized_residuals[~low], 'ok', label=label) # normalized residuals
            if low.any():
                ax.plot(bin_centers[low], normalized_residuals[low], 'o', mfc='none', mec='black',
                        label=f'merged in $\\chi^2$ (MC < {min_count:g})')
            
            # Add a horizontal line at y=0 for reference
            ax.axhline(0, color='gray', linestyle='--')
            # Label the residual plot
            ax.set_ylabel('Normalized Residuals')
            
        elif len(bins)==2: # 2d residual
            residuals = abs(data - model) # abs to make the plot simpler
            res_val = unp.nominal_values(residuals)
            # full 2D arrays: excess_variance flattens them itself
            # 2D: bins are not merged (neighbours in the flattened array are not
            # neighbours in the plane), low-count bins are left out of the chi2
            res = excess_variance(unp.nominal_values(data), unp.nominal_values(model),
                                  model_err=unp.std_devs(model), normalize=True, n_par=n_par,
                                  min_count=min_count, low_count='drop')
            label = mismodeling_label(res)
                
            # Plot the residuals in ax
            self.plot_2d(bins=bins, hist=res_val, fig=fig, ax=ax,title_name=label)
        
    def plot_ratios(self, bins, data, model, ax):
        # Compute ratios (Data / Model) and their errors
        bin_centers = (bins[:-1] + bins[1:]) /2
        mask_model = model != 0
        ratios = unp.uarray(np.ones_like(model), np.zeros_like(model))
        ratios[mask_model] = data[mask_model] / model[mask_model]
        rat_val = unp.nominal_values(ratios)
        rat_err = unp.std_devs(ratios)

        # Plot the ratios in ax
        ax.errorbar(x=bin_centers, y=rat_val, yerr=rat_err, fmt='.',color='black',
                     markeredgecolor='white',markeredgewidth=0.5)
        # Add a horizontal line at y=0 for reference
        ax.axhline(1, color='gray', linestyle='--')
        # Label the residual plot
        ax.set_ylabel('Ratios')
    
    def plot_data_mc_stacked(self,variable,bins,cut=None,weights={},data_sig_mask=False, density=False,mask=[],
                             apply_eventByEvent_correction=False, eventByEvent_weight_col='total_PID_weight',
                             ratio_or_residual='residual', bottom_plot=r'$D\tau\nu$',
                             figsize=(8,5),legend_nc=2,legend_fs=12,text_fs=14,title=None, n_par=0,
                             mc_uncertainty=False, save_path=None, dpi=300, min_count=10):
        # n_par: parameters floated to obtain the MC (post-fit plots), passed to plot_residuals
        # mc_uncertainty: add the PID uncertainty to the MC error and draw the MC band
        # min_count: bins with fewer expected MC entries are merged for the chi2
        assert ratio_or_residual in ['ratio', 'residual'], 'must choose ratio or residual'
        
        # Create a figure with two subplots: one for the histogram, one for the residual plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                       gridspec_kw={'height_ratios': [5, 1]})
            
        # MC
        mc_counts = self.plot_mc_1d(bins=bins, variable=variable, ax=ax1, cut=cut,mask=mask, density=density,
                                    weights=weights, apply_eventByEvent_correction=apply_eventByEvent_correction, 
                                    eventByEvent_weight_col=eventByEvent_weight_col,
                                    mc_uncertainty=mc_uncertainty)
        mc_cov = self.mc_model_cov
        # Data
        if self.data is not None:
            data_counts = self.plot_data_1d(bins=bins, variable=variable, sig_mask=data_sig_mask,
                                            ax=ax1, cut=cut, scale=weights.get('data', 1))
            # plot residuals
            if ratio_or_residual=='ratio':
                self.plot_ratios(bins=bins, data=data_counts, model=mc_counts, ax=ax2)
            elif ratio_or_residual=='residual':
                # Residuals (Data - Model)
                self.plot_residuals(bins=bins, data=data_counts, model=mc_counts, ax=ax2, n_par=n_par,
                                    model_cov=mc_cov, min_count=min_count)
                ax2.legend(bbox_to_anchor=(1,1),fancybox=True, shadow=True, fontsize=legend_fs)
            
        else: # if MC only
            data_counts = unp.uarray(np.zeros_like(mc_counts), np.zeros_like(mc_counts))
            bottom_comp = self.plot_mc_1d(bins=bins, sub_df=self.samples[bottom_plot], sub_name=bottom_plot, 
                                        variable=variable, ax=ax2, cut=cut,mask=mask, density=density, legend='simple_color',
                                        weights=weights, apply_eventByEvent_correction=apply_eventByEvent_correction, 
                                        eventByEvent_weight_col=eventByEvent_weight_col,
                                        mc_uncertainty=mc_uncertainty)
            ax2.set_ylabel(bottom_plot, fontsize=text_fs)
        
        # restrict title length
        print('cut=',cut)
        if title is None:
            if self.data is None:
                ax1.set_title(f'MC distribution', fontsize=text_fs)
            else:
                ax1.set_title(f'Data vs MC', fontsize=text_fs)
        else: 
            ax1.set_title(title, fontsize=text_fs)
            
        ax1.set_ylabel(f'# of events per bin {(bins[1]-bins[0]):.3f} GeV', fontsize=text_fs)
        ax1.legend(bbox_to_anchor=(1,1),ncol=legend_nc, fancybox=True, shadow=True,labelspacing=1.5, fontsize=legend_fs)
        ax1.grid()
        ax1.set_xlim(bins[0], bins[-1])   # shared, so ax2 follows
        ax2.grid(axis='x')                # optional: vertical gridlines in the residual panel too
        ax2.set_xlabel(self.var_name_dictionary.get(variable, variable), fontsize=text_fs)
        
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.show()

        return data_counts, mc_counts
        
        
    def plot_mc_sig_control(self,variable,bins,cut=None,weights={},mask=[], density=False,
                            apply_eventByEvent_correction=False,
                            eventByEvent_weight_col='total_PID_weight',
                            bkg_name='bkg_fakeD',samples_sig=None,
                            figsize=(10,5),legend_nc=1,text_fs=12):
        if type(variable)==str:
            # Create a figure with two subplots: one for the histogram, one for the residual plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, 
                                           gridspec_kw={'height_ratios': [5, 1]})
        elif type(variable)==list:
            # Create a figure with two subplots: one for sig, one for the control
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        if bkg_name=='bkg_fakeD':
            print('This part of the function needs to be rewritten!')

            
        elif bkg_name == 'bkg_BBbar':

            if samples_sig is None:
                raise ValueError(
                    "samples_sig must be provided for bkg_name='bkg_BBbar'"
                )
        
            components = [
                'bkg_combinatorial',
                'bkg_hadronicB_secondaryL',
            ]
        
            # --------------------------------------------------
            # Control region:
            # self.samples contains CR samples
            # Plot the two components as a stack
            # --------------------------------------------------
        
            control_mask = [
                name for name in self.sorted_order
                if name not in components
            ]
        
            control_counts = self.plot_mc_1d(
                bins=bins,
                variable=variable,
                ax=ax1,
                cut=cut,
                weights=weights,
                mask=control_mask,
                legend='simple_color',
                density=density,
                apply_eventByEvent_correction=apply_eventByEvent_correction,
                eventByEvent_weight_col=eventByEvent_weight_col,
            )
        
            # --------------------------------------------------
            # Signal region:
            # combine both components and treat the total
            # visually like "data"
            # --------------------------------------------------
        
            signal_df = pd.concat(
                [
                    samples_sig['bkg_combinatorial'],
                    samples_sig['bkg_hadronicB_secondaryL'],
                ],
                ignore_index=True
            )
        
            signal_counts = self.plot_data_1d(
                bins=bins,
                ax=ax1,
                sub_df=signal_df,
                variable=variable,
                cut=cut,
                scale=weights.get('signal region', 1),
                name='BBbar_bkg MC signal region',
                density=density,
                apply_eventByEvent_correction=apply_eventByEvent_correction,
                eventByEvent_weight_col=eventByEvent_weight_col,
            )
        
            # --------------------------------------------------
            # Residual
            # --------------------------------------------------
        
            self.plot_residuals(
                bins=bins,
                data=signal_counts,
                model=control_counts,
                ax=ax2,
            )
        
        elif bkg_name in ['bkg_continuum',]:
            sample_control = self.samples[bkg_name]
            if samples_sig is None:
                print(f'Error: samples_sig is required for {bkg_name}')
                return
            else:
                sample_sig = samples_sig[bkg_name]
                
            regions = {'control region': sample_control,
                       'signal region': sample_sig}
            
            sig_total = self.plot_data_1d(bins=bins, sub_df=sample_sig, variable=variable, name='signal region',
                                          ax=ax1,cut=None, scale=weights.get('signal region',1), density=density,
                                          apply_eventByEvent_correction=apply_eventByEvent_correction,
                                          eventByEvent_weight_col=eventByEvent_weight_col)

            control_total = self.plot_mc_1d(bins=bins, sub_df=sample_control, sub_name='control region', 
                                            variable=variable,ax=ax1,cut=cut,
                                            weights={'control region':weights.get('control region',1)}, mask=mask,
                                            density=density,
                                            apply_eventByEvent_correction=apply_eventByEvent_correction,
                                            eventByEvent_weight_col=eventByEvent_weight_col)
            
            # Residuals (Data - Model) and their errors
            self.plot_residuals(bins=bins, data=sig_total, model=control_total, ax=ax2)
            ax2.set_xlabel(f'{variable}')
        
        if type(variable)==str:
            ax1.set_title(f'signal region MC vs control region MC ({bkg_name=})')
            ylabel = 'Density' if density else f'# of events per bin {(bins[1]-bins[0]):.3f} GeV'
            ax1.set_ylabel(ylabel, fontsize=text_fs)
            ax1.legend(bbox_to_anchor=(1,1),ncol=legend_nc, fancybox=True, shadow=True,labelspacing=1.5, fontsize=text_fs)
            ax1.grid()
            ax1.set_xlim(bins[0], bins[-1])   # shared, so ax2 follows
            ax2.grid(axis='x')                # optional: vertical gridlines in the residual panel too
            ax2.legend(bbox_to_anchor=(1,1),fancybox=True, shadow=True, fontsize=text_fs)
            ax2.set_xlabel(f'{variable}', fontsize=text_fs)
        elif type(variable)==list:
            fig.suptitle(f'signal region vs weighted control region ({bkg_name=})')
        # Adjust the layout to avoid overlapping of the subplots
        plt.tight_layout()
        plt.show()

    def plot_data_subtracted_and_mc(self,var_list,bin_list,cut=None,weights={},
                                    correction=False,mask=['bkg_fakeD'],figsize=(10,10)):
        # get data in sig and sidebands regions
        data_left = self.data.query('D_M<1.83').copy()
        data_sig = self.data.query('1.84<D_M<1.9').copy()
        data_right = self.data.query('D_M>1.91').copy()
            
        # calculate the 2d hists
        variable_x, variable_y = var_list
        edges_x, edges_y = bin_list
        if var_list==['B0_CMS3_weMissM2','p_D_l']:
            var_x_label = '$M_{miss}^2$    [$GeV^2/c^4$]'
            var_y_label = r'$|p_D| + |p_{\ell}|$    [GeV/c]'
        else:
            var_x_label = var_list[0]
            var_y_label = var_list[1]
            
        data_sig_2d = self.plot_2d(bins=bin_list, sub_df=data_sig, variables=var_list, 
                                   weights={'sub_df':weights.get('data signal region',1)},
                                   fig=None, ax=None, cut=cut, title_name='data signal region')
        
        data_left_2d = self.plot_2d(bins=bin_list, sub_df=data_left, variables=var_list, 
                                   weights={'sub_df':weights.get('data left sideband',1)},
                                   fig=None, ax=None, cut=cut, title_name='data left sideband')
        
        data_right_2d = self.plot_2d(bins=bin_list, sub_df=data_right, variables=var_list, 
                                   weights={'sub_df':weights.get('data right sideband',1)},
                                   fig=None, ax=None, cut=cut, title_name='data right sideband')
        
        data_sb_2d = data_left_2d + data_right_2d

        # subtract the sidebands from sig region
        data_subtracted_2d = data_sig_2d - data_sb_2d

        # get 2 projections
        data_subtracted_x = data_subtracted_2d.sum(axis=1)  # Sum along the y-axis
        data_subtracted_y = data_subtracted_2d.sum(axis=0)  # Sum along the x-axis
        
        mc_proj_query = f'{edges_x[0]} < {variable_x} < {edges_x[-1]} and {edges_y[0]} < {variable_y} < {edges_y[-1]}'
           
        
        # Create figure and define subplots layout
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(11,11, figure=fig, wspace=5, hspace=1)
        ax1 = fig.add_subplot(gs[:4,:5])
        ax2 = fig.add_subplot(gs[:5,5:])
        ax3 = fig.add_subplot(gs[5, 5:])
        ax4 = fig.add_subplot(gs[5:10,:6])
        ax5 = fig.add_subplot(gs[10, :6])
        ax6 = fig.add_subplot(gs[7:,6:])

        # Top-left: 2D histogram of Data (p_D_l vs B0_CMS3_weMissM2)
        self.plot_2d(bins=bin_list, hist= abs(unp.nominal_values(data_subtracted_2d)),
                    fig=fig, ax=ax1, cut=cut, title_name='Data, D_M sidebands subtracted')
        ax1.set_xlabel(var_x_label)
        ax1.set_ylabel(var_y_label)
    
        
        # Top-right: 1D histogram of p_D_l projection + residuals
        data_y = self.plot_data_1d(bins=edges_y, hist=data_subtracted_y, ax=ax2,cut=cut, name='Data')
        mc_y = self.plot_mc_1d(bins=edges_y, variable=variable_y, ax=ax2, weights=weights,
                               cut=f'1.84<D_M<1.9 and {mc_proj_query} and '+cut, correction=correction,mask=mask,legend='simple_color')
        if var_list==['B0_CMS3_weMissM2','p_D_l']:
            ax2.set_title(r'$|p_D| + |p_{\ell}|$ Projection')
        else:
            ax2.set_title(f'{var_list[1]} Projection')
        ax2.grid()
        ax2.legend(ncol=1, framealpha=0, shadow=False,labelspacing=1.5,fontsize=8)
        # Residual plot below
        self.plot_residuals(bins=edges_y, data=data_y, model=mc_y, ax=ax3)
        ax3.set_xlabel(var_y_label)
        ax3.legend(bbox_to_anchor=(0.8,-1.2),ncol=2, framealpha=0, shadow=False,labelspacing=1.5)
        
        
        # Bottom-left: 1D histogram of mm2 projection + residuals
        data_x = self.plot_data_1d(bins=edges_x, hist=data_subtracted_x, ax=ax4,cut=cut, name='Data')
        mc_x = self.plot_mc_1d(bins=edges_x, variable=variable_x, ax=ax4, weights=weights,
                               cut=f'1.84<D_M<1.9 and {mc_proj_query} and '+cut, correction=correction,mask=mask,legend='simple_color')
        if var_list==['B0_CMS3_weMissM2','p_D_l']:
            ax4.set_title('$M_{miss}^2$ Projection')
        else:
            ax4.set_title(f'{var_list[0]} Projection')
        ax4.grid()
        ax4.legend(ncol=1, framealpha=0, shadow=False,labelspacing=1.5,fontsize=8)
        # Residual plot below
        self.plot_residuals(bins=edges_x, data=data_x, model=mc_x, ax=ax5)
        ax5.set_xlabel(var_x_label)
        ax5.legend(bbox_to_anchor=(0.8,12.5),ncol=2, framealpha=0, shadow=False,labelspacing=1.5)
        
        
        # Bottom-right: 2D histogram of MC (B0_CMS3_weMissM2 vs p_D_l)
        self.plot_2d(bins=bin_list, mask=mask,weights=weights,variables=var_list,
                    fig=fig, ax=ax6, cut=f'1.84<D_M<1.9 and {mc_proj_query} and '+cut, title_name='MC, fakeD removed')
        ax6.set_xlabel(var_x_label)
        ax6.set_ylabel(var_y_label)
        
        # Adjust layout to avoid overlap
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.4, wspace=0.4)
        plt.show()
        
        
        ################### Trimming and return the flattened data ###############
        histograms = {}
        for name, df in self.samples.items():
            if name in mask or name=='others':
                continue

            df_sig_region = df.query('1.855<D_M<1.885 and '+cut)
            if len(df_sig_region)==0:
                continue
                
            weight = get_weights(df_sig_region, weights.get(name, 1) )

            # Compute weighted histogram, event by event weight
            counts, xedges, yedges = np.histogram2d(
                df_sig_region[variable_x], df_sig_region[variable_y],
                bins=bin_list, weights=weight)

            # Compute sum of weight^2 for uncertainties
            staterr_squared, _, _ = np.histogram2d(
                df_sig_region[variable_x], df_sig_region[variable_y],
                bins=bin_list, weights=weight**2)

            # Store as uarray: Transpose to have consistent shape (y,x) if needed
            if name in [r'$D^{\ast\ast}\ell\nu$_narrow',r'$D^{\ast\ast}\ell\nu$_broad']:
                # merge the 2 resonant D** modes
                key = r'$D^{\ast\ast}\ell\nu$'
                # Get the existing value (or 0 if missing), add the new array, and save it
                histograms[key] = histograms.get(key, 0) + unp.uarray(counts, np.sqrt(staterr_squared))
            
            else:
                # store other modes individually
                histograms[name] = unp.uarray(counts, np.sqrt(staterr_squared))

        # combine the D** resonant and gap
        if weights.get(r'$D\ell\nu$_gap',1)==0:
            if r'$D^{\ast\ast}\ell\nu$' in histograms:
                histograms[r'$D^{\ast\ast}\ell\nu$ + gap'] = histograms.pop(r'$D^{\ast\ast}\ell\nu$')

        # Determine which bins pass the threshold based on sum of all templates
        indices_threshold = np.where(unp.nominal_values(data_subtracted_2d) >= 2)
        # Flatten the templates after cutting
        template_flat = {name: round_uarray(hist[indices_threshold]) for name, hist in histograms.items()}
        # Flatten data
        data_flat = round_uarray(data_subtracted_2d[indices_threshold])  # uarray
        
        #################### Prepare return tuples: (template dict, asimov data)
        temp_data = (template_flat, data_flat)
        return indices_threshold, temp_data
    
        
    def plot_all_2Dhist(self, bin_list:list, var_list=['B0_CMS3_weMissM2','p_D_l'], 
                        title='Generic MC 1/ab',cut=None, mask=[1.6,1]):
        variable_x, variable_y = var_list
        xedges, yedges = bin_list
       
        # create a mask
        mask_arr = np.ones((len(yedges)-1,len(xedges)-1)) # switch the shape for x,y as plotting counts.T
        if mask:
            # apply mask at mm2<1.6 and p_D_l>1 
            mm2_split = mask[0]
            pDl_split = mask[1]
            mm2_split_index, = np.asarray(np.isclose(xedges,mm2_split,atol=0.2)).nonzero()
            pDl_split_index, = np.asarray(np.isclose(yedges,pDl_split,atol=0.2)).nonzero()
            mask_arr[:,mm2_split_index[0]:] = mask[2] # select the small mm2
            mask_arr[:pDl_split_index[0],:] = mask[2] # select the large pDl
            
        fig = plt.figure(figsize=[16,20])
        for i, name in enumerate(self.sorted_order):
            if name not in self.samples.keys():
                continue
                    
            sample = self.samples[name]
            sample_size = len(sample.query(cut)) if cut else len(sample)
            if sample_size==0:
                continue
            ax = fig.add_subplot(5,3,i+1)
            (counts, xe, ye) = np.histogram2d(
                            sample.query(cut)[variable_x] if cut else sample[variable_x], 
                            sample.query(cut)[variable_y] if cut else sample[variable_y],
                            bins=[xedges, yedges])

            im = ax.imshow(counts.T, origin='lower', aspect='auto', 
                     cmap='rainbow', norm=mcolors.LogNorm(),alpha=mask_arr,
                     extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
            fig.colorbar(im, ax=ax)
#             X, Y = np.meshgrid(xedges, yedges)
#             im=ax.pcolormesh(X, Y, counts, cmap='rainbow', norm=mcolors.LogNorm(), alpha=mask_arr)
            ax.grid()
            ax.set_xlim(xedges.min(),xedges.max())
            ax.set_ylim(yedges.min(),yedges.max())
            ax.set_title(name,fontsize=14)

        fig.suptitle(f'{title} ({cut=})', y=0.92, fontsize=18)
        fig.supylabel(r'$|p^\ast_{D}|+|p^\ast_{\ell}| \ \ [GeV]$', x=0.05,fontsize=18)
        fig.supxlabel(r'$M_{miss}^2\ \ \ [GeV^2/c^4]$', y=0.08,fontsize=18)

        
    def plot_2Dhist_and_projections(self, bin_list:list, var_list=['B0_CMS3_weMissM2','p_D_l'], cut=None):
        variable_x, variable_y = var_list
        xedges, yedges = bin_list

        for name, sample in self.samples.items():
            sample_size = len(sample.query(cut)) if cut else len(sample)
            if sample_size==0:
                continue
            # Compute 2d hist
            (counts, xe, ye) = np.histogram2d(
                            sample.query(cut)[variable_x] if cut else sample[variable_x], 
                            sample.query(cut)[variable_y] if cut else sample[variable_y],
                            bins=[xedges, yedges])
            # Compute projections
            x_projection = counts.sum(axis=1)  # Sum along the y-axis
            y_projection = counts.sum(axis=0)  # Sum along the x-axis

            # Plot the 2D histogram
            fig, ax = plt.subplots(1, 3, figsize=(18, 5))

            # 2D Histogram
            im = ax[0].imshow(counts.T, origin='lower', aspect='auto', 
                             cmap='rainbow', norm=mcolors.LogNorm(),
                             extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
            fig.colorbar(im, ax=ax[0])
            ax[0].set_title(name)
            ax[0].set_xlabel('$M_{miss}^2$', fontsize=14)
            ax[0].set_ylabel(r'$|p_D| + |p_{\ell}|$', fontsize=14)
            ax[0].grid()

            # X Projection
            ax[1].bar(xedges[:-1], x_projection, width=np.diff(xedges), align='edge')
            ax[1].set_title('$M_{miss}^2$ Projection')
            ax[1].set_xlabel('$M_{miss}^2$')
            ax[1].set_ylabel('Counts')
            ax[1].grid()

            # Y Projection
            ax[2].barh(yedges[:-1], y_projection, height=np.diff(yedges), align='edge')
            ax[2].set_title(r'$|p_D| + |p_{\ell}|$ Projection')
            ax[2].set_xlabel('Counts')
            ax[2].set_ylabel(r'$|p_D| + |p_{\ell}|$')
            ax[2].grid()

            plt.tight_layout()
            plt.show()
        

    def plot_correlation(self, df, cut=None, target='B0_CMS3_weMissM2', variables=analysis_variables):
        fig = plt.figure(figsize=[50,300])
        for i in range(len(variables)):
            ax = fig.add_subplot(17,4,i+1)
            ax.hist2d(x=df.query(cut)[variables[i]] if cut else df[variables[i]], 
                      y=df.query(cut)[target] if cut else df[target], 
                      bins=30,cmap='rainbow', norm=mcolors.LogNorm())
            ax.set_ylabel(target,fontsize=30)
            ax.set_xlabel(variables[i],fontsize=30)
            ax.grid()
        

    def plot_FOM(self, sigModes, bkgModes, variable, bins, cut=None, 
                 reverse_selection=False, weight_column=None):
        # the binomial error might be incorrect for weight!=1
        
        # define signal / tot sample
        sig = pd.concat([self.samples[i] for i in sigModes])
        tot = pd.concat([self.samples[i] for i in sigModes+bkgModes])
        
        # create histograms
        sig_hist, _ = np.histogram(sig[variable], bins=bins,
                                   weights=None if weight_column is None else sig[weight_column])
        tot_hist, _ = np.histogram(tot[variable], bins=bins,
                                   weights=None if weight_column is None else tot[weight_column])
        
        # define statistics
        p_sig = np.zeros_like(sig_hist)
        p_sig[tot_hist>0] = sig_hist[tot_hist>0] / tot_hist[tot_hist>0]
        sig_unc = np.sqrt( sig_hist * (1 - p_sig) ) # binomial error
        tot_hist_unc = unp.uarray( tot_hist, np.sqrt(tot_hist) ) # poisson error
        sig_hist_unc = unp.uarray( sig_hist, sig_unc )
        bkg_hist_unc = tot_hist_unc - sig_hist_unc
        
        if reverse_selection: # x = f'{variable}<{i}'
            cumsig = sig_hist_unc.cumsum()
            cumbkg = bkg_hist_unc.cumsum()
        else: # x = f'{variable}>{i}'
            cumsig = sig_hist_unc.sum() - sig_hist_unc.cumsum()
            cumbkg = bkg_hist_unc.sum() - bkg_hist_unc.cumsum()

        # FOM calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            # Initialize variables with 0 ± 0
            shape = unp.nominal_values(cumsig).shape
            fom = unp.uarray(np.zeros(shape), np.zeros(shape))
            purity = unp.uarray(np.zeros(shape), np.zeros(shape))

            mask = cumsig + cumbkg > 0
            fom[mask] = cumsig[mask] / unp.sqrt(cumsig + cumbkg)[mask]
            purity[mask] = cumsig[mask] / (cumsig + cumbkg)[mask]
            sig_eff = cumsig / sig_hist.sum()


        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        color = 'tab:blue'
        ax1.set_ylabel('Efficiency', color=color)  # we already handled the x-label with ax1
        ax1.errorbar(x=bins[:-1], y=unp.nominal_values(sig_eff), yerr=unp.std_devs(sig_eff), 
                     color=color,label='Signal Efficiency')
        ax1.errorbar(x=bins[:-1], y=unp.nominal_values(purity), yerr=unp.std_devs(purity), 
                     color='green',label='Purity')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')
        ax1.grid()
        ax1.set_xlabel('Signal Probability')
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
        color = 'tab:red'
        ax2.set_ylabel('FOM', color=color)
        ax2.errorbar(x=bins[:-1], y=unp.nominal_values(fom), yerr=unp.std_devs(fom), 
                     color=color,label='FOM')
        ax2.tick_params(axis='y', labelcolor=color)
        #ax2.grid()
        ax2.legend(loc='upper right')

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title(f'FOM for {variable=}')
        plt.xlim(0,1)
        plt.ylim(bottom=0)
        plt.show()
# # +
###################### fit projection plots #####################
def fit_project_cabinetry(fit_result, templates_2d,staterror_2d,data_2d, 
                                      edges_list, direction='mm2', slice_thresholds=None):
    assert direction in ['mm2', 'p_D_l'], 'direction must be mm2 or p_D_l'

    def plot(bins, fitted_1d, data_1d, ax1, ax2, ax3, legend=True):        

        bin_width = np.diff(bins)
        bin_centers = (bins[:-1] + bins[1:]) /2
        # plot the templates with defined colors
        c = my_cmap.colors
        # sort the components to plot in order of fitted templates_project size
        sorted_order = ['bkg_fakeD',    'bkg_continuum',    'bkg_combinatorial',
                        'bkg_fakeL',     'bkg_hadronicB_secondaryL',   r'$D\ell\nu$_gap',
                        r'$D^{\ast\ast}\ell\nu$',           r'$D^{\ast\ast}\tau\nu$',
                        r'$D^\ast\ell\nu$',                 r'$D\ell\nu$',
                        r'$D^\ast\tau\nu$',                 r'$D\tau\nu$']
        
        # plot data and fitted values
        data_val = unp.nominal_values(data_1d)
        data_err = unp.std_devs(data_1d)
        ax1.errorbar(x=bin_centers, y=data_val, yerr=data_err, fmt='.',
                    color='black',markeredgecolor='white',markeredgewidth=0.5)
        
        bottom_hist = np.zeros_like(data_1d)
        for i,name in enumerate(sorted_order):
            values = unp.nominal_values(fitted_1d[name])
            errors = unp.std_devs(fitted_1d[name])
            
            ax1.bar(x=bins[:-1], height=values, bottom=bottom_hist, color = c[i],
                    width=bin_width, align='edge', label=name)
            bottom_hist = bottom_hist + values
        
        # plot the residual and pull
        
        # Combine all fitted templates to get the total fitted projection
        total_fitted_1d = np.sum(list(fitted_1d.values()), axis=0)
        # Calculate residuals
        residual = data_1d - total_fitted_1d
        residual_val = unp.nominal_values(residual)
        residual_err = unp.std_devs(residual)
        pull = np.array([0 if residual_err[i]==0 else (residual_val[i]/residual_err[i]) for i in range(len(residual))])
                        
        ax2.errorbar(x=bin_centers, y=residual_val, yerr=residual_err, fmt='.',
                    color='black',markeredgecolor='white',markeredgewidth=0.5)
        ax2.axhline(y=0, linestyle='-', linewidth=1, color='r')
        ax3.scatter(x=bin_centers, y=pull, c='black')
        ax3.axhline(y=0, linestyle='-', linewidth=1, color='r')            

        ax1.grid()
        ax1.set_ylabel('# of counts per bin',fontsize=12)
        ax1.set_xlim(bin_edges.min(), bin_edges.max())
        ax1.set_ylim(0, data_val.max()*1.2)
        ax2.set_ylabel('residual',fontsize=12)
        ax2.set_xlim(bin_edges.min(), bin_edges.max())
        ax3.set_ylabel('pull',fontsize=12)
        ax3.set_xlim(bin_edges.min(), bin_edges.max())
        if legend:
            ax1.legend(bbox_to_anchor=(1,1),ncol=1, fancybox=True, shadow=True,labelspacing=1)
    
    def combine_templates_with_uncertainties(templates_2d, staterror_2d):
        combined_templates = {}

        for name in templates_2d.keys():
            # Ensure that both arrays have the same shape
            assert templates_2d[name].shape == staterror_2d[name].shape, \
                f"Shape mismatch between template and staterror for {name}"

            # Combine the template values with their uncertainties
            combined_templates[name] = unp.uarray(templates_2d[name], staterror_2d[name])

        return combined_templates
    
    # get fit results
    assert len(fit_result.bestfit) == len(fit_result.uncertainty), "Values and uncertainties lists must have the same length."
    # Combine values and uncertainties into ufloat objects
    combined_result = [ufloat(val, unc) for val, unc in zip(fit_result.bestfit, fit_result.uncertainty)]
    
    # get 2d templates and data
    components_names = [n.rstrip('norm').rstrip('_') for n in fit_result.labels[:12]]
    combined_templates_2d = combine_templates_with_uncertainties(templates_2d, staterror_2d)
    data_2d = unp.uarray(data_2d, np.sqrt(data_2d))
    fitted_2d = {}
    
    # Calculate fitted values, 2d
    for i, name in enumerate(components_names):
        fitted_2d[name] = combined_templates_2d[name] * combined_result[i]
          
    # setup the appropriate axis for 1D projection
    if direction == 'mm2':
        axis = 0
        axis_label = '$M_{miss}^2$'
        axis_unit = '$[GeV^2/c^4]$'
        other_axis_label = r'$|p_D|\ +\ |p_l|$'
        other_axis_unit = '[GeV]'
    elif direction == 'p_D_l':
        axis = 1
        axis_label = r'$|p_D|\ +\ |p_l|$'
        axis_unit = '[GeV]'
        other_axis_label = '$M_{miss}^2$'
        other_axis_unit = '$[GeV^2/c^4]$'
    
    # Plotting
    bin_edges = edges_list[axis]
    if not slice_thresholds:
        # calculate 1d projection if not slice
        # sum over the same axis due to data_2d being transposed
        data_1d = np.sum(data_2d, axis=axis)
        fitted_1d = {name: np.sum(template, axis=axis) for name, template in fitted_2d.items()}
    
        # plot
        fig = plt.figure(figsize=(6,9))
        gs = gridspec.GridSpec(3,1, height_ratios=[0.7,0.15,0.15])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        gs.update(hspace=0.2) 

        plot(bin_edges, fitted_1d, data_1d, ax1, ax2, ax3)
        ax1.set_title(f'Fit projection to {axis_label}',fontsize=14)
        ax3.set_xlabel(axis_label,fontsize=14)

    elif slice_thresholds:
        # calculate 1d projection if slice
        # Determine which bins correspond to the threshold
        other_bin_edges = edges_list[1-axis]
        threshold_value = slice_thresholds[axis]
        below_threshold_indices = np.where(other_bin_edges < threshold_value)[0]
        above_threshold_indices = np.where(other_bin_edges >= threshold_value)[0][:-1] # bin_edges has 1 extra indices than counts

        # Split the data and templates based on the threshold
        data_below_threshold = np.sum(data_2d[below_threshold_indices, :], axis=axis) if axis == 0 else np.sum(data_2d[:, below_threshold_indices], axis=axis)
        data_above_threshold = np.sum(data_2d[above_threshold_indices, :], axis=axis) if axis == 0 else np.sum(data_2d[:, above_threshold_indices], axis=axis)

        fitted_below_threshold = {name: np.sum(template[below_threshold_indices,:], axis=axis) if axis == 0 else np.sum(template[:,below_threshold_indices], axis=axis) 
                                  for name, template in fitted_2d.items()}
        fitted_above_threshold = {name: np.sum(template[above_threshold_indices,:], axis=axis) if axis == 0 else np.sum(template[:,above_threshold_indices], axis=axis) 
                                  for name, template in fitted_2d.items()}
        
        # plot
        fig = plt.figure(figsize=(16,9))
        spec = gridspec.GridSpec(6,7, figure=fig, wspace=1, hspace=0.5)
        ax1 = fig.add_subplot(spec[:-2,:3])
        ax2 = fig.add_subplot(spec[:-2,3:])
        ax3 = fig.add_subplot(spec[-2,:3])
        ax4 = fig.add_subplot(spec[-2,3:])
        ax5 = fig.add_subplot(spec[-1,:3])
        ax6 = fig.add_subplot(spec[-1,3:])
        #gs.update(hspace=0) 

        plot(bin_edges, fitted_below_threshold, data_below_threshold, ax1, ax3, ax5, legend=False)
        plot(bin_edges, fitted_above_threshold, data_above_threshold, ax2, ax4, ax6, legend=True)

        ax1.set_title(f'{other_axis_label} < {threshold_value}  {other_axis_unit}',fontsize=12)
        ax2.set_title(f'{other_axis_label} > {threshold_value}  {other_axis_unit}',fontsize=12)
        fig.suptitle(f'Fit projection to {axis_label} in slices of {other_axis_label}',fontsize=14)
        fig.supxlabel(axis_label + '  ' + axis_unit,fontsize=14)
        
    return fig



# plotting version: two residual plots, residual_signal = data - all_temp
def mpl_projection_residual_iMinuit(Minuit, templates_2d, data_2d, edges, slices=[1.6,1],direction='mm2', plot_with='pltbar'):
    assert direction in ['mm2', 'p_D_l'], 'direction must be in [mm2, p_D_l]'
    assert plot_with in ['mplhep', 'pltbar'], 'plot_with must be in [mplhep, pltbar]'

    fitted_components_names = list(Minuit.parameters)
    #### fitted_templates_2d = templates / normalization * yields
    fitted_templates_2d = [templates_2d[i]/templates_2d[i].sum() * Minuit.values[i] for i in range(len(templates_2d))]
    # fitted_templates_err = templates_2d_err, yields_err in quadrature
                         # = fitted_templates_2d x sqrt( (1/templates_2d) + (yield_err/yield)**2 ) if templates_2d[i,j]!=0
                         # = 0 if yield ==0 or templates_2d[i,j]==0
    fitted_templates_err = np.zeros_like(templates_2d)
    non_zero_masks = [np.where(t!= 0) for t in templates_2d]
    for i in range(len(templates_2d)):
        if Minuit.values[i]==0:
            continue
        else:
            fitted_templates_err[i][non_zero_masks[i]] = fitted_templates_2d[i][non_zero_masks[i]] * \
            np.sqrt(1/templates_2d[i][non_zero_masks[i]] + (Minuit.errors[i]/Minuit.values[i])**2)

    def extend(x):
        return np.append(x, x[-1])

    def errorband(bins, template_sum, template_err, ax):
        fitted_sum = np.sum(template_sum, axis=0)
        fitted_err = poisson_error(np.sum(np.array(template_err)**2, axis=0)) # assuming the correlations between each template are 0
        ax.fill_between(bins, extend(fitted_sum - fitted_err), extend(fitted_sum + fitted_err),
        step="post", color="black", alpha=0.3, linewidth=0, zorder=100,)   

    def plot_with_hep(bins, templates_project, templates_project_err, data, signal_name, ax1, ax2, ax3):
        data_project = data.sum(axis=axis_to_be_summed_over)
        # plot the templates and data
        hep.histplot(templates_project, bin_edges, stack=True, histtype='fill', sort='yield_r', label=fitted_components_names, ax=ax1)
        # errorband(bin_edges, templates_project, templates_project_err, ax1)
        hep.histplot(data_project, bin_edges, histtype='errorbar', color='black', w2=data_project, ax=ax1)
        # plot the residual
        signal_index = fitted_components_names.index(signal_name)
        residual = data_project - np.sum(templates_project, axis=0)
        residual_signal = residual + templates_project[signal_index]
        # Error assuming the correlations between data and templates, between each template, are 0
        residual_err = poisson_error(data_project + np.sum(np.array(templates_project_err)**2, axis=0))
        residual_err_signal = poisson_error(residual_err**2 - np.array(templates_project_err[signal_index]))

        pull = [0 if residual_err[i]==0 else (residual[i]/residual_err[i]) for i in range(len(residual))]
        pull_signal = [0 if residual_err_signal[i]==0 else (residual_signal[i]/residual_err_signal[i]) for i in range(len(residual_signal))]
        #hep.histplot(residual, bin_edges, histtype='errorbar', color='black', yerr=residual_err, ax=ax2)
        hep.histplot(residual, bin_edges, histtype='errorbar', color='black', ax=ax2)
        ax2.axhline(y=0, linestyle='-', linewidth=1, color='r')
        #hep.histplot(residual_signal, bin_edges, histtype='errorbar', color='black', yerr=residual_err_signal, ax=ax3)
        hep.histplot(pull, bin_edges, histtype='errorbar', color='black', ax=ax3)
        ax3.axhline(y=0, linestyle='-', linewidth=1, color='r')

        ax1.grid()
        ax1.set_ylabel('# of counts per bin',fontsize=16)
        ax1.set_xlim(bin_edges.min(), bin_edges.max())
        ax1.set_ylim(0, data_project.max()*1.2)
        ax2.set_ylabel('pull',fontsize=14)
        ax2.set_xlim(bin_edges.min(), bin_edges.max())
        ax3.set_ylabel('pull + signal',fontsize=10)
        ax3.set_xlim(bin_edges.min(), bin_edges.max())
        ax1.legend(bbox_to_anchor=(1,1),ncol=1, fancybox=True, shadow=True,labelspacing=1)

    def plot_with_bar(bins, templates_project, templates_project_err, data, ax1, ax2, ax3,signal_name=None):        
        # calculate the arguments for plotting
        bin_width = bins[1]-bins[0]
        bin_centers = (bins[:-1] + bins[1:]) /2
        data_project = data.sum(axis=axis_to_be_summed_over)
        data_err = poisson_error(data_project)

        # plot the templates with defined colors
        c = plt.cm.tab20.colors
        # sort the components to plot in order of fitted templates_project size
        sorted_indices = sorted(range(len(templates_2d)), key=lambda i: np.sum(templates_project[i]), reverse = True)
        bottom_hist = np.zeros(data.shape[1-axis_to_be_summed_over])
        for i in sorted_indices:
            binned_counts = templates_project[i]
            ax1.bar(x=bins[:-1], height=binned_counts, bottom=bottom_hist, color = c[i],
                    width=bin_width, align='edge', label=fitted_components_names[i])
            bottom_hist = bottom_hist + binned_counts
        # errorband(bin_edges, templates_project, templates_project_err, ax1)

        # plot the data
        ax1.errorbar(x=bin_centers, y=data_project, yerr=data_err, fmt='.',
                    color='black',markeredgecolor='white',markeredgewidth=0.5)
        # plot the residual
        residual = data_project - np.sum(templates_project, axis=0)
        # Error assuming the correlations between data and templates, between each template, are 0
        residual_err = poisson_error(data_project + np.sum(np.array(templates_project_err)**2, axis=0))

        pull = [0 if residual_err[i]==0 else (residual[i]/residual_err[i]) for i in range(len(residual))]
        ax2.errorbar(x=bin_centers, y=residual, yerr=residual_err, fmt='.',
                    color='black',markeredgecolor='white',markeredgewidth=0.5)
        ax2.axhline(y=0, linestyle='-', linewidth=1, color='r')
        ax3.scatter(x=bin_centers, y=pull, c='black')
        ax3.axhline(y=0, linestyle='-', linewidth=1, color='r')            

        ax1.grid()
        ax1.set_ylabel('# of counts per bin',fontsize=16)
        ax1.set_xlim(bin_edges.min(), bin_edges.max())
        ax1.set_ylim(0, data_project.max()*1.2)
        ax2.set_ylabel('residual',fontsize=14)
        ax2.set_xlim(bin_edges.min(), bin_edges.max())
        ax3.set_ylabel('pull',fontsize=14)
        ax3.set_xlim(bin_edges.min(), bin_edges.max())
        ax1.legend(bbox_to_anchor=(1,1),ncol=1, fancybox=True, shadow=True,labelspacing=1)

#         signal_index = fitted_components_names.index(signal_name)
#         residual_signal = residual + templates_project[signal_index]
#         residual_err_signal = poisson_error(residual_err**2 - np.array(templates_project_err[signal_index]))
#         pull_signal = [0 if residual_err_signal[i]==0 else (residual_signal[i]/residual_err_signal[i]) for i in range(len(residual_signal))]

    if direction=='mm2':
        direction_label = '$M_{miss}^2$'
        direction_unit = '$[GeV^2/c^4]$'
        other_direction_label = r'$|p_D|\ +\ |p_l|$'
        other_direction_unit = '[GeV]'
        axis_to_be_summed_over = 0

        bin_edges = edges[axis_to_be_summed_over] #xedges
        slice_position = slices[1-axis_to_be_summed_over] #p_D_l
        slice_index, = np.asarray(np.isclose(edges[1-axis_to_be_summed_over],slice_position,atol=0.2)).nonzero()
        first_slice_index = (slice_index[0]-1)
        second_slice_index = (slice_index[0])

        # parameters for slices==True
        fitted_project_slice1 = [temp[:first_slice_index,:].sum(axis=axis_to_be_summed_over) for temp in fitted_templates_2d]
        fitted_project_slice2 = [temp[second_slice_index:,:].sum(axis=axis_to_be_summed_over) for temp in fitted_templates_2d]
        fitted_project_slice1_err = [poisson_error((err**2)[:first_slice_index,:].sum(axis=axis_to_be_summed_over)) for err in fitted_templates_err]
        fitted_project_slice2_err = [poisson_error((err**2)[second_slice_index:,:].sum(axis=axis_to_be_summed_over)) for err in fitted_templates_err]
        data_slice1 = data_2d[:first_slice_index,:]
        data_slice2 = data_2d[second_slice_index:,:]

    elif direction=='p_D_l':
        direction_label = r'$|p_D|\ +\ |p_l|$'
        direction_unit = '[GeV]'
        other_direction_label = '$M_{miss}^2$'
        other_direction_unit = '$[GeV^2/c^4]$'
        axis_to_be_summed_over = 1

        bin_edges = edges[axis_to_be_summed_over] #yedges
        slice_position = slices[1-axis_to_be_summed_over] #mm2
        slice_index, = np.asarray(np.isclose(edges[1-axis_to_be_summed_over],slice_position,atol=0.2)).nonzero()
        first_slice_index = (slice_index[0]-1)
        second_slice_index = (slice_index[0])

        # parameters for slices==True
        fitted_project_slice1 = [temp[:,:first_slice_index].sum(axis=axis_to_be_summed_over) for temp in fitted_templates_2d]
        fitted_project_slice2 = [temp[:,second_slice_index:].sum(axis=axis_to_be_summed_over) for temp in fitted_templates_2d]
        fitted_project_slice1_err = [poisson_error((err**2)[:,:first_slice_index].sum(axis=axis_to_be_summed_over)) for err in fitted_templates_err]
        fitted_project_slice2_err = [poisson_error((err**2)[:,second_slice_index:].sum(axis=axis_to_be_summed_over)) for err in fitted_templates_err]
        data_slice1 = data_2d[:,:first_slice_index]
        data_slice2 = data_2d[:,second_slice_index:]

    if not slices:
        fig = plt.figure(figsize=(6.4,6.4))
        gs = gridspec.GridSpec(3,1, height_ratios=[0.7,0.15,0.15])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        gs.update(hspace=0.3) 
        fitted_project = [temp.sum(axis=axis_to_be_summed_over) for temp in fitted_templates]
        fitted_project_err = [temp.sum(axis=axis_to_be_summed_over) for temp in fitted_templates_err]

        # plot the templates and data and templates_err
        if plot_with=='mplhep':
            plot_with_hep(bin_edges, fitted_project, fitted_project_err, counts, '$D\\tau\\nu$', ax1, ax2,ax3)
        elif plot_with=='pltbar':
            plot_with_bar(bin_edges, fitted_project, fitted_project_err, counts, '$D\\tau\\nu$', ax1, ax2,ax3)
        ax1.set_title(f'Fitting projection to {direction_label}')
        ax3.set_xlabel(direction_label)

    elif slices:
        fig = plt.figure(figsize=(16,9))
        spec = gridspec.GridSpec(6,2, figure=fig, wspace=0.4, hspace=0.5)
        ax1 = fig.add_subplot(spec[:-2, 0])
        ax2 = fig.add_subplot(spec[:-2, 1])
        ax3 = fig.add_subplot(spec[-2, 0])
        ax4 = fig.add_subplot(spec[-2, 1])
        ax5 = fig.add_subplot(spec[-1, 0])
        ax6 = fig.add_subplot(spec[-1, 1])
        #gs.update(hspace=0) 

        # plot the templates and data and template_err
        if plot_with=='mplhep':
            plot_with_hep(bin_edges, fitted_project_slice1, fitted_project_slice1_err, data_slice1, slice1_signal, ax1, ax3, ax5)
            plot_with_hep(bin_edges, fitted_project_slice2, fitted_project_slice2_err, data_slice2, slice2_signal, ax2, ax4, ax6)
        elif plot_with=='pltbar':
            plot_with_bar(bin_edges, fitted_project_slice1, fitted_project_slice1_err, data_slice1, ax1, ax3, ax5)
            plot_with_bar(bin_edges, fitted_project_slice2, fitted_project_slice2_err, data_slice2, ax2, ax4, ax6)

        ax1.set_title(f'{other_direction_label} < {slice_position}  {other_direction_unit}',fontsize=14)
        ax2.set_title(f'{other_direction_label} > {slice_position}  {other_direction_unit}',fontsize=14)
        fig.suptitle(f'Fitted projection to {direction_label} in slices of {other_direction_label}',fontsize=16)
        fig.supxlabel(direction_label + '  ' + direction_unit,fontsize=16)


######################### plotly #######################
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# class ply:
#     def __init__(self, df):
#         self.df = df
        
#     def hist(self, variable='B0_CMS3_weMissM2', cut=None, facet=False):
#         # Create a histogram
#         fig=px.histogram(self.df.query(cut) if cut else self.df, 
#                          x=variable, color='mode', nbins=60, 
#                          marginal='box', #opacity=0.5, barmode='overlay',
#                          color_discrete_sequence=px.colors.qualitative.Plotly,
#                          template='simple_white', title='Signal MC',
#                          facet_col='p_D_l_region' if facet else None)

#         # Manage the layout
#         fig.update_layout(font_family='Rockwell', hovermode='closest',
#                           legend=dict(orientation='h',title='',x=1,y=1,xanchor='right',yanchor='bottom'))

#         # Manage the hover labels
#         count_by_color = self.df.groupby('mode')['__event__'].count()
#         for i, (color, count) in enumerate(count_by_color.items()):
#             fig.update_traces(hovertemplate='Bin_Count: %{y}<br>Overall_Count: '+str(count),selector={'name':color})
#             #fig.add_annotation(x=1+i*3, y=8000, text=f'Total Count ({color}): {count}', showarrow=True)

#         # Update axes labels
#         if variable=='B0_CMS3_weMissM2':
#             fig.update_xaxes(title_text="$M_{miss}^2\ \ [GeV^2/c^4]$", row=1)

#         # Show the plot
#         fig.show()
        
#     def hist2d(self, cut=None, facet=False):
#         # Define number of colors to generate
#         color_sequence = ['rgb(255,255,255)'] + px.colors.sequential.Rainbow[1:]
#         num_colors = 9
#         # Generate colors with uniform spacing and Rainbow color scale
#         my_colors = [[i/(num_colors-1), color_sequence[i]] for i in range(num_colors)]

#         # Create a 2d histogram
#         fig = px.density_heatmap(self.df.query(cut) if cut else self.df, 
#                                  x="B0_CMS3_weMissM2", y="p_D_l",
#                                  marginal_x='histogram', marginal_y='histogram',
#                                  nbinsx=40,nbinsy=40,color_continuous_scale=my_colors,
#                                  template='simple_white', title='Signal MC',
#                                  facet_col='mode' if facet else None,
#                                  facet_col_wrap=3 if facet else None,)

#         # Update axes labels
#         fig.update_xaxes(title_text="$M_{miss}^2\ \ [GeV^2/c^4]$", row=1)
#         fig.update_yaxes(title_text="$|p_D|+|p_l|\ \ [GeV/c]$",row=1, col=1)

#         fig.show()
        
#     def plot_FOM(self, sigModes, bkgModes, variable, test_points,cut=None):
#         # calculate the FOM, efficiencies
#         sig = self.df.loc[self.df['mode'].isin(sigModes)]
#         bkg = self.df.loc[self.df['mode'].isin(bkgModes)]
#         sig_tot = len(sig)
#         bkg_tot = len(bkg)
#         BDT_FOM = []
#         BDT_FOM_err = []
#         BDT_sigEff = []
#         BDT_sigEff_err = []
#         BDT_bkgEff = []
#         BDT_bkgEff_err = []
#         for i in test_points:
#             nsig = len(sig.query(f"{cut} and {variable}>{i}" if cut else f"{variable}>{i}"))
#             nbkg = len(bkg.query(f"{cut} and {variable}>{i}" if cut else f"{variable}>{i}"))
#             tot = nsig+nbkg
#             tot_err = np.sqrt(tot)
#             FOM = nsig / tot_err # s / √(s+b)
#             FOM_err = np.sqrt( (tot_err - FOM/2)**2 /tot**2 * nsig + nbkg**3/(4*tot**3) + 9*nbkg**2*np.sqrt(nsig*nbkg)/(4*tot**5) )

#             BDT_FOM.append(FOM)
#             BDT_FOM_err.append(FOM_err)

#             sigEff = nsig / sig_tot
#             sigEff_err = sigEff * np.sqrt(1/nsig + 1/sig_tot)
#             bkgEff = nbkg / bkg_tot
#             bkgEff_err = bkgEff * np.sqrt(1/nbkg + 1/bkg_tot)
#             BDT_sigEff.append(sigEff)
#             BDT_sigEff_err.append(sigEff_err)
#             BDT_bkgEff.append(bkgEff)
#             BDT_bkgEff_err.append(bkgEff_err)
        

#         # Create figure with secondary y-axis
#         fig = make_subplots(specs=[[{"secondary_y": True}]])

#         # Add traces
#         fig.add_trace(
#             go.Scatter(x=test_points, y=BDT_FOM, name="FOM",
#                        error_y=dict(type='data',array=BDT_FOM_err,visible=True)),
#             secondary_y=True,
#         )

#         fig.add_trace(
#             go.Scatter(x=test_points, y=BDT_sigEff, name="sig_eff",
#                        error_y=dict(type='data',array=BDT_sigEff_err,visible=True)),
#             secondary_y=False,
#         )
        
#         fig.add_trace(
#             go.Scatter(x=test_points, y=BDT_bkgEff, name="bkg_eff",
#                        error_y=dict(type='data',array=BDT_bkgEff_err,visible=True)),
#             secondary_y=False,
#         )

#         # Add figure title
#         fig.update_layout(
#             title_text="MVA Performance",
#             template='simple_white',
#             hovermode='x',
#             legend=dict(orientation='h',title='',x=1,y=1.1,xanchor='right',yanchor='bottom')
#         )

#         # Set x-axis title
#         fig.update_xaxes(title_text=variable)

#         # Set y-axes titles
#         fig.update_yaxes(title_text="<b>FOM</b>", secondary_y=True)
#         fig.update_yaxes(title_text="Efficiency", secondary_y=False)

#         fig.show()
        
#     def plot_cut_efficiency(self, cut, variable='B0_CMS3_weQ2lnuSimple',bins=15):
#         # Create figure with secondary y-axis
#         fig = make_subplots()
        
#         for mode in self.df['mode'].unique():
#             if mode in ['bkg_continuum','bkg_fakeDTC','bkg_fakeB','bkg_others']:
#                 continue
#             comp=self.df.loc[self.df['mode']==mode]
#             (bc, bins1) = np.histogram(comp[variable], bins=bins)
#             (ac, bins1) = np.histogram(comp.query(cut)[variable], bins=bins1)
#             bc+=1
#             ac+=1
#             efficiency = ac / bc
#             factor = [i if i<1 else 0 for i in 1/ac + 1/bc] # mannually set the uncertainty to 0 if bin count==0
#             efficiency_err = efficiency * np.sqrt(factor)
#             bin_centers = (bins1[:-1] + bins1[1:]) /2
            
#             # Add traces
#             fig.add_trace(
#                 go.Scatter(x=bin_centers, y=efficiency, name=mode,
#                            error_y=dict(type='data',array=efficiency_err,visible=True))
#             )
        
        
#         # Add figure title
#         fig.update_layout(
#             title_text=f'Efficiency for {cut=}',
#             template='simple_white',
#             hovermode='closest',
#             legend=dict(orientation='h',title='',x=1,y=1,xanchor='right',yanchor='bottom')
#         )

#         # Set x-axis title
#         fig.update_xaxes(title_text=variable)

#         # Set y-axes titles
#         fig.update_yaxes(title_text="<b>Efficiency</b>")

#         fig.show()

# # plotting version: residual = data - all_temp
# def ply_projection_residual(Minuit, templates_2d, data_2d, edges, slices=[1.6,1],direction='mm2'):
#     if direction not in ['mm2', 'p_D_l']:
#         raise ValueError('direction in [mm2, p_D_l]')
#     fitted_components_names = list(Minuit.parameters)
#     #### fitted_templates_2d = templates / normalization * yields
#     fitted_templates_2d = [templates_2d[i]/templates_2d[i].sum() * Minuit.values[i] for i in range(len(templates_2d))]
#     # fitted_templates_err = templates_2d_err, yields_err in quadrature
#                          # = fitted_templates_2d x sqrt( (1/templates_2d) + (yield_err/yield)**2 ) if templates_2d[i,j]!=0
#                          # = 0 if yield ==0 or templates_2d[i,j]==0
#     fitted_templates_err = np.zeros_like(templates_2d)
#     non_zero_masks = [np.where(t!= 0) for t in templates_2d]
#     for i in range(len(templates_2d)):
#         if Minuit.values[i]==0:
#             continue
#         else:
#             fitted_templates_err[i][non_zero_masks[i]] = fitted_templates_2d[i][non_zero_masks[i]] * \
#             np.sqrt(1/templates_2d[i][non_zero_masks[i]] + (Minuit.errors[i]/Minuit.values[i])**2)        

#     if direction=='mm2':
#         direction_label = '$M_{miss}^2$'
#         direction_unit = '$[GeV^2/c^4]$'
#         other_direction_label = r'$|p_D|\ +\ |p_l|$'
#         other_direction_unit = '[GeV]'
#         axis_to_be_summed_over = 0

#         bin_edges = edges[axis_to_be_summed_over] #xedges
#         slice_position = slices[1-axis_to_be_summed_over] #p_D_l
#         slice_index, = np.asarray(np.isclose(edges[1-axis_to_be_summed_over],slice_position,atol=0.1)).nonzero()
#         first_slice_index = (slice_index[0]-1)
#         second_slice_index = (slice_index[0])

#         # parameters for slices==True
#         fitted_project_slice1 = [temp[:first_slice_index,:].sum(axis=axis_to_be_summed_over) for temp in fitted_templates_2d]
#         fitted_project_slice2 = [temp[second_slice_index:,:].sum(axis=axis_to_be_summed_over) for temp in fitted_templates_2d]
#         fitted_project_slice1_err = [np.sqrt((err**2)[:first_slice_index,:].sum(axis=axis_to_be_summed_over)) for err in fitted_templates_err]
#         fitted_project_slice2_err = [np.sqrt((err**2)[second_slice_index:,:].sum(axis=axis_to_be_summed_over)) for err in fitted_templates_err]
#         data_slice1 = data_2d[:first_slice_index,:]
#         data_slice2 = data_2d[second_slice_index:,:]


#     elif direction=='p_D_l':
#         direction_label = r'$|p_D|\ +\ |p_l|$'
#         direction_unit = '[GeV]'
#         other_direction_label = '$M_{miss}^2$'
#         other_direction_unit = '$[GeV^2/c^4]$'
#         axis_to_be_summed_over = 1

#         bin_edges = edges[axis_to_be_summed_over] #yedges
#         slice_position = slices[1-axis_to_be_summed_over] #mm2
#         slice_index, = np.asarray(np.isclose(edges[1-axis_to_be_summed_over],slice_position,atol=0.1)).nonzero()
#         first_slice_index = (slice_index[0]-1)
#         second_slice_index = (slice_index[0])

#         # parameters for slices==True
#         fitted_project_slice1 = [temp[:,:first_slice_index].sum(axis=axis_to_be_summed_over) for temp in fitted_templates_2d]
#         fitted_project_slice2 = [temp[:,second_slice_index:].sum(axis=axis_to_be_summed_over) for temp in fitted_templates_2d]
#         fitted_project_slice1_err = [np.sqrt((err**2)[:,:first_slice_index].sum(axis=axis_to_be_summed_over)) for err in fitted_templates_err]
#         fitted_project_slice2_err = [np.sqrt((err**2)[:,second_slice_index:].sum(axis=axis_to_be_summed_over)) for err in fitted_templates_err]
#         data_slice1 = data_2d[:,:first_slice_index]
#         data_slice2 = data_2d[:,second_slice_index:]

#     else:
#         raise ValueError('Current version only supports projection to either mm2 or p_D_l')

        
#     def plot(bins, templates_project, templates_project_err, data, column):        
#         # calculate the arguments for plotting
#         bin_width = bins[1]-bins[0]
#         bin_centers = (bins[:-1] + bins[1:]) /2
#         data_project = data.sum(axis=axis_to_be_summed_over)
#         data_err = np.sqrt(data_project)

#         # sort the components to plot in order of fitted templates_project size
#         c = px.colors.qualitative.Light24
#         sorted_indices = sorted(range(len(templates_2d)), key=lambda i: np.sum(templates_project[i]), reverse = True)
#         bottom_hist = np.zeros(data.shape[1-axis_to_be_summed_over])
#         for i in sorted_indices:
#             binned_counts = templates_project[i]
#             fig.add_trace(go.Bar(x=bins[:-1], y=binned_counts, width=bin_width,
#                                  alignmentgroup=1, name=fitted_components_names[i],
#                                  legendgroup=fitted_components_names[i],
#                                  marker=dict(color=c[i]),
#                                  showlegend=True if column==1 else False), 
#                           row=1, col=column)

#         # plot the data
#         fig.add_trace(go.Scatter(x=bin_centers, y=data_project, name='data',mode='markers',
#                                 error_y=dict(type='data',array=data_err,visible=True),
#                                 legendgroup='data',marker=dict(color=c[11]),
#                                 showlegend=True if column==1 else False),
#                       row=1, col=column)

#         # plot the residual
#         residual = data_project - np.sum(templates_project, axis=0)
#         # Error assuming the correlations between data and templates, between each template, are 0
#         residual_err = np.sqrt(data_project + np.sum(np.array(templates_project_err)**2, axis=0))
                        
#         pull = [0 if residual_err[i]==0 else (residual[i]/residual_err[i]) for i in range(len(residual))]
#         fig.add_trace(go.Scatter(x=bin_centers, y=residual,name='residual',mode='markers',
#                         error_y=dict(type='data',array=residual_err,visible=True),
#                                 legendgroup='residual',marker=dict(color=c[12]),
#                                 showlegend=True if column==1 else False),
#               row=2, col=column)
#         fig.add_trace(go.Scatter(x=bin_centers, y=pull,name='pull',mode='markers',
#                                 legendgroup='pull',marker=dict(color=c[13]),
#                                 showlegend=True if column==1 else False),
#               row=3, col=column)
        

#     # create subplots
#     fig = make_subplots(rows=3, cols=2, row_heights=[0.7, 0.15,0.15],vertical_spacing=0.05,
#                     subplot_titles=(f'{other_direction_label} < {slice_position}  {other_direction_unit}',
#                                     f'{other_direction_label} > {slice_position}  {other_direction_unit}',
#                                     '','','',''))

#     # plot the templates and data and template_err
#     plot(bin_edges, fitted_project_slice1, fitted_project_slice1_err, data_slice1, column=1)
#     plot(bin_edges, fitted_project_slice2, fitted_project_slice2_err, data_slice2, column=2)
    
#     # Set x/y-axis title
#     fig.update_xaxes(title_text=direction_label + direction_unit,row=3)
#     fig.update_yaxes(title_text='# of counts per bin', row=1, col=1)
#     fig.update_yaxes(title_text='residual', row=2, col=1)
#     fig.update_yaxes(title_text='pull', row=3, col=1)

#     # Add figure title
#     fig.update_layout(
#         width=850,height=650,
#         title_text=f'Fitted projection to {direction_label} in slices of {other_direction_label}',
#         template='simple_white',
#         hovermode='closest',
#         barmode='stack',
#         legend=dict(orientation='h',title='',x=1,y=1.1,xanchor='right',yanchor='bottom'),
#         shapes=[dict(type='line', y0=0, y1=0, xref='paper', 
#                      x0=bin_edges.min(), x1=bin_edges.max())],
#     )

#     fig.show()
# endregion
