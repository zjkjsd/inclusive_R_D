# -*- coding: utf-8 -*-
"""Template construction, pyhf/cabinetry workspaces, the D-mass fit, pull
linearity checks, and pyhf/toy utilities.

Split out of utilities.py. Imports only go one way: this module imports from
utilities.py, never the reverse, and it does not import plotting.py.
"""
import numpy as np
import matplotlib.pyplot as plt

from utilities import (
    apply_eventByEvent_weight,
    excess_variance,
    get_weights,
    mismodeling_label,
    poisson_error,
    round_uarray,
)


##############################################################################        
##                          Templates and workspace                         ##
##############################################################################

from uncertainties import ufloat, correlated_values, UFloat
import uncertainties.unumpy as unp
import copy
from termcolor import colored

def rebin_histogram(counts, threshold):
    """
    Rebins a histogram, merging bins until the count exceeds a threshold.
    Handles uncertainties if `counts` is a uarray.
    
    Parameters:
        counts (array-like or unp.uarray): Counts with or without uncertainties.
        threshold (float): Minimum count threshold to stop merging bins.
        
    Returns:
        new_counts (array-like or unp.uarray): Rebinned counts (with propagated uncertainties if input has uncertainties).
        new_bin_edges (array-like): New bin edges after rebinning.
        old_bin_edges (array-like): The original dummy bin edges (0 to len(counts)).
    """
    # Generate dummy bin edges
    dummy_bin_edges = np.arange(len(counts) + 1)

    # Check if counts have uncertainties
    has_uncertainties = isinstance(counts[0], UFloat)

    # Extract nominal values and uncertainties if necessary
    if has_uncertainties:
        counts_nominal = unp.nominal_values(counts)
        counts_uncertainties_squared = unp.std_devs(counts) ** 2
    else:
        counts_nominal = counts
        counts_uncertainties_squared = None  # No uncertainties in this case

    new_counts_nominal = []
    new_uncertainties_squared = []
    new_edges = [dummy_bin_edges[0]]
    
    i = 0
    while i < len(counts_nominal):
        bin_count_nominal = counts_nominal[i]
        bin_uncertainty_squared = (
            counts_uncertainties_squared[i] if counts_uncertainties_squared is not None else 0
        )
        start_edge = dummy_bin_edges[i]
        end_edge = dummy_bin_edges[i + 1]
        
        # Merge bins until bin_count is above the threshold
        while bin_count_nominal < threshold and i < len(counts_nominal) - 1:
            i += 1
            bin_count_nominal += counts_nominal[i]
            if counts_uncertainties_squared is not None:
                bin_uncertainty_squared += counts_uncertainties_squared[i]
            end_edge = dummy_bin_edges[i + 1]
        
        new_counts_nominal.append(bin_count_nominal)
        if counts_uncertainties_squared is not None:
            new_uncertainties_squared.append(bin_uncertainty_squared)
        new_edges.append(end_edge)
        
        i += 1

    # Combine nominal values and uncertainties into uarray if applicable
    if has_uncertainties:
        new_counts = unp.uarray(
            new_counts_nominal, np.sqrt(np.array(new_uncertainties_squared))
        )
    else:
        new_counts = np.array(new_counts_nominal)
    
    return new_counts, np.array(new_edges), dummy_bin_edges

# Function to rebin another histogram using new bin edges
def rebin_histogram_with_new_edges(counts_with_uncertainties, old_bin_edges, new_bin_edges):
    """
    Rebins a histogram with counts and uncertainties grouped using unp.uarray.

    Parameters:
        counts_with_uncertainties (unp.uarray): Counts with uncertainties as a single uarray.
        old_bin_edges (array-like): Original bin edges.
        new_bin_edges (array-like): New bin edges for rebinning.

    Returns:
        new_counts_with_uncertainties (unp.uarray): Rebinned counts with uncertainties.
    """
    old_bin_edges = np.asarray(old_bin_edges)
    new_bin_edges = np.asarray(new_bin_edges)
    counts = unp.nominal_values(counts_with_uncertainties)
    uncertainties = unp.std_devs(counts_with_uncertainties)

    if len(old_bin_edges) != len(counts) + 1:
        raise ValueError("old_bin_edges must contain exactly len(counts) + 1 entries")
    if new_bin_edges[0] != old_bin_edges[0] or new_bin_edges[-1] != old_bin_edges[-1]:
        raise ValueError("new_bin_edges must span the full old-bin range")

    # Each new edge must coincide with an old edge: this routine combines bins,
    # rather than splitting them and inventing an intra-bin distribution.
    edge_indices = np.searchsorted(old_bin_edges, new_bin_edges)
    if (
        np.any(edge_indices == len(old_bin_edges))
        or not np.array_equal(old_bin_edges[edge_indices], new_bin_edges)
    ):
        raise ValueError("new_bin_edges must be a subset of old_bin_edges")

    new_counts = np.add.reduceat(counts, edge_indices[:-1])
    variances = np.add.reduceat(uncertainties**2, edge_indices[:-1])
    return unp.uarray(new_counts, np.sqrt(variances))


def create_templates_new(samples:dict, bins_sr:list, bins_sb:list,
                     variables=['B0_recMissM2','p_D_l'], cut=None,
                     bin_threshold=1, merge_threshold=10,scale_lumi=1,
                     use_real_data_instead_of_asimov=False, real_data=None,
                     apply_eventByEvent_correction=False, eventByEvent_weight_col=None,
                     sample_to_exclude=['bkg_fakeTracks','bkg_other_TDTl','bkg_other_signal'],
                     sample_weights={r'$D^{\ast\ast}\ell\nu$_broad':1,
                                     r'$D\ell\nu$_gap':1,}):


    #################### Create template 2d histograms with uncertainties for signal channel ################
    assert len(bins_sr)!=len(variables), 'Dimensions of variables and bins are not equal'

    print('Creating templates for the signal region')
    histograms_sr = {}
    for name, df_sr_sb in samples.items():
        if name in sample_to_exclude:
            continue

        df_sr = df_sr_sb.query('1.855<D_M<1.885')
        if cut is not None:
            df_sr=df_sr.query(cut)

        weight_sr = get_weights(df_sr, sample_weights.get(name,1))

        if apply_eventByEvent_correction: # update the weight
            weight_sr = apply_eventByEvent_weight(df_sr, weight_sr, eventByEvent_weight_col)
            
        # Compute weighted histogram, event by event weight
        if len(variables)==2:
            counts_sr, xedges, yedges = np.histogram2d(
                df_sr[variables[0]], df_sr[variables[1]],
                bins=bins_sr, weights=weight_sr)

            # Compute sum of weight^2 for uncertainties
            staterr_squared_sr, _, _ = np.histogram2d(
                df_sr[variables[0]], df_sr[variables[1]],
                bins=bins_sr, weights=weight_sr**2)
            
        elif len(variables)==1:
            counts_sr, edges = np.histogram(
                df_sr[variables[0]],bins=bins_sr[0], weights=weight_sr)

            staterr_squared_sr, _ = np.histogram(
                df_sr[variables[0]],bins=bins_sr[0], weights=weight_sr**2)

     
        # Store as uarray: Transpose to have consistent shape (y,x) if needed
        if name in [r'$D^{\ast\ast}\ell\nu$_narrow',r'$D^{\ast\ast}\ell\nu$_broad']:
            # merge the 2 resonant D** modes
            key = r'$D^{\ast\ast}\ell\nu$'
            # Get the existing value (or 0 if missing), add the new array, and save it
            histograms_sr[key] = histograms_sr.get(key, 0) + unp.uarray(counts_sr, np.sqrt(staterr_squared_sr))

        else:
            # store other modes individually
            histograms_sr[name] = unp.uarray(counts_sr, np.sqrt(staterr_squared_sr))

    ################### Trimming and flattening ###############
    # Determine which bins pass the threshold based on sum of all templates
    sr_hists_sum = np.sum(list(histograms_sr.values()), axis=0)  # uarray sum
    indices_threshold_sr = np.where(unp.nominal_values(sr_hists_sum) >= bin_threshold)

    # remove sample name if no events
    histograms_sr = {name:hist for name,hist in histograms_sr.items() if np.sum(hist)!=0}
    if sample_weights[r'$D\ell\nu$_gap']==0:
        if r'$D^{\ast\ast}\ell\nu$' in histograms_sr:
            histograms_sr[r'$D^{\ast\ast}\ell\nu$ + gap'] = histograms_sr.pop(r'$D^{\ast\ast}\ell\nu$')

    # Flatten the templates after cutting
    template_sr_flat = {name: round_uarray(hist[indices_threshold_sr]) for name, hist in histograms_sr.items()}
    # Asimov data is the sum of all templates
    asimov_data_sr = round_uarray(np.sum(list(template_sr_flat.values()), axis=0))  # uarray

    ###################################### Create templates for D_M sidebands channel ################################
    print('Creating templates for D_M sidebands channel')
    histograms_sb = {}
    for name, df_sr_sb in samples.items():
        if name in sample_to_exclude:
            continue

        df_sb = df_sr_sb.query('D_M<1.85 or 1.9<D_M')
        if cut is not None:
            df_sb=df_sb.query(cut)

        weight_sb = get_weights(df_sb, sample_weights.get(name,1))

        if apply_eventByEvent_correction: # update the weight
            weight_sb = apply_eventByEvent_weight(df_sb, weight_sb, eventByEvent_weight_col)
            
        # Compute weighted histogram, event by event weight
        if len(variables)==2:
            counts_sb, xedges, yedges = np.histogram2d(
                df_sb[variables[0]], df_sb[variables[1]],
                bins=bins_sb, weights=weight_sb)

            # Compute sum of weight^2 for uncertainties
            staterr_squared_sb, _, _ = np.histogram2d(
                df_sb[variables[0]], df_sb[variables[1]],
                bins=bins_sb, weights=weight_sb**2)
            
        elif len(variables)==1:
            counts_sb, edges = np.histogram(
                df_sb[variables[0]],bins=bins_sb[0], weights=weight_sb)

            staterr_squared_sb, _ = np.histogram(
                df_sb[variables[0]],bins=bins_sb[0], weights=weight_sb**2)

     
        # Store as uarray: Transpose to have consistent shape (y,x) if needed
        if name in [r'$D^{\ast\ast}\ell\nu$_narrow',r'$D^{\ast\ast}\ell\nu$_broad']:
            # merge the 2 resonant D** modes
            key = r'$D^{\ast\ast}\ell\nu$'
            # Get the existing value (or 0 if missing), add the new array, and save it
            histograms_sb[key] = histograms_sb.get(key, 0) + unp.uarray(counts_sb, np.sqrt(staterr_squared_sb))
        else:
            # store other modes individually
            histograms_sb[name] = unp.uarray(counts_sb, np.sqrt(staterr_squared_sb))

    ################### Trimming and flattening ###############
    # Determine which bins pass the threshold based on sum of all templates
    sb_hists_sum = np.sum(list(histograms_sb.values()), axis=0)  # uarray sum
    indices_threshold_sb = np.where(unp.nominal_values(sb_hists_sum) >= bin_threshold)

    # remove sample name if no events
    histograms_sb = {name:hist for name,hist in histograms_sb.items() if np.sum(hist)!=0}
    if sample_weights[r'$D\ell\nu$_gap']==0:
        if r'$D^{\ast\ast}\ell\nu$' in histograms_sb:
            histograms_sb[r'$D^{\ast\ast}\ell\nu$ + gap'] = histograms_sb.pop(r'$D^{\ast\ast}\ell\nu$')

    # Flatten the templates after cutting
    template_sb_flat = {name: round_uarray(hist[indices_threshold_sb]) for name, hist in histograms_sb.items()}
    # Asimov data is the sum of all templates
    asimov_data_sb = round_uarray(np.sum(list(template_sb_flat.values()), axis=0))  # uarray


    ################## Create a new set of templates with merged bins ###########
    if len(variables)==2:
        # Rebin asimov_data according to merge_threshold
        new_counts_sr, new_dummy_bin_edges_sr, old_dummy_bin_edges_sr = rebin_histogram(asimov_data_sr, merge_threshold)
        print(f'creating new templates with merged bins in the signal region, original template length = {len(asimov_data_sr)}, new template (merge small bins) length = {len(new_counts_sr)}')

        if len(asimov_data_sb)!=0:
            new_counts_sb, new_dummy_bin_edges_sb, old_dummy_bin_edges_sb = rebin_histogram(asimov_data_sb, merge_threshold)
            print(f'creating new templates with merged bins in the D mass sidebands, original template length = {len(asimov_data_sb)}, new template (merge small bins) length = {len(new_counts_sb)}')
        else:
            new_counts_sb = asimov_data_sb
            print('Templates with merged bins in the D mass sidebands are not created, because no events are populated in the sidebands')

        template_sr_flat_merged = {}
        template_sb_flat_merged = {}
        for name, t in template_sr_flat.items():
            # Rebin using new edges
            # Note: old_dummy_bin_edges and new_dummy_bin_edges come from rebin_histogram
            # For consistency, we must use the same old bin edges as for asimov_data:
            rebinned_sr = rebin_histogram_with_new_edges(t, old_dummy_bin_edges_sr, new_dummy_bin_edges_sr)
            template_sr_flat_merged[name] = round_uarray(rebinned_sr)

        for name, t in template_sb_flat.items():
            if len(asimov_data_sb)!=0:
                rebinned_sb = rebin_histogram_with_new_edges(t, old_dummy_bin_edges_sb, new_dummy_bin_edges_sb)
            else:
                rebinned_sb = t
            template_sb_flat_merged[name] = round_uarray(rebinned_sb)

        asimov_data_sr_merged = round_uarray(np.sum(list(template_sr_flat_merged.values()), axis=0))
        asimov_data_sb_merged = round_uarray(np.sum(list(template_sb_flat_merged.values()), axis=0))
            
    elif len(variables)==1:
        template_sr_flat_merged = {}
        asimov_data_sr_merged = []
        template_sb_flat_merged = {}
        asimov_data_sb_merged = []
    
    ################################### Prepare return tuples: (template dict, data) ##############################

    if use_real_data_instead_of_asimov and real_data is not None:
        real_data_sr = real_data.query('1.855<D_M<1.885')
        real_data_sb = real_data.query('D_M<1.85 or 1.9<D_M')
        if cut is not None:
            real_data_sr = real_data_sr.query(cut)
            real_data_sb = real_data_sb.query(cut)
        (data_sr_2d, _1, _2) = np.histogram2d(real_data_sr[variables[0]], real_data_sr[variables[1]], bins=bins_sr)
        data_sr_flat = round_uarray(data_sr_2d[indices_threshold_sr])  # uarray
        
        (data_sb_2d, _1, _2) = np.histogram2d(real_data_sb[variables[0]], real_data_sb[variables[1]], bins=bins_sb)
        data_sb_flat = round_uarray(data_sb_2d[indices_threshold_sb])  # uarray
        
    else:
        data_sr_flat = asimov_data_sr
        data_sb_flat = asimov_data_sb

    temp_sr = (template_sr_flat, data_sr_flat)
    temp_sb = (template_sb_flat, data_sb_flat)
    temp_sr_merged = (template_sr_flat_merged, asimov_data_sr_merged)
    temp_sb_merged = (template_sb_flat_merged, asimov_data_sb_merged)

    return temp_sr, temp_sb, temp_sr_merged, temp_sb_merged
    


def create_2d_template_from_1d(template_flat: dict, data_flat: unp.uarray,
                               indices_threshold: tuple, bins: list):
    """
    Convert a flattened 1D template back to a 2D histogram with uncertainties based on provided binning.

    Parameters:
        template_flat (dict): Flattened 1D templates for different samples, with values as unp.array.
        data_flat (unp.uarray): Flattened Asimov data as unp.array.
        indices_threshold (tuple): The indices that correspond to the bins retained after applying the threshold.
        bins (list): The bin edges for the 2D histogram.

    Returns:
        temp_2d (dict): 2D histograms for each sample as unp.array.
        data_2d (unp.uarray): 2D Asimov dataset as unp.array.
    """
    temp_2d = {}
    
    # Extract bin dimensions
    xbins, ybins = bins
    shape_2d = (len(xbins) - 1, len(ybins) - 1)
    
    for name, flat_data in template_flat.items():
        # Initialize an empty 2D array for the histogram
        hist_2d = unp.uarray(np.zeros(shape_2d).T, np.zeros(shape_2d).T)
        # Assign the flattened data back into the appropriate indices
        hist_2d[indices_threshold] = flat_data
        temp_2d[name] = hist_2d

    # Initialize an empty 2D array for the Asimov data
    data_2d = unp.uarray(np.zeros(shape_2d).T, np.zeros(shape_2d).T)
    # Assign the flattened data back into the appropriate indices
    data_2d[indices_threshold] = data_flat
    
    return temp_2d, data_2d

def compare_2d_hist(data, model, bins_x, bins_y, 
                    xlabel='X-axis', ylabel='Y-axis', 
                    data_label='Data', model_label='Model', n_par=0):
    """
    Compare two 2D histograms with residual plots by projecting onto x and y axes.
    Parameters:
        data (2D array): First 2D histogram values (data).
        model (2D array): Second 2D histogram values (model).
        bins_x (1D array): Bin edges for x-axis.
        bins_y (1D array): Bin edges for y-axis.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        data_label (str): Label for the first histogram (data).
        model_label (str): Label for the second histogram (model).
        n_par (int): Number of parameters floated to obtain the model (post-fit plots);
            reduces ndf of the shape term. After a fit, data/MC = R is ~1 by
            construction, so only the shape line of the residual legend is meaningful.
    """

    def compute_residuals(data, model):
        # Compute residuals and errors (assuming Poisson for data)
        residuals = data - model
        residuals[unp.nominal_values(data) == 0] = 0  # Mask bins with 0 data
        res_val = unp.nominal_values(residuals)
        res_err = unp.std_devs(residuals)
        return res_val, res_err

    def plot_residuals(ax, bin_centers, res_val, res_err, label):
        # label: normalization / shape / chi2 text from mismodeling_label
        ax.errorbar(x=bin_centers,y=res_val,yerr=res_err,fmt='.',color='black',
                     markeredgecolor='white',markeredgewidth=0.5, label=label)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_ylabel('Residuals')
        ax.legend()

    def label_for(d, m):
        # nominal values -> data error from the expected count (Pearson), MC error from the uarray
        res = excess_variance(unp.nominal_values(d), unp.nominal_values(m),
                              model_err=unp.std_devs(m), normalize=True, n_par=n_par)
        return mismodeling_label(res)

    # Project histograms onto x-axis
    projData_x = np.sum(data, axis=0)
    projModel_x = np.sum(model, axis=0)

    # Project histograms onto y-axis
    projData_y = np.sum(data, axis=1)
    projModel_y = np.sum(model, axis=1)

    lab_x = label_for(projData_x, projModel_x)
    lab_y = label_for(projData_y, projModel_y)

    # Bin centers
    bin_centers_x = (bins_x[:-1] + bins_x[1:]) / 2
    bin_centers_y = (bins_y[:-1] + bins_y[1:]) / 2

    # Residuals
    res_x, res_err_x = compute_residuals(projData_x, projModel_x)
    res_y, res_err_y = compute_residuals(projData_y, projModel_y)

    # Create the figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), gridspec_kw={'height_ratios': [4, 1]})

    # X-axis projection (top-left)
    axes[0, 0].hist(bin_centers_x, bins=bins_x, weights=unp.nominal_values(projModel_x), 
                    histtype='step', label=model_label)
    axes[0, 0].errorbar(x=bin_centers_x,y=unp.nominal_values(projData_x),
                        yerr=unp.std_devs(projData_x),fmt='.',color='black',
                     markeredgecolor='white',markeredgewidth=0.5, label=data_label)
    axes[0, 0].set_ylabel('# of Events')
    axes[0, 0].set_title(f'Projection onto {xlabel}')
    axes[0, 0].grid()
    axes[0, 0].legend()

    # X-axis residuals (bottom-left)
    plot_residuals(axes[1, 0], bin_centers_x, res_x, res_err_x, lab_x)
    axes[1, 0].set_xlabel(xlabel)

    # Y-axis projection (top-right)
    axes[0, 1].hist(bin_centers_y, bins=bins_y, weights=unp.nominal_values(projModel_y), 
                    histtype='step', label=model_label)
    axes[0, 1].errorbar(bin_centers_y, unp.nominal_values(projData_y), 
                        yerr=unp.std_devs(projData_y), fmt='ok', label=data_label)
    axes[0, 1].set_ylabel('# of Events')
    axes[0, 1].set_title(f'Projection onto {ylabel}')
    axes[0, 1].grid()
    axes[0, 1].legend()

    # Y-axis residuals (bottom-right)
    plot_residuals(axes[1, 1], bin_centers_y, res_y, res_err_y, lab_y)
    axes[1, 1].set_xlabel(ylabel)

    plt.tight_layout()
    plt.show()


def create_workspace(temp_data_channels: list, 
                     mc_uncer: bool = True, fakeD_uncer: bool = True) -> dict:
    """
    Create a structured workspace dictionary for statistical analysis and fitting.

    Args:
        temp_asimov_channels (list): A list of tuples, where each tuple contains:
                                     - A dictionary of sample templates with bin data.
                                     - The corresponding Asimov dataset.
        mc_uncer (bool, optional): If True, includes statistical uncertainties for all MC backgrounds. Default is True.
        fakeD_uncer (bool, optional): If True, includes statistical uncertainties for the 'bkg_fakeD' sample. Default is True.

    Returns:
        dict: A structured dictionary containing:
              - 'channels': A list of channels with their respective samples and uncertainties.
              - 'measurements': A list defining the measurement setup.
              - 'observations': The observed data for each channel.
              - 'version': The version identifier of the workspace format.
    """

    # Initialize key workspace components
    channels = []
    observations = []
    measurements = [{"name": "R_D", "config": {"poi": "$D\\tau\\nu$_norm", "parameters": []}}]
    version = "1.0.0"

    # NOTE (fake-D normalisation): the run-dependent fake-D factor
    # (0.87 for run1, 1.0 for run2) is applied ONLY inside the BBbar weight
    # tuning, 5_BBbkg_weights_optuna_minuit.py.  It is deliberately NOT applied
    # when building templates: here bkg_fakeD floats freely and absorbs it.
    # Applying it in both places would double-count the correction.
    normfactor = [ r'$D\ell\nu$', r'$D^\ast\ell\nu$', r'$D\tau\nu$', r'$D^{\ast\ast}\ell\nu$ + gap','bkg_fakeD',]
    normfactor += ['BBbar_measured_hadronic', 'BBbar_semileptonic',      'BBbar_unmeasured:2-body',
                   'BBbar_unmeasured:3-body', 'BBbar_unmeasured:4-body', 'BBbar_unmeasured:5-body',
                   'BBbar_unmeasured:6-body', 'BBbar_unmeasured:7-body', 'BBbar_unmeasured:5+-body',] # combinatorial bkg control sample
    # NOTE (BBbar categories): prepare_bbbar_reweighting emits ':5+-body' when called
    # with cap_nbody=5, which is what the tuning script uses; the separate
    # 5-, 6- and 7-body keys are only produced with cap_nbody=None and are
    # retained for backwards compatibility with earlier workspaces.
    # NOTE (scope): floating the BBbar families is appropriate for the tuning
    # context, but in the signal-region fit they should be fixed or
    # constrained -- otherwise the final fit reopens what the tuning of the
    # family weights already constrained.
    normsys_modifiers = {
                    r'$D^\ast\tau\nu$': {
                        'name': r'$D^\ast\tau\nu$_norm',
                        'type': 'normsys',
                        'data': {"hi": 1.1, "lo": 0.9}
                    },
                     r'$D^{\ast\ast}\tau\nu$': {
                        'name': r'$D^{\ast\ast}\tau\nu$_norm',
                        'type': 'normsys',
                        'data': {"hi": 1.3, "lo": 0.7}
                    },
#                      'bkg_combinatorial': {
#                         'name': 'bkg_combinatorial_norm',
#                         'type': 'normsys',
#                         'data': {"hi": 1.2, "lo": 0.8}
#                     },
#                      'bkg_hadronicB_secondaryL': {
#                         'name': 'bkg_hadronicB_secondaryL_norm',
#                         'type': 'normsys',
#                         'data': {"hi": 1.2, "lo": 0.8}
#                     },
                     'bkg_continuum': {
                        'name': 'bkg_continuum_norm',
                        'type': 'normsys',
                        'data': {"hi": 1.15, "lo": 1}
                    },
#                      'bkg_fakeL': {
#                         'name': 'bkg_fakeL_norm',
#                         'type': 'normsys',
#                         'data': {"hi": 1.2, "lo": 0.8}
#                     },
#                      'bkg_fakeTracks': {
#                         'name': 'bkg_fakeTracks_norm',
#                         'type': 'normsys',
#                         'data': {"hi": 1.2, "lo": 0.8}
#                     },
                     }

    # Loop over each channel (index, tuple of template_flat and asimov_data)
    for ch_index, (template_flat, data_flat) in enumerate(temp_data_channels):
        
        # Store observed data for the channel
        observations.append({
            'name': f'channel_{ch_index}',
            'data': unp.nominal_values(data_flat).tolist()  # Extract nominal values from uncertainties
        })
        
        # Initialize channel structure
        channels.append({
            'name': f'channel_{ch_index}',
            'samples': []
        })

        # Loop over each sample in the channel
        sample_names = list(template_flat.keys())
        
        for sample_name in sample_names:
            sample_data = template_flat[sample_name]
            if np.sum(sample_data) == 0:
                continue  # skip samples with 0 events

            # Build the sample entry
            if sample_name in normsys_modifiers:
                sample_entry = {
                    'name': sample_name,
                    'data': unp.nominal_values(sample_data).tolist(),
                    'modifiers': [ normsys_modifiers[sample_name] ] }
            elif sample_name in normfactor:
                sample_entry = {
                    'name': sample_name,
                    'data': unp.nominal_values(sample_data).tolist(),
                    'modifiers': [
                        {
                            'name': sample_name + '_norm',
                            'type': 'normfactor',
                            'data': None  # Normalization factor modifier
                        }
                    ]
                }
            else:
                sample_entry = {
                    'name': sample_name,
                    'data': unp.nominal_values(sample_data).tolist(),
                    'modifiers': [ ]
                }

            # Add uncertainty modifiers for statistical errors
            bkg_comp = ['bkg_fakeD', ] # r'$D\tau\nu$', r'$D^\ast\tau\nu$', r'$D^{\ast\ast}\tau\nu$'
            if (sample_name in bkg_comp) and fakeD_uncer:
                # Add statistical uncertainty for signals using shapesys
                sample_entry['modifiers'].append({
                    'name': f'mcStat_ch{ch_index}', # fakeD_stat
                    'type': 'staterror', # 'shapesys'
                    'data': unp.std_devs(sample_data).tolist()
                })
            elif (sample_name not in bkg_comp) and mc_uncer:
                # Add statistical uncertainty for all other components using staterror
                sample_entry['modifiers'].append({
                    'name': f'mcStat_ch{ch_index}',
                    'type': 'staterror',
                    'data': unp.std_devs(sample_data).tolist()
                })

            # Append the sample entry to the channel
            channels[ch_index]['samples'].append(sample_entry)
            

            # Define parameter bounds based on whether it's a background or signal sample
            if sample_name == 'bkg_fakeD':
                par_config = {"name": sample_name+'_norm', "bounds": [[-5, 5]], } # "inits": [0]
            elif sample_name.startswith('bkg'):
                par_config = {"name": sample_name+'_norm', "bounds": [[-5, 5]], } #"fixed":True}
            else:
                par_config = {"name": sample_name+'_norm', "bounds": [[-5, 5]],}

            # Add parameter configuration if it doesn't already exist
            if par_config not in measurements[0]['config']['parameters']:
                measurements[0]['config']['parameters'].append(par_config)
    
    # Construct the final workspace dictionary
    workspace = {
        'channels': channels,
        'measurements': measurements,
        'observations': observations,
        'version': version
    }

    return workspace


# for samp_index, sample in enumerate(workspace['channels'][ch_index]['samples']):
#     sample = {'name': 'new_sample'}  # This would not update the list in `workspace`
# Using sample as a reference to the list element is perfectly fine and will not cause bugs 
# as long as you're modifying the contents of sample (like updating values or appending to a list). 
# However, be cautious when assigning a completely new value to sample itself, 
# as that won't update the original list.

def extract_temp_asimov_channels(workspace: dict, mc_uncer: bool = True) -> list:
    """
    Extracts `temp_asimov_channels` from a workspace.

    Parameters:
        workspace (dict): The workspace from which to extract templates and Asimov data.
        mc_uncer (bool, optional): Whether to include statistical uncertainties. Default is True.

    Returns:
        list: A list of tuples for each channel:
            - template_flat (dict): Flattened templates for each sample as unp.array.
            - asimov_data (unp.array): Asimov data as unp.array.
    """
    temp_asimov_channels = []

    for ch_index, channel in enumerate(workspace['channels']):
        # Extract flattened templates
        template_flat = {}
        for sample in channel['samples']:
            # Extract nominal values and uncertainties
            nominal_values = np.array(sample['data'])
            if mc_uncer:
                # Find the staterror modifier for uncertainties
                staterror_mod = next((m for m in sample['modifiers'] if m['type'] == 'staterror'), None)
                if staterror_mod:
                    uncertainties = np.array(staterror_mod['data'])
                else:
                    uncertainties = np.sqrt(nominal_values)
            else:
                uncertainties = np.sqrt(nominal_values)
            # Store as unp.array
            template_flat[sample['name']] = unp.uarray(nominal_values, uncertainties)

        # Extract Asimov data
        asimov_data_nominal = np.array(workspace['observations'][ch_index]['data'])
        asimov_data_uncertainties = np.sqrt(asimov_data_nominal)  # Default uncertainties to poisson
        asimov_data = unp.uarray(asimov_data_nominal, asimov_data_uncertainties)

        # Append the reconstructed channel to the list
        temp_asimov_channels.append((template_flat, asimov_data))

    return temp_asimov_channels

def inspect_temp_asimov_channels(t1, t2=None):
    """
    Inspect and compare the templates and Asimov data for multiple channels.

    Parameters:
        t1 (list): First list of channel data, where each element is a tuple:
            - template_flat (dict): Flattened templates for each sample as unp.array.
            - asimov_data (unp.array): Asimov data as unp.array.
        t2 (list, optional): Second list of channel data for comparison, structured like `t1`. Default is None.

    Returns:
        None: Prints the inspection and comparison results to the console.

    Notes:
        - If `t2` is provided, the function compares the templates and Asimov data in `t1` and `t2`.
        - The function checks for equality of `unp.array` objects in both inputs using `np.array_equal`.
        - Outputs differences in templates and Asimov data for mismatched channels.
    """
    for ch_index, (template_flat, asimov_data) in enumerate(t1):
        print(f"Channel {ch_index}:")
        for name, data in template_flat.items():
            print(f"  Sample: {name}, Data: {data}")
            if t2 is not None:
                nominal1 = unp.nominal_values(data)
                nominal2 = unp.nominal_values(t2[ch_index][0][name])
                std1 = unp.std_devs(data)
                std2 = unp.std_devs(t2[ch_index][0][name])
                if np.array_equal(nominal1, nominal2) and np.array_equal(std1, std2):
                    print(colored(f'    {name} templates are equal in the 2 inputs','green'))
                else:
                    print(colored(f'    {name} templates are different in the 2 inputs','red'))
                    print(colored(f'    {np.array_equal(nominal1, nominal2)=}, {np.array_equal(std1, std2)=}','red'))
                    print(f"    Sample: {name}, Data (from t2): {t2[ch_index][0][name]}")
        print(f"  Asimov Data: {asimov_data}")
        if t2 is not None:
            nominal1 = unp.nominal_values(asimov_data)
            nominal2 = unp.nominal_values(t2[ch_index][1])
            std1 = unp.std_devs(asimov_data)
            std2 = unp.std_devs(t2[ch_index][1])
            if np.array_equal(nominal1, nominal2) and np.array_equal(std1, std2):
                print(colored('    Asimov data are equal in the 2 inputs','green'))
            else:
                print(colored('    Asimov data are different in the 2 inputs','red'))
                print(colored(f'    {np.array_equal(nominal1, nominal2)=}, {np.array_equal(std1, std2)=}','red'))
                print(f"    Asimov Data (from t2): {t2[ch_index][1]}")



# # +
################################ 1d fit #############################
from iminuit import cost, Minuit
from scipy.stats import norm
from scipy.integrate import quad

class polynomial:
    def __init__(self, par, x_min, x_max):
        self.par = par
        self.x_min = x_min
        self.x_max = x_max
        
    def function(self, x):
        return np.polyval(p=self.par, x=x)
    
    def pdf(self, x):
        # Compute the normalization constant
        normalization_constant, _ = quad(self.function, self.x_min, self.x_max)

        # Now normalize the polynomial
        return self.function(x) / normalization_constant
    
    def cdf(self, x):
        normalization_constant, _ = quad(self.function, self.x_min, self.x_max)

        # Calculate the cumulative distribution function for each x
        def cumulative_value(x_val):
            return quad(self.function, self.x_min, x_val)[0] / normalization_constant

        # Apply the cumulative_value function to each element of x
        return np.array([cumulative_value(val) for val in np.atleast_1d(x)])


def poly_integral_ufloat(coeffs, x0, x1):
    """
    Given polynomial coefficients in decreasing order of powers, 
    compute the definite integral from x0 to x1 analytically.

    For coeffs = [a0, a1, a2, ..., aN] (a0 * x^N + a1 * x^(N-1) + ... + aN),
    the indefinite integral F(x) is:
      a0/(N+1) * x^(N+1) + a1/(N) * x^N + ... + aN * x
    We return F(x1) - F(x0).

    coeffs can be either floats or ufloat's (with correlations).
    The returned value is float or ufloat accordingly.
    """
    # Highest power is len(coeffs)-1
    N = len(coeffs) - 1

    def F(x):
        # Build sum_{k=0..N} [ coeffs[k] * x^(N-k+1)/(N-k+1) ]
        # indexing: k=0 => power = N
        # So the exponent in x is (N-k+1), the coefficient is coeffs[k] / (N-k+1).
        s = 0
        for k, ak in enumerate(coeffs):
            power = N - k + 1
            # If power == 0, that means the constant's integral => ak * x
            # but in practice power should go from N+1 down to 1
            s += ak * (x**power) / power
        return s

    return F(x1) - F(x0)

    
def poly_coeffs_from_result(result, num_poly_params):
    """
    Extract the last 'num_poly_params' coefficients from `result` as a list 
    in decreasing power order, suitable for np.polyval.
    """
    # e.g. if num_poly_params == 2, we extract result[-2:] in decreasing order
    return list(result[-num_poly_params:])  # already in decreasing order in your code


class fit_Dmass:
    def __init__(self,x_edges, hist, poly_only):
        self.x_edges = x_edges
        self.y_val = unp.nominal_values(hist)
        self.y_err = unp.std_devs(hist)
        self.x_min = min(x_edges)
        self.x_max = max(x_edges)
        self.poly_only = poly_only

    # np.polynomial.Polynomial.fit and np.polyval handle the order of polynomial coefficients differently.
    # np.polyval expects the coefficients in decreasing order of powers, i.e., from the highest degree term to the constant term.
    # np.polynomial.Polynomial stores the coefficients in increasing order of powers (from the constant term to the highest degree).
        
    # fit polynomial
    def gauss_polyno(self, x, par):
        return par[0] * norm.pdf(x, par[1], par[2]) + np.polyval(par[3:], x)# for len(par) == 2, this is a line
    
    def gauss_poly_cdf(self, x, *par):
        sig_gauss = par[0] * norm.cdf(x, par[1], par[2])
        bkg_poly = par[3] * polynomial(par[4:],self.x_min,self.x_max).cdf(x)
        return bkg_poly + sig_gauss
        
    def estimate_init(self, x, y, deg):
        # polynomial
        sideband_mask = (x < 1.822) | (1.92 < x)
        p = np.polynomial.Polynomial.fit(x[sideband_mask], y[sideband_mask], deg=deg)
        init = p.convert().coef[::-1] # Reverse the coefficient order
        p_init = tuple([round(i,1) for i in init])
        # gaussian
        mean = np.average(x, weights=y)
        variance = np.average((x - mean)**2, weights=y)
        std = float(np.sqrt(variance))
        
        return round(mean,2), round(std,2), p_init
    
    def fit_gauss_poly_LS(self, deg,loss='linear', x=None, y_val=None, y_err=None):#'soft_l1'
        # get starting values
        if x is None:
            x = self.x_edges[1:]
        if y_val is None:
            y_val = self.y_val
            y_err = self.y_err
        g_mean, g_std, p_init = self.estimate_init(x,y_val,deg)
        norm_estimate = round(y_val.sum() * np.diff(x)[0], 1)
        init = np.array([norm_estimate, g_mean, g_std, *p_init])
        print('initial parameters=', init)
        
        # cost function and minuit
        c = cost.LeastSquares(x,y_val,y_err,model=self.gauss_polyno,loss=loss)
        m = Minuit(c, init)

        # fit the bkg in sideband first
        m.limits["x0", "x1", "x2"] = (0, None)
        m.fixed["x0", "x1", "x2"] = True
        if self.poly_only:
            m.values["x0"] = 0
        # temporarily mask out the signal
        c.mask = (x < 1.82) | (1.92 < x)
        m.simplex().migrad()
        
        if not self.poly_only:
            # fit the signal with the bkg fixed
            c.mask = (x < 1.822) | ((1.854 < x) & (x < 1.886)) | (1.92 < x) # include the signal
            m.fixed = False  # release all parameters
            m.fixed["x3","x4"] = True  # fix background amplitude
            m.simplex().migrad()

            # fit everything together to get the correct uncertainties
            m.fixed = False
            m.migrad()
        
        # fit result
        result = correlated_values(m.values, m.covariance)
        return m, c, result
    
    def fit_gauss_poly_ML(self, deg, xe=None, hist=None):
        # get starting values
        if xe is None:
            xe = self.x_edges
        if hist is None:
            hist = self.y_val
        g_mean, g_std, p_init = self.estimate_init(xe[1:],hist,deg)
        norm_estimate = round(hist.sum() * np.diff(xe)[0], 1)
        init = np.array([norm_estimate, g_mean, g_std, round(hist.sum(),1),*p_init])
        print('initial parameters=', init)
            
        # cost function and minuit
        c = cost.ExtendedBinnedNLL(n=hist,xe=xe,scaled_cdf=self.gauss_poly_cdf) 
        m = Minuit(c, *init)
        
        # fit the bkg in sideband first
        m.limits["x0", "x1", "x2"] = (0, None)
        m.fixed["x0", "x1", "x2"] = True
        if self.poly_only:
            m.values["x0"] = 0
        # temporarily mask out the signal
        x_re = xe[1:] # right edge
        c.mask = (x_re < 1.82) | (1.92 < x_re)
        m.simplex().migrad()

        if not self.poly_only:
            # fit the signal with the bkg fixed
            c.mask = (x_re < 1.822) | ((1.854 < x_re) & (x_re < 1.886)) | (1.92 < x_re) # include the signal
            m.fixed = False  # release all parameters
            m.fixed["x3","x4","x5"] = True  # fix background amplitude
            m.simplex().migrad()

            # fit everything together to get the correct uncertainties
            m.fixed = False
            m.migrad()
        
        result = correlated_values(m.values, m.covariance)
        return m, c, result

    
    def poly_integral(self, xrange, result):
        """
        Compute the integral over 'xrange' of the polynomial part 
        (with full uncertainty propagation) using the fitted parameters `result`.
        """
        x0, x1 = xrange

        # --------------------
        # Case 1: len(result) == 5
        # --------------------
        # Typically means: [A_gauss, mu, sigma, p0, p1]
        # i.e. only 2 polynomial coefficients -> a linear polynomial
        if len(result) == 5:
            # Extract the polynomial part (the last 2 parameters)
            poly_pars = poly_coeffs_from_result(result, num_poly_params=2)  # p0, p1 in decreasing order
            # Do the exact integral of that polynomial from x0 to x1
            area_ufloat = poly_integral_ufloat(poly_pars, x0, x1)

            # area_ufloat is a ufloat, so you can extract nominal value and std dev as needed:
            area_nom  = unp.nominal_values(area_ufloat)
            area_std  = unp.std_devs(area_ufloat)

            print(f"Area under polynomial from {x0} to {x1} = {area_nom:.3f} ± {area_std:.3f}")
            return area_ufloat

        # --------------------
        # Case 2: len(result) == 6
        # --------------------
        # Typically means: [A_gauss, mu, sigma, N_poly, p0, p1]
        # i.e. 2 polynomial coefficients plus an amplitude factor par[-3]
        # Then your code uses: yields = par[-3]* polynomial(par[-2:], ...).cdf(x)
        else:
            # The "amplitude" scaling factor in front of the polynomial:
            scale = result[-3]  
            # The actual polynomial coefficients:
            poly_pars = poly_coeffs_from_result(result, num_poly_params=2)

            # We want the fraction of the *normalized polynomial* between x0 and x1.
            #   cdf(x) = [ ∫(p(x') dx' from x_min to x ) ] / [ ∫(p(x') dx' from x_min to x_max ) ]
            # Then multiplied by 'scale'.
            #
            # We'll do that analytically as well:
            # Let F(x) = ∫(p(x') dx') from x_min up to x (the indefinite integral minus F(x_min)).
            # Let DEN = F(x_max) - F(x_min).
            # cdf(x) = [F(x) - F(x_min)] / DEN.
            # The "yield" from x0 to x1 is scale * [cdf(x1) - cdf(x0)].

            # 1) Compute total polynomial integral from x_min to x_max
            poly_total = poly_integral_ufloat(poly_pars, self.x_min, self.x_max)

            # 2) Function that returns the integral from x_min up to x
            def F(x):
                return poly_integral_ufloat(poly_pars, self.x_min, x)

            # cdf(x)
            def poly_cdf(x):
                return (F(x) / poly_total)

            # The yield from x0..x1 is scale * [cdf(x1) - cdf(x0)]
            yields_ufloat = scale * (poly_cdf(x1) - poly_cdf(x0))

            yield_nom = unp.nominal_values(yields_ufloat)
            yield_std = unp.std_devs(yields_ufloat)
            print(f"Yields from {x0} to {x1} = {yield_nom:.3f} ± {yield_std:.3f}")
            return yields_ufloat


#     def plot_result(self, x, y, yerr, result):
#         # Generate x, y values for plotting the fitted function
#         x_plot = np.linspace(min(x), max(x), 500)
#         y_plot = self.polyno(x_plot, result)

#         # Calculate y and residual for plotting the residual plot
#         y_fit = self.polyno(x, result)
#         y_data = unp.uarray(y, yerr)
#         residual = y_data - y_fit

#         # Create a figure with two subplots: one for the histogram, one for the residual plot
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), gridspec_kw={'height_ratios': [5, 1]})

#         # Plot data points and fitted function in ax1
#         ax1.errorbar(x, y, yerr, fmt='o', label='Data')
#         ax1.plot(x_plot, unp.nominal_values(y_plot), label='Fitted polynomial', color='red')
#         ax1.grid()
#         ax1.legend()
#         ax1.set_ylabel('n Events per bin')
#         #plt.ylim(0,1)

#         # Plot the residuals in ax2
#         ax2.errorbar(x, unp.nominal_values(residual), yerr=unp.std_devs(residual), fmt='o', color='black')
#         # Add a horizontal line at y=0 for reference
#         ax2.axhline(0, color='gray', linestyle='--')
#         # Label the residual plot
#         ax2.set_ylabel('Residuals')
#         # ax2.set_xlabel(f'{variable}')

#         # Adjust the layout to avoid overlapping of the subplots
#         plt.tight_layout()
#         # Show the plot
#         plt.show()


class fit_pull_linearity:
    @classmethod
    def gauss(cls, x, mu, sigma):  # cls refers to the class itself
        return norm.pdf(x, mu, sigma)

    @classmethod
    def polyno(cls, x, par):
        return np.polyval(par, x)  # for len(par) == 2, this is a line

    @classmethod
    def line(cls, x, x0, x1):
        return x0 + x1*x

    @staticmethod
    def fit_gauss(x):
        # get starting values:
        mean = np.mean(x)
        std = np.std(x)

        # cost function and minuit
        cost_gauss = cost.UnbinnedNLL(data=x, pdf=fit_pull_linearity.gauss)
        m_gauss = Minuit(fcn=cost_gauss, mu=round(mean,1), sigma=round(std,1))
        m_gauss.migrad()

        # fit result
        result = correlated_values(m_gauss.values, m_gauss.covariance)
        # correlated_values will keep the correlation between mu, sigma
        return result # mu, sigma

    @staticmethod
    def fit_linear(x, y, yerr):
        # get starting values
        p = np.polynomial.Polynomial.fit(x, y, deg=1)
        y_intercept, slope = p.convert().coef

        # cost function and minuit
        cost_poly = cost.LeastSquares(x,y,yerr,model=fit_pull_linearity.polyno,loss='soft_l1')
        m_line = Minuit(cost_poly, (round(y_intercept,1),round(slope,1)) )
        m_line.migrad()

        # fit result
        result = correlated_values(m_line.values, m_line.covariance)
        return result # slope, y_int if model==polyno; y_int, slope if model==line



############################### pyhf utils ####################################
import pyhf
import cabinetry
import json
from tqdm.auto import tqdm

class pyhf_utils:
    def __init__(self, toy_temp, fit_temp, par_fix=['bkg_fakeTracks_norm']):
                 # toy_pars=[1]*12, fit_inits = [0.4]*12  not used in code
        
        # load toy and fit templates
        tt = cabinetry.workspace.load(toy_temp)
        ft = cabinetry.workspace.load(fit_temp)
        model_toy, _ = cabinetry.model_utils.model_and_data(tt)
        model_fit, _ = cabinetry.model_utils.model_and_data(ft)
        
        # Get norm parameter names in the correct order
        norm_parameter_names = [par for par in model_toy.config.par_order if par.endswith('_norm')]
        # Create a boolean list for fixing parameters
        fix_mask = [par in par_fix for par in norm_parameter_names]
        
        # set up the parameter configuration
        for i, par_name in enumerate(norm_parameter_names):
            # model_toy.config.param_set(par_name).suggested_init=[toy_pars[i]]
            # model_toy.config.param_set(par_name).suggested_fixed=fix_mask[i]
            # model_fit.config.param_set(par_name).suggested_init=[fit_inits[i]]
            model_fit.config.param_set(par_name).suggested_fixed=fix_mask[i]

        # setup init attributes
        self.model_toy = model_toy
        self.model_fit = model_fit
        # Set the pars for generating toys or default `model_toy.config.suggested_init()`
        self.toy_pars = cabinetry.model_utils.asimov_parameters(model_toy) # returns ndarray
        # Create a list of parameter names for samples that are not fixed
        self.minos_pars = [par for par in norm_parameter_names if par not in par_fix]

    def toyAndFit_inits(self):
        norm_parameter_names = [par for par in self.model_toy.config.par_order if par.endswith('_norm')]
        nNorm_pars = len(norm_parameter_names)
        fit_inits_toSave = self.model_fit.config.suggested_init()[:nNorm_pars]
        
        return self.toy_pars.tolist(), fit_inits_toSave
        
    def toy_generator(self, n_toys):
        pdf_toy = self.model_toy.make_pdf(pyhf.tensorlib.astensor(self.toy_pars))
        toys = pdf_toy.sample((n_toys,))

        return toys
        
    def fit_scipy_minuit(self, data):
        try: # fit with scipy for an initial guess
            pyhf.set_backend('numpy', 'scipy')
            init_pars = pyhf.infer.mle.fit(pdf=self.model_fit, data=data).tolist()

        except Exception as e: # use the suggested init
            init_pars = self.model_fit.config.suggested_init()

        # fit with minuit
        pyhf.set_backend('numpy', 'minuit')
        result = cabinetry.fit.fit(model=self.model_fit,data=data,
                                   init_pars=init_pars,goodness_of_fit=True,
                                   minos=self.minos_pars
                                  )
        
        return result
        

class toy_utils:
    def __init__(self,pars_toFix=[],toy_workspace='',
                 part=0,binning=[],fit_workspace=''):
        
        self.part = part
        self.binning = binning
        self.pars_toFix = pars_toFix
        self.toy_workspace = toy_workspace
        self.fit_workspace = toy_workspace if fit_workspace=='' else fit_workspace
        
    def generate_fit_toys(self,n_toys):
        # initialize util tools for pyhf
        pyhf_tools = pyhf_utils(toy_temp=self.toy_workspace,
                                     par_fix=self.pars_toFix,
                                     fit_temp=self.fit_workspace,
                                     # toy_pars=toy_pars,
                                     # fit_inits = fit_inits
                                   )
        # generate toys
        toys = pyhf_tools.toy_generator(n_toys=n_toys)

        # retrive the toy and fit initials
        toy_pars, fit_inits = pyhf_tools.toyAndFit_inits()
        
        # prepare containers for fit results
        fit_results = {
            'best_twice_nll': [],
            'pval': [],
            'expected_results':[],
            'best_fit': [],
            'hesse_uncertainty': [],
            'minos_uncertainty_up': [],
            'minos_uncertainty_down': []
        }
                
        failed_fits = 0
        attempted_fits = 0
        successful_fits = 0
        
        # fit toys
        with tqdm(total=n_toys, desc='Fitting toys') as pbar:
            while attempted_fits < n_toys:
                data = toys[attempted_fits]
                try:
                    # fit
                    res = pyhf_tools.fit_scipy_minuit(data=data)
                    
                    # save fit results
                    fit_results['best_twice_nll'].append(res.best_twice_nll)
                    fit_results['pval'].append(res.goodness_of_fit)
                    fit_results['expected_results'].append(fit_inits)
                    fit_results['best_fit'].append(res.bestfit[:len(fit_inits)])
                    fit_results['hesse_uncertainty'].append(res.uncertainty[:len(fit_inits)])
#                     main_data, aux_data = model.fullpdf_tv.split(pyhf.tensorlib.astensor(data))
#                     fit_results['main_data'].append(main_data.tolist())
#                     fit_results['aux_data'].append(aux_data.tolist())
                    
                    # save minos results
                    all_pars = res.labels[:len(fit_inits)]
                    # get minos if res.minos_unc has keys in all_pars, otherwise get 1
                    fit_results['minos_uncertainty_up'].append(
                        [abs(res.minos_uncertainty.get(x,[1,1])[1]) for x in all_pars])
                    fit_results['minos_uncertainty_down'].append(
                        [abs(res.minos_uncertainty.get(x,[1,1])[0]) for x in all_pars])

                    successful_fits += 1
                    pbar.update(1)

                except Exception as e:
                    failed_fits += 1
                    print(f"Fit failed: {e}")
                attempted_fits += 1
            pbar.close()

        for key in fit_results.keys():
            # convert to json safe lists (these are much quicker to load then the yaml files later)
            fit_results[key] = np.array(fit_results[key]).tolist()

        out_dict = {
            'poi': res.labels[:len(fit_inits)],
            'toy_pars': toy_pars,
            'n_toys': n_toys,
            'results': fit_results,
            'failed_fits': failed_fits,
            'attempted_fits': attempted_fits,
            'part': self.part,
            'binning': self.binning,
        }
        
        return out_dict
    
    @staticmethod
    def merge_toy_results(result_files):
        merged_toy_results_dict = {}
        failed_fits = 0
        for input_file in tqdm(result_files):
            with open(input_file, 'r') as f:
                try: 
                    in_dict = json.load(f)
                except json.JSONDecodeError as e:
                    failed_fits += 10
                    print(f"Error decoding JSON: {e}")
                    continue
                    
                failed_fits += in_dict['failed_fits']
                merged_toy_results_dict['poi'] = in_dict['poi']
                
                for k, v in in_dict['results'].items():
                    if not k in merged_toy_results_dict.keys():
                        merged_toy_results_dict[k] = v
                    else:
                        merged_toy_results_dict[k].extend(v)
                        
        out_dict = {
            'toy_results': merged_toy_results_dict,
            'failed_fits': failed_fits
        }
        
        return out_dict
    
    @staticmethod
    def calculate_pulls(merged_dict, normalize=True, minos_error=True):
        merged_results = merged_dict['toy_results']
        
        fitted = np.array(merged_results['best_fit'])
        truth = np.array(merged_results['expected_results'])
        diff = fitted - truth

        # calculate pulls
        if minos_error:
            # minos errors
            minos_up = np.array(merged_results['minos_uncertainty_up'])
            minos_down = np.array(merged_results['minos_uncertainty_down'])
            pulls = np.where(diff > 0, diff / minos_up, diff / minos_down)
            error_toplot = np.where(diff > 0, minos_up, minos_down) # will show in the plot
        else:
            # hesse error
            hesse_error = np.array(merged_results['hesse_uncertainty'])
            pulls = diff / hesse_error
            error_toplot = hesse_error # will show in the plot
        
        if not normalize:
            pulls = diff
            
        # save
        merged_dict['toy_results']['pulls']=pulls.tolist()
        merged_dict['toy_results']['error_toplot']=error_toplot.tolist()
        
        return merged_dict
    
    @staticmethod
    def calculate_linear_xy(merged_dict, minos_error=True):
        merged_results = merged_dict['toy_results']
        
        truth = np.array(merged_results['expected_results'])
        fitted = np.array(merged_results['best_fit'])
        diff = fitted - truth
        
        # choose hesse or minos error
        if minos_error:
            # minos errors
            minos_up = np.array(merged_results['minos_uncertainty_up'])
            minos_down = np.array(merged_results['minos_uncertainty_down'])
            error = np.where(diff > 0, minos_up, minos_down)
        else:
            # hesse error
            error = np.array(merged_results['hesse_uncertainty'])
        
        # Find unique rows in truth, every N toys share the same truth
        unique_truth, unique_indices_in_truth, inverse_indices = np.unique(truth,axis=0, 
                                                                           return_index=True, 
                                                                           return_inverse=True)

        # Compute weighted mean and SEM for each unique truth value
        weighted_means = []
        SEM_values = []

        for i in range(len(unique_truth)):
            mask = (inverse_indices == i)  # Get indices for each unique row
            fitted_group = fitted[mask]
            error_group = error[mask]

            # Compute weights (w = 1 / sigma^2)
            weights = 1 / (error_group**2)

            # Weighted mean
            weighted_mean = np.sum(fitted_group * weights, axis=0) / np.sum(weights, axis=0)

            # Standard error of the mean (SEM)
            SEM = np.sqrt(1 / np.sum(weights, axis=0))

            weighted_means.append(weighted_mean)
            SEM_values.append(SEM)
        
        # Convert and save
        merged_dict['toy_results']['truth']=unique_truth.tolist()
        merged_dict['toy_results']['weighted_means']=np.array(weighted_means).tolist()
        merged_dict['toy_results']['SEM']=np.array(SEM_values).tolist()
        
        return merged_dict
    
    @staticmethod
    def plot_toy_gaussian(x: list, mu:ufloat,sigma: ufloat,
                          file_name: str,vertical_lines: list = [0],
                          extra_info=None, title_info=None, ylabel='Trials',
                          xlabel: str = r'$(\mu-\mu_{in}) /\sigma_{\mu}$',
                          figsize=(6, 6 / 1.618), show: bool = False):
        # set up the figure
        fig = plt.figure(figsize=figsize)
        bins = np.linspace(-5 * sigma.n, +5 * sigma.n, 101)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        bin_width = bins[1] - bins[0]
        
        # set up the fitted gaussian
        def gaussian(x, mu, sigma): # user defined gauss for uncertainties
            return 1. / (((2. * np.pi)**0.5) * sigma) * np.e**(-(((x - mu) / sigma)**2) / 2)
        gauss_x = np.linspace(bins[0], bins[-1], 2001)
        gauss_y = gaussian(gauss_x, mu, sigma)
        gauss_y_nominal = unp.nominal_values(gauss_y)
        gauss_y_std = unp.std_devs(gauss_y)

        # calculate the error band
        hist, _ = np.histogram(x, bins=bins)
        norm_nominal = hist.sum() * bin_width * gauss_y_nominal
        norm_up = hist.sum() * bin_width * (gauss_y_nominal + gauss_y_std)
        norm_down = hist.sum() * bin_width * (gauss_y_nominal - gauss_y_std)
        
        # plot the gauss curve, error band, and data points with errorbar
        gauss_curve = plt.plot(gauss_x, norm_nominal, lw=1)
        plt.fill_between(gauss_x, norm_up, norm_down, color=gauss_curve[0].get_color(), alpha=0.3)
        plt.errorbar(x=bin_centers,y=hist,yerr=poisson_error(hist),fmt='.',color='black',
                     markeredgecolor='white',markeredgewidth=0.5)

        # set up reference line and text
        for v in vertical_lines:
            plt.axvline(v, color='gray', ls='--', zorder=-100)

        # display text for fitted parameters
        plt.text(0.95,0.95, fr'$\mu_{{G}}=${round(mu.n,3)}$\pm${round(mu.s,3)}',
                 va='top',ha='right',usetex=False, transform=plt.gca().transAxes)
        plt.text(0.95, 0.88, fr'$\sigma_{{G}}=${round(sigma.n,3)}$\pm${round(sigma.s,3)}',
                 va='top', ha='right', usetex=False, transform=plt.gca().transAxes)

        if title_info is not None:
            plt.title(title_info, loc='right')
        if extra_info is not None:
            plt.text(0.05, 0.95, extra_info, va='top', ha='left', usetex=False, 
                     transform=plt.gca().transAxes, fontsize=10)
        
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.ylim(0)
        plt.xlim(bins[0], bins[-1])
        plt.savefig(file_name, bbox_inches='tight')
        
        if show:
            plt.show()
        plt.close()

    @staticmethod
    def plot_linearity_test(x:list, y: list, yerr: list,
                            slope: ufloat,intercept: ufloat,
                            file_name: str, bonds: list = [0,1],
                            x_offset: list = [0],
                            extra_info=None, title_info=None,
                            xlabel= r'$\mu_{in}$', ylabel=r'$\mu$',
                            figsize=(6, 6 / 1.618), show: bool = False):
        # set up the figure
        plt.figure(figsize=figsize)
        x_array_line = np.linspace(bonds[0], bonds[1], 1001)

        # plot the fitted line and data point with errorbar
        y_line = x_array_line * slope + intercept
        y_nominal = unp.nominal_values(y_line)
        y_std = unp.std_devs(y_line)
        line = plt.plot(np.array([x_array_line[0], x_array_line[-1]]) + x_offset,
                        np.array([y_nominal[0], y_nominal[-1]]), lw=1.0)
        plt.fill_between(x=x_array_line+x_offset, y1=y_nominal+y_std, y2=y_nominal-y_std,
                         color=line[0].get_color(),alpha=0.3)
        plt.errorbar(x=np.array(x) + x_offset, y=np.array(y),yerr=yerr,fmt='.',color='black',
                     markeredgecolor='white',markeredgewidth=0.5, label=None)
        
        # set up extra reference line and text
        plt.plot(np.array([bonds[0], bonds[1]]) + x_offset, [bonds[0], bonds[1]], color='gray', label='Diagonal', lw=0.5, zorder=-100, ls='--')
        plusminus = '+' if intercept >= 0 else '-'
        eq = fr"""({round(slope.n,3)}$\pm${round(slope.s,3)})$\mu_{{in}}$${plusminus}$({abs(round(intercept.n,3))}$\pm${round(intercept.s,3)})"""
        plt.text(0.02, 0.85, eq, usetex=False,color=line[0].get_color(),
                 transform=plt.gca().transAxes, fontsize=9)

        if extra_info is not None:
            plt.text(0.05, 0.95, extra_info, va='top', ha='left', usetex=False, 
                     transform=plt.gca().transAxes, fontsize=12)

        left, right = plt.xlim()
        plt.xlim(left, right + 0.1 * (right - left))
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if title_info is not None:
            plt.title(title_info, loc='right')
        plt.legend()
        plt.savefig(file_name, bbox_inches='tight')
        
        if show:
            plt.show()
        plt.close()

# # +
