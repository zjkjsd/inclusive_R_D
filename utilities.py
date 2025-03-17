# -*- coding: utf-8 -*-
# region
############################## define relevant variables ########################

spectators = ['__weight__', 'D_CMS_p', 'ell_CMS_p', 'B0_CMS3_weMissM2', 'B0_CMS3_weQ2lnuSimple']

CS_variables = ["B0_R2",       "B0_thrustOm",   "B0_cosTBTO",    "B0_cosTBz",
                "B0_KSFWV3",   "B0_KSFWV4",     "B0_KSFWV5",     "B0_KSFWV6",
                "B0_KSFWV7",   "B0_KSFWV8",     "B0_KSFWV9",     "B0_KSFWV10",
                "B0_KSFWV13",  "B0_KSFWV14",    "B0_KSFWV15",    "B0_KSFWV16",
                "B0_KSFWV17",  "B0_KSFWV18",]   
#                 "B0_thrustBm", "B0_KSFWV1",     "B0_KSFWV2",     "B0_KSFWV11", 
#                 "B0_KSFWV12" correlates with mm2 or p_D_l

DTC_variables = ['D_vtxReChi2',                'D_A1FflightDistanceSig_IP',
                 'D_daughterInvM_1_2',         'D_daughterInvM_0_1',]

B_variables = ['B0_vtxReChi2',               'B0_dr',   
               'B0_CMS_cos_angle_0_1',       'B0_D_l_DisSig',
               'B0_roeMbc_my_mask',          'B0_roeDeltae_my_mask',
               'B0_roeEextra_my_mask',       'B0_TagVReChi2IP',]

training_variables = CS_variables + DTC_variables + B_variables
mva_variables = training_variables + spectators

analysis_variables=['__experiment__',     '__run__',       '__event__',      '__production__',
                    'B0_isContinuumEvent','B0_mcPDG',      'B0_mcErrors',    'B0_mcDaughter_0_PDG',
                    'B0_mcDaughter_1_PDG','B0_deltaE',     'B0_Mbc',         'B0_CMS2_weMbc', 
                    'B0_CMS0_weDeltae',   'B0_cos_angle_0_1',   
                    'D_mcErrors',         'D_genGMPDG',    'D_genMotherPDG', 'D_mcPDG',
                    'D_BFM',              'D_M',           'D_p',            
                    'D_K_mcErrors',       'D_pi1_mcErrors','D_pi2_mcErrors', 'D_K_charge',
                    'D_K_cosTheta',       'D_K_p',         'D_K_PDG',        'D_K_mcPDG',
                    'ell_genMotherPDG',   'ell_mcPDG',     'ell_mcErrors',   'ell_genGMPDG',
                    'ell_p',              'ell_pValue',    'ell_charge',     'ell_theta',
                    'ell_PDG',            'ell_eID',
                    'mode',               'Ecms',          'p_D_l',          'B_D_ReChi2',
                    'sig_prob',           'fakeD_prob',    'fakeB_prob',     'continuum_prob',]
#                  'D_K_kaonIDNN',        'D_K_pionIDNN',  'D_pi2_kaonIDNN', 'D_pi2_pionIDNN',
#                  'D_pi1_kaonIDNN',      'D_pi1_pionIDNN',] 'B0_Lab5_weMissPTheta','B0_roeCharge_my_mask', 
#                'B0_nROE_Tracks_my_mask',  'B0_nROE_Photons_my_mask',  'B0_nROE_NeutralHadrons_my_mask',

combinatorial_vars = [
        'D_511_0_daughterPDG', 'D_511_1_daughterPDG', 'D_511_2_daughterPDG',
        'D_511_3_daughterPDG', 'D_511_4_daughterPDG', 'D_511_5_daughterPDG',
        'D_511_6_daughterPDG', 'D_521_0_daughterPDG', 'D_521_1_daughterPDG', 
        'D_521_2_daughterPDG', 'D_521_3_daughterPDG', 'D_521_4_daughterPDG',
        'D_521_5_daughterPDG', 'D_521_6_daughterPDG'
    ]

veto_vars = ['B0_DstVeto_massDiff_0','B0_DstVeto_massDiffErr_0',
             'B0_DstVeto_massDiffSignif_0','B0_DstVeto_vtxReChi2',
             'B0_DstVeto_isDst']

all_relevant_variables = mva_variables + analysis_variables + combinatorial_vars + veto_vars

DecayMode_new = {'bkg_fakeTracks':0,         'bkg_fakeD':1,           'bkg_TDFl':2,
                 'bkg_continuum':3,          'bkg_combinatorial':4,   'bkg_singleBbkg':5,
                 'bkg_other_TDTl':6,         'bkg_other_signal':7,
                 r'$D\tau\nu$':8,            r'$D^\ast\tau\nu$':9,    r'$D\ell\nu$':10,
                 r'$D^\ast\ell\nu$':11,                r'$D^{\ast\ast}\tau\nu$':12,
                 r'$D^{\ast\ast}\ell\nu$_narrow':13,   r'$D^{\ast\ast}\ell\nu$_broad':14,
                 r'$D\ell\nu$_gap_pi':15,              r'$D\ell\nu$_gap_eta':16}

# sidebands / signal region
r_D = 753/89529
r_Dst = 557/57253

# for event classification
pi_pdg = [111, 211, -211]
eta_pdg = [221]

D_Dst_pdg = [411, 413, -411, -413]
Dstst_narrow_pdg = [10413, 10423, 415, 425, -10413, -10423, -415, -425]
Dstst_broad_pdg  = [10411, 10421, 20413, 20423, -10411, -10421, -20413, -20423]
Dstst_pdg   = Dstst_narrow_pdg + Dstst_broad_pdg

# for combinatorial classification
D_mesons_pdg = D_Dst_pdg + Dstst_pdg + [421,-421,423,-423,431,-431] # D0, D*0, D_s
charm_baryons_pdg = [4122, -4122, # Lambda_c
                     4112, -4112, 4212, -4212, # Sigma_c
                     4132, -4132, 4232, -4232, # Xi_c
                     4332, -4332, # Omega_c0
                    ]
single_charm_pdg = D_mesons_pdg + charm_baryons_pdg

double_charm_pdg = {30443, 9010443, # psi
                    4412, -4412, 4422, -4422, # Xi_cc+
                    4432, -4432, # Omega_cc+
                   }
leptons = {11, -11, 13, -13}
Bpdg = {511, -511, 521, -521}

################################ dataframe samples ###########################
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
import lightgbm as lgb

def apply_mva_bcs(df, features, cut, library='ag', version='',model=None,bcs='vtx',importance=False):
    # load model
    if library is not None:
        if library=='ag':
            predictor = TabularPredictor.load(f"/home/belle/zhangboy/inclusive_R_D/AutogluonModels/{version}")
            pred = predictor.predict_proba(df, model)
            pred = pred.rename(columns={0: 'sig_prob', 
                                        1: 'fakeD_prob',
                                        2: 'combinatorial_prob',
                                        3: 'continuum_prob'})

        elif library=='lgbm':
            predictor = lgb.Booster(model_file='/home/belle/zhangboy/inclusive_R_D/BDTs/LightGBM/lgbm_multiclass.txt')
            pred_array = predictor.predict(df[features], num_iteration=predictor.best_iteration)
            pred = pd.DataFrame(pred_array, columns=['sig_prob','fakeD_prob',
                                                     'combinatorial_prob','continuum_prob'])
            if importance: # feature importances
                lgb.plot_importance(predictor, figsize=(18,20))
        # combine the predict result
        pred['largest_prob'] = pred[['sig_prob','fakeD_prob','combinatorial_prob','continuum_prob']].max(axis=1)
        df_pred = pd.concat([df, pred], axis=1)

        # apply the MVA cut and BCS
        df_cut=df_pred.query(cut)
        
    else:
        df_cut = df.query(cut)
        
    if bcs=='vtx':
        df_bestSelected=df_cut.loc[df_cut.groupby(['__experiment__','__run__','__event__','__production__'])['B_D_ReChi2'].idxmin()]
    elif bcs=='mva':
        df_bestSelected=df_cut.loc[df_cut.groupby(['__experiment__','__run__','__event__','__production__'])['sig_prob'].idxmax()]
    
    return df_bestSelected


def classify_mc_dict(df, mode, template=True) -> dict:
    samples = {}
    lepton_PDG = {'e':11, 'mu':13}
    
    ################## Define D and lepton #################
    trueD = 'D_mcErrors==0'
    truel = f'abs(ell_mcPDG)=={lepton_PDG[mode]}'
    
    fakeD = '0<D_mcErrors<512'
    fakel = f'abs(ell_mcPDG)!={lepton_PDG[mode]} and ell_mcErrors!=512'
    
    fakeTracks = 'B0_mcErrors==512'
    
    ################# Define B ####################
    
    FD = f'{fakeD} and ({fakel} or {truel})' # i.e. Not a fakeTrack lepton
    TDFl = f'{trueD} and {fakel}'
    TDTl = f'{trueD} and {truel}'
    
    # more categories with TDTl
    continuum = f'{TDTl} and B0_isContinuumEvent==1'
    combinatorial = f'{TDTl} and B0_mcPDG==300553'
    signals = f'{TDTl} and (abs(B0_mcPDG)==511 or abs(B0_mcPDG)==521) and \
    (ell_genMotherPDG==B0_mcPDG or ell_genGMPDG==B0_mcPDG and abs(ell_genMotherPDG)==15)'
    singleBbkg = f'{TDTl} and B0_isContinuumEvent==0 and B0_mcPDG!=300553 and \
    ( (abs(B0_mcPDG)!=511 and abs(B0_mcPDG)!=521) or \
    ( ell_genMotherPDG!=B0_mcPDG and (ell_genGMPDG!=B0_mcPDG or abs(ell_genMotherPDG)!=15) ) )'
    
    # more categories with signals
    B2D_tau = f'{signals} and B0_mcDaughter_0_PDG*B0_mcDaughter_1_PDG==411*15'
    B2D_ell = f'{signals} and B0_mcDaughter_0_PDG*B0_mcDaughter_1_PDG==411*{lepton_PDG[mode]}'
    B2Dst_tau = f'{signals} and B0_mcDaughter_0_PDG*B0_mcDaughter_1_PDG==413*15'
    B2Dst_ell = f'{signals} and B0_mcDaughter_0_PDG*B0_mcDaughter_1_PDG==413*{lepton_PDG[mode]}'
    
    B2Dstst_tau = f'{signals} and B0_mcDaughter_0_PDG in @Dstst_pdg and abs(B0_mcDaughter_1_PDG)==15'
    B2Dstst_ell_narrow = f'{signals} and B0_mcDaughter_0_PDG in @Dstst_narrow_pdg and abs(B0_mcDaughter_1_PDG)=={lepton_PDG[mode]}'
    B2Dstst_ell_broad = f'{signals} and B0_mcDaughter_0_PDG in @Dstst_broad_pdg and abs(B0_mcDaughter_1_PDG)=={lepton_PDG[mode]}'

    B2D_ell_gap_pi = f'{signals} and B0_mcDaughter_0_PDG in @D_Dst_pdg and B0_mcDaughter_1_PDG in @pi_pdg'
    B2D_ell_gap_eta = f'{signals} and B0_mcDaughter_0_PDG in @D_Dst_pdg and B0_mcDaughter_1_PDG in @eta_pdg'
    
    ######################### Apply selection ###########################
    
    # Fake background components:
    samples.update({
        'bkg_fakeD': df.query(FD).copy(),
        'bkg_TDFl':  df.query(TDFl).copy(),
        'bkg_fakeTracks': df.query(fakeTracks).copy(),
    })
    
    # True Dl background components:
    bkg_continuum     = df.query(continuum).copy()
    bkg_combinatorial = df.query(combinatorial).copy()
    bkg_singleBbkg    = df.query(singleBbkg).copy()
    df_signals_all    = df.query(signals).copy()
    df_TDTl_all       = df.query(TDTl).copy()
    
    classified_TDTl_indices = pd.concat([bkg_continuum,bkg_combinatorial,
                                         bkg_singleBbkg,df_signals_all]).index
    
    bkg_other_TDTl = df_TDTl_all.loc[~df_TDTl_all.index.isin(classified_TDTl_indices)].copy()
#     bkg_other_TDTl = pd.concat([]).drop_duplicates(subset=['__experiment__', '__run__', '__event__', '__production__'], keep=False)
    
    samples.update({
        'bkg_continuum': bkg_continuum,
        'bkg_combinatorial': bkg_combinatorial,
        'bkg_singleBbkg': bkg_singleBbkg,
        'bkg_other_TDTl': bkg_other_TDTl,
    })
    
    # True Dl signal components:
    D_tau_nu     = df.query(B2D_tau).copy()
    D_l_nu       = df.query(B2D_ell).copy()
    Dst_tau_nu   = df.query(B2Dst_tau).copy()
    Dst_l_nu     = df.query(B2Dst_ell).copy()
    Dstst_tau_nu = df.query(B2Dstst_tau).copy()
    Dstst_l_nu_narrow = df.query(B2Dstst_ell_narrow).copy()
    Dstst_l_nu_broad  = df.query(B2Dstst_ell_broad).copy()
    D_l_nu_gap_pi = df.query(B2D_ell_gap_pi).copy()
    D_l_nu_gap_eta = df.query(B2D_ell_gap_eta).copy()
    
    classified_signal_indices = pd.concat([D_tau_nu, Dst_tau_nu, D_l_nu,
                                           Dst_l_nu, Dstst_tau_nu,
                                           Dstst_l_nu_narrow,
                                           Dstst_l_nu_broad,
                                           D_l_nu_gap_pi, D_l_nu_gap_eta,]).index
    
    bkg_other_signal = df_signals_all.loc[~df_signals_all.index.isin(classified_signal_indices)].copy()
 
    # Assign signal samples with LaTeX style names:
    samples.update({
        r'$D\tau\nu$':      D_tau_nu,
        r'$D^\ast\tau\nu$': Dst_tau_nu,
        r'$D\ell\nu$':      D_l_nu,
        r'$D^\ast\ell\nu$': Dst_l_nu,
        r'$D^{\ast\ast}\tau\nu$': Dstst_tau_nu,
        r'$D^{\ast\ast}\ell\nu$_narrow': Dstst_l_nu_narrow,
        r'$D^{\ast\ast}\ell\nu$_broad': Dstst_l_nu_broad,
        r'$D\ell\nu$_gap_pi': D_l_nu_gap_pi,
        r'$D\ell\nu$_gap_eta': D_l_nu_gap_eta,
        'bkg_other_signal': bkg_other_signal,
    })
    
    # Finally, assign a 'mode' to each sample based on an external mapping (DecayMode_new)
    # (Make sure that DecayMode_new is defined in your namespace.)
    for name, subset_df in samples.items():
        subset_df['mode'] = DecayMode_new.get(name, -1)
    
    return samples
    
    
def classify_combinatorial(df):
    """
    Classifies combinatorial background into 7 distinct classes based on:
      1. D mother decay type.
      2. Lepton mother PDG classification.

    Args:
        df (pd.DataFrame): Input DataFrame containing necessary columns.

    Returns:
        dict: A dictionary where keys are class names and values are sub-DataFrames.
    """
    # Define relevant columns for D mother decay classification
    study_cols = combinatorial_vars

    # 1. Semileptonic B: at least one lepton appears in the study columns
    mask_sl = df[study_cols].isin(leptons).any(axis=1)

    # 2. Hadronic single charm: no leptons and exactly one value in the D_mesons
    charm_mask1 = df[study_cols].abs().isin(single_charm_pdg)
    mask_cx = (~mask_sl) & (charm_mask1.sum(axis=1) == 1)

    # 3. Hadronic double charm: no leptons and two values in the D_mesons or 1 value in psi
    charm_mask2 = df[study_cols].abs().isin(double_charm_pdg)
    mask_ccx = (~mask_sl) & ( (charm_mask1.sum(axis=1) == 2) | (charm_mask2.sum(axis=1) == 1) )

    # Lepton classification:
    mask_primary_ell = (df['ell_genMotherPDG'].abs().isin(Bpdg)) | (
        (df['ell_genGMPDG'].abs().isin(Bpdg)) & (df['ell_genMotherPDG'].abs() == 15)
    )
    mask_secondary_ell = ~mask_primary_ell

    # Build dictionary of classified samples:
    class_dict = {
        'DSemiB_ellPri': df[mask_sl & mask_primary_ell].copy(),
        'DSemiB_ellSec': df[mask_sl & mask_secondary_ell].copy(),
        'DHad1Charm_ellPri': df[mask_cx & mask_primary_ell].copy(),
        'DHad1Charm_ellSec': df[mask_cx & mask_secondary_ell].copy(),
        'DHad2Charm_ellPri': df[mask_ccx & mask_primary_ell].copy(),
        'DHad2Charm_ellSec': df[mask_ccx & mask_secondary_ell].copy(),
    }

    # Catch unclassified rows
    classified_indices = pd.concat(class_dict.values()).index
    class_dict['others'] = df.loc[~df.index.isin(classified_indices)].copy()

    return class_dict


# Function to check for duplicate entries in a dictionary of Pandas DataFrames
def check_duplicate_entries(data_dict):
    # Create an empty list to store duplicate pairs
    duplicate_pairs = []

    # Iterate through the dictionary values (assuming each value is a DataFrame)
    keys = list(data_dict.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            df1 = data_dict[keys[i]][['__experiment__','__run__','__event__','__production__']]
            df2 = data_dict[keys[j]][['__experiment__','__run__','__event__','__production__']]

            # Check for duplicates between the two DataFrames
            duplicates = pd.merge(df1, df2, indicator=True, how='inner')
            if not duplicates.empty:
                duplicate_pairs.append((keys[i], keys[j]))

    if duplicate_pairs:
        print("Duplicate pairs found:")
        for pair in duplicate_pairs:
            print(pair)
    else:
        print("No duplicate pairs found.")
        
        
############################### Templates and workspace ######################
from uncertainties import ufloat, correlated_values
import uncertainties.unumpy as unp
from uncertainties import UFloat
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
            new_counts_nominal, np.sqrt(new_uncertainties_squared)
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
    # Extract nominal values (counts) and standard deviations (uncertainties)
    counts = unp.nominal_values(counts_with_uncertainties).round().astype(int)
    uncertainties = unp.std_devs(counts_with_uncertainties)

    # Repeat the bin centers based on counts for rebinning
    new_counts, _ = np.histogram(
        np.repeat(old_bin_edges[:-1], counts), bins=new_bin_edges
    )
    
    # Initialize new uncertainties (sum in quadrature)
    new_uncertainties_squared = np.zeros_like(new_counts, dtype=float)

    # Combine uncertainties for the new bins
    for i in range(len(old_bin_edges) - 1):
        bin_value = old_bin_edges[i]
        new_bin_index = np.digitize(bin_value, new_bin_edges) - 1  # Find new bin index
        new_uncertainties_squared[new_bin_index] += uncertainties[i] ** 2  # Sum uncertainties in quadrature

    # Take square root of summed uncertainties square
    new_uncertainties = np.sqrt(new_uncertainties_squared)

    # Combine counts and uncertainties into a single uarray
    new_counts_with_uncertainties = unp.uarray(new_counts, new_uncertainties)
    return new_counts_with_uncertainties


def create_templates(samples:dict, bins:list, scale_lumi=1,
                     variables=['B0_CMS3_weMissM2','p_D_l'],
                     bin_threshold=1, merge_threshold=10,
                     fakeD_from_sideband=False, data=None,
                     sample_to_exclude=['bkg_fakeTracks','bkg_other_TDTl','bkg_other_signal'],
                     sample_weights={r'$D^{\ast\ast}\ell\nu$_broad':1,
                                     r'$D\ell\nu$_gap_pi':1, 
                                     r'$D\ell\nu$_gap_eta':1}):
    """
    Creates 2D templates with uncertainties from input samples and applies rebinning and flattening.

    Parameters:
        samples (dict): Dictionary of data samples, where keys are sample names and values are pandas DataFrames.
        bins (list): List defining the bin edges for the 2D histogram.
        scale_lumi (float, optional): Scaling factor for luminosity. Default is 1.
        variables (list, optional): List of two variable names to use for the 2D histogram. Default is ['B0_CMS3_weMissM2', 'p_D_l'].
        bin_threshold (float, optional): Minimum count threshold for trimming bins. Default is 1.
        merge_threshold (float, optional): Minimum count threshold for merging adjacent bins. Default is 10.
        fakeD_from_sideband (bool, optional): Whether to include fakeD templates derived from D_M sidebands. Default is False.
        data (pandas.DataFrame, optional): Data to be used for fakeD sidebands if `fakeD_from_sideband` is True. Default is None.
        sample_to_exclude (list, optional): List of sample names to exclude from template creation. Default includes specific background samples.
        sample_weights (dict, optional): Dictionary specifying custom weights for specific samples.
            Keys are sample names, and values are weight factors. Default is:
            {
                '$D^{\ast\ast}\ell\nu$_broad': 1,
                '$D\ell\nu$_gap_pi': 1,
                '$D\ell\nu$_gap_eta': 1
            }

    Returns:
        tuple:
            - indices_threshold (np.ndarray): Indices of bins that pass the count threshold.
            - temp_sig (tuple): Tuple containing:
                - template_flat (dict): Flattened templates with keys as sample names and values as uarray of counts and uncertainties.
                - asimov_data (unp.uarray): Summed template representing the Asimov dataset (counts and uncertainties).
            - temp_merged (tuple): Tuple containing:
                - template_flat_merged (dict): Re-binned templates with merged bins based on `merge_threshold`.
                - asimov_data_merged (unp.uarray): Merged Asimov dataset.
            - temp_with_sb (tuple): Tuple containing:
                - template_flat_with_sb (dict): Templates including fakeD derived from sidebands (if applicable).
                - asimov_data_with_sb (unp.uarray): Asimov dataset including fakeD contributions.

    Notes:
        - Templates are represented as `unp.uarray` objects that encapsulate counts and uncertainties.
        - Sample weights are applied when computing weighted histograms.
        - Bins with counts below `bin_threshold` are trimmed.
        - Adjacent bins with counts below `merge_threshold` are merged.
        - If `fakeD_from_sideband` is True, additional templates are created using sidebands of the D_M variable.
    """

    def round_uarray(uarray):
        """Rounds a uarray to 4 decimal places."""
        nominal = np.round(unp.nominal_values(uarray), 4)
        std_dev = np.round(unp.std_devs(uarray), 4)
        return unp.uarray(nominal, std_dev)

    #################### Create template 2d histograms with uncertainties ################
    histograms = {}
    for name, df_sig_sb in samples.items():
        if name in sample_to_exclude:
            continue

        df_sig_sb = df_sig_sb.copy()
        df = df_sig_sb.query('1.855<D_M<1.885')
        
        if name in sample_weights.keys():
            df.loc[:, '__weight__'] = sample_weights[name]
            
        # Compute weighted histogram
        counts, xedges, yedges = np.histogram2d(
            df[variables[0]], df[variables[1]],
            bins=bins, weights=df['__weight__'])

        # Compute sum of weight^2 for uncertainties
        staterr_squared, _, _ = np.histogram2d(
            df[variables[0]], df[variables[1]],
            bins=bins, weights=(df['__weight__']**2))

        # Store as uarray: Transpose to have consistent shape (y,x) if needed
        if name in [r'$D^{\ast\ast}\ell\nu$_narrow',r'$D^{\ast\ast}\ell\nu$_broad']:
            # merge the 2 resonant D** modes
            if r'$D^{\ast\ast}\ell\nu$' in histograms:
                histograms[r'$D^{\ast\ast}\ell\nu$'] += round_uarray(unp.uarray(counts.T, np.sqrt(staterr_squared.T)))
            else:
                histograms[r'$D^{\ast\ast}\ell\nu$'] = round_uarray(unp.uarray(counts.T, np.sqrt(staterr_squared.T)))
        elif name in [r'$D\ell\nu$_gap_pi', r'$D\ell\nu$_gap_eta']:
            # merge the 2 Dellnu gap modes
            if r'$D\ell\nu$_gap' in histograms:
                histograms[r'$D\ell\nu$_gap'] += round_uarray(unp.uarray(counts.T, np.sqrt(staterr_squared.T)))
            else:
                histograms[r'$D\ell\nu$_gap'] = round_uarray(unp.uarray(counts.T, np.sqrt(staterr_squared.T)))
        else:
            # store other modes individually
            histograms[name] = round_uarray(unp.uarray(counts.T, np.sqrt(staterr_squared.T)))

    ################### Trimming and flattening ###############
    # Determine which bins pass the threshold based on sum of all templates
    all_2dHists_sum = np.sum(list(histograms.values()), axis=0)  # uarray sum
    indices_threshold = np.where(unp.nominal_values(all_2dHists_sum) >= bin_threshold)

    # Flatten the templates after cutting
    template_flat = {name: round_uarray(hist[indices_threshold]) for name, hist in histograms.items()}
    # Asimov data is the sum of all templates
    asimov_data = round_uarray(np.sum(list(template_flat.values()), axis=0))  # uarray

    #################### Create additional templates for fakeD from sidebands ###################
    if fakeD_from_sideband and 'bkg_fakeD' not in sample_to_exclude:
        print('Creating the fakeD template from the sidebands')
        if data is None: # MC
            df_all = pd.concat(samples.values(), ignore_index=True)
        else:
            df_all = data
        df_sidebands = df_all.query('D_M<1.83 or 1.91<D_M').copy()

        # Compute the sideband histogram and assume poisson error
        bin_D_M = np.linspace(1.79,1.95,81)
        D_M_s2, _ = np.histogram(df_sidebands['D_M'], bins=bin_D_M)
        D_M_side_count = round_uarray(unp.uarray(D_M_s2, np.sqrt(D_M_s2)))

        # Fit a polynomial to the D_M sidebands
        fitter = fit_Dmass(x_edges=bin_D_M, hist=D_M_side_count, poly_only=True)
        m_ml, c_ml, result_ml = fitter.fit_gauss_poly_ML(deg=1)

        yields_left = fitter.poly_integral(xrange=[1.79,1.82],result=result_ml)
        yields_sig = fitter.poly_integral(xrange=[1.855,1.885],result=result_ml)
        yields_right = fitter.poly_integral(xrange=[1.92,1.95],result=result_ml)
        print(f'sig/left = {round_uarray(yields_sig/yields_left)}, \
        sig/right = {round_uarray(yields_sig/yields_right)}')

        # Construct the fakeD 2d template from sidebands
        region_yield = {'D_M<1.83': yields_left, '1.91<D_M': yields_right}
        hist_sbFakeD = 0
        for region_i, yields_i in region_yield.items():
            df_i = df_sidebands.query(region_i)
            weights_i = df_i['__weight__']
            (side_counts_i, _1, _2) = np.histogram2d(
                df_i[variables[0]], df_i[variables[1]], bins=bins)

            # compute the counts and errors, scale with the fit result
            hist_side_i = unp.uarray(side_counts_i.T, np.sqrt(side_counts_i.T))
            scaled_side_i = hist_side_i * (yields_sig/yields_i/2)
            hist_sbFakeD += scaled_side_i

        # Create new 2d hists with fakeD replaced by sideband
        hists_with_sbFakeD = {k: v for k, v in histograms.items()}

        modified_hist_sbFakeD = hist_sbFakeD - r_D*hists_with_sbFakeD[r'$D\ell\nu$']
        modified_hist_sbFakeD -= r_Dst*hists_with_sbFakeD[r'$D^\ast\ell\nu$']

        # Replace negative nominal values with zero, retain uncertainties
        n_mod_hist_side = unp.nominal_values(modified_hist_sbFakeD)
        s_mod_hist_side = unp.std_devs(modified_hist_sbFakeD)
        n2_mod_hist_side = np.where(n_mod_hist_side < 0, 0, n_mod_hist_side)
        modified_hist_sbFakeD_2 = round_uarray(unp.uarray(n2_mod_hist_side, s_mod_hist_side))

        hists_with_sbFakeD['bkg_fakeD'] = modified_hist_sbFakeD_2

        ################### Trimming and flattening ###############
        # Determine which bins pass the threshold based on sum of all templates
        all_2dHists_with_sbFakeD_sum = np.sum(list(hists_with_sbFakeD.values()), axis=0)  # uarray sum
        indices_threshold_with_sbFakeD = np.where(unp.nominal_values(all_2dHists_with_sbFakeD_sum) >= bin_threshold)

        if np.array_equal(indices_threshold_with_sbFakeD, indices_threshold):
            print(colored('fakeD template from sidebands and signal region have the same global 0-entry bins', "green"))

        else:
            # Combine row and column indices into a single structured array for both sets
            combined_indices_with_sbFakeD = set(zip(indices_threshold_with_sbFakeD[0], indices_threshold_with_sbFakeD[1]))
            combined_indices = set(zip(indices_threshold[0], indices_threshold[1]))

            # Find the intersection of the two sets
            common_indices = combined_indices_with_sbFakeD.intersection(combined_indices)

            # Separate back into row and column indices
            new_indices_threshold = (
                np.array([idx[0] for idx in common_indices]),
                np.array([idx[1] for idx in common_indices])
            )
            print(colored('fakeD template from sidebands and signal region have different global 0-entry bins', "red"))
            print('created a new indices_threshold masking the 0-entry bins in sig OR sidebands')
            print(colored(f'applying the new mask, number of bins was {len(asimov_data)}, now is {len(common_indices)}', "blue"))

        # Flatten the templates after cutting
        template_flat_with_sb = {name: round_uarray(hist[new_indices_threshold]) for name, hist in hists_with_sbFakeD.items()}
        # Asimov data is the sum of all templates
        asimov_data_with_sb = round_uarray(np.sum(list(template_flat_with_sb.values()), axis=0))

        # Do the same for the signal region
        template_flat = {name: round_uarray(hist[new_indices_threshold]) for name, hist in histograms.items()}
        asimov_data = round_uarray(np.sum(list(template_flat.values()), axis=0))  # uarray
        indices_threshold = new_indices_threshold

    else:
        template_flat_with_sb = {}
        asimov_data_with_sb = []

    ################## Create a new set of templates with merged bins ###########
    # Rebin asimov_data according to merge_threshold
    new_counts, new_dummy_bin_edges, old_dummy_bin_edges = rebin_histogram(asimov_data, merge_threshold)
    print(f'creating a new template with merged bins, original template length = {len(asimov_data)}, new template (merge small bins) length = {len(new_counts)}')

    template_flat_merged = {}
    for name, t in template_flat.items():
        # Rebin using new edges
        # Note: old_dummy_bin_edges and new_dummy_bin_edges come from rebin_histogram
        # For consistency, we must use the same old bin edges as for asimov_data:
        rebinned = rebin_histogram_with_new_edges(t, old_dummy_bin_edges, new_dummy_bin_edges)
        template_flat_merged[name] = round_uarray(rebinned)

    asimov_data_merged = round_uarray(np.sum(list(template_flat_merged.values()), axis=0))

    #################### Prepare return tuples: (template dict, asimov data)
    temp_sig = (template_flat, asimov_data)
    temp_with_sb = (template_flat_with_sb, asimov_data_with_sb)
    temp_merged = (template_flat_merged, asimov_data_merged)

    return indices_threshold, temp_sig, temp_with_sb, temp_merged



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
                    data_label='Data', model_label='Model'):
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
    """

    def compute_residuals(data, model):
        # Compute residuals and errors (assuming Poisson for data)
        residuals = data - model
        residuals[unp.nominal_values(data) == 0] = 0  # Mask bins with 0 data
        res_val = unp.nominal_values(residuals)
        res_err = unp.std_devs(residuals)
        return res_val, res_err

    def plot_residuals(ax, bin_centers, res_val, res_err):
        # Mask bins with zero errors
        mask = res_err != 0
        chi2 = np.sum((res_val[mask] / res_err[mask]) ** 2)
        ndf = len(res_val[mask])
        label = f'reChi2 = {chi2:.3f} / {ndf} = {chi2/ndf:.3f}' if ndf else 'reChi2 not calculated'
        ax.errorbar(bin_centers, res_val, yerr=res_err, fmt='ok', label=label)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_ylabel('Residuals')
        ax.legend()

    # Project histograms onto x-axis
    projData_x = np.sum(data, axis=0)
    projModel_x = np.sum(model, axis=0)

    # Project histograms onto y-axis
    projData_y = np.sum(data, axis=1)
    projModel_y = np.sum(model, axis=1)

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
    axes[0, 0].errorbar(bin_centers_x, unp.nominal_values(projData_x), 
                        yerr=unp.std_devs(projData_x), fmt='ok', label=data_label)
    axes[0, 0].set_ylabel('# of Events')
    axes[0, 0].set_title(f'Projection onto {xlabel}')
    axes[0, 0].grid()
    axes[0, 0].legend()

    # X-axis residuals (bottom-left)
    plot_residuals(axes[1, 0], bin_centers_x, res_x, res_err_x)
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
    plot_residuals(axes[1, 1], bin_centers_y, res_y, res_err_y)
    axes[1, 1].set_xlabel(ylabel)

    plt.tight_layout()
    plt.show()


def create_workspace(temp_asimov_channels: list, 
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

    # Extract sample names from the first set of templates
    sample_names = list(temp_asimov_channels[0][0].keys())

    # Loop over each channel (index, tuple of template_flat and asimov_data)
    for ch_index, (template_flat, asimov_data) in enumerate(temp_asimov_channels):
        
        # Store observed data for the channel
        observations.append({
            'name': f'channel_{ch_index}',
            'data': unp.nominal_values(asimov_data).tolist()  # Extract nominal values from uncertainties
        })
        
        # Initialize channel structure
        channels.append({
            'name': f'channel_{ch_index}',
            'samples': []
        })

        # Loop over each sample in the channel
        for sample_index, sample_name in enumerate(sample_names):
            # Add the nominal template data for the sample
            channels[ch_index]['samples'].append({
                'name': sample_name,
                'data': unp.nominal_values(template_flat[sample_name]).tolist(),
                'modifiers': [
                    {
                        'name': sample_name+'_norm',
                        'type': 'normfactor',
                        'data': None  # Normalization factor modifier
                    }
                ]
            })

            # Add uncertainty modifiers for statistical errors
            if sample_name == 'bkg_fakeD' and fakeD_uncer:
                # Add statistical uncertainty for 'bkg_fakeD' using shapesys
                channels[ch_index]['samples'][sample_index]['modifiers'].append({
                    'name': f'fakeD_stat_uncer_ch{ch_index}',
                    'type': 'shapesys',
                    'data': unp.std_devs(template_flat[sample_name]).tolist()
                })
            elif sample_name != 'bkg_fakeD' and mc_uncer:
                # Add statistical uncertainty for all other MC backgrounds using staterror
                channels[ch_index]['samples'][sample_index]['modifiers'].append({
                    'name': f'mc_stat_uncer_ch{ch_index}',
                    'type': 'staterror',
                    'data': unp.std_devs(template_flat[sample_name]).tolist()
                })

            # Define parameter bounds based on whether it's a background or signal sample
            if sample_name.startswith('bkg'):
                par_config = {"name": sample_name+'_norm', "bounds": [[0, 2]], "inits": [1.0], "fixed":True}
            else:
                par_config = {"name": sample_name+'_norm', "bounds": [[-5, 5]], "inits": [1.0]}

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



# def calculate_FOM3d(sig_data, bkg_data, variables, test_points):
#     sig = pd.concat(sig_data)
#     bkg = pd.concat(bkg_data)
#     sig_tot = len(sig)
#     bkg_tot = len(bkg)
#     BDT_FOM = []
#     BDT_FOM_err = []
#     BDT_sigEff = []
#     BDT_sigEff_err = []
#     BDT_bkgEff = []
#     BDT_bkgEff_err = []
#     for i in test_points[0]:
#         for j in test_points[1]:
#             for k in test_points[2]:
#                 nsig = len(sig.query(f"{variables[0]}>{i} and {variables[1]}>{j} and {variables[2]}>{k}"))
#                 nbkg = len(bkg.query(f"{variables[0]}>{i} and {variables[1]}>{j} and {variables[2]}>{k}"))
#                 tot = nsig+nbkg
#                 tot_err = np.sqrt(tot)
#                 FOM = nsig / tot_err # s / √(s+b)
#                 FOM_err = np.sqrt( (tot_err - FOM/2)**2 /tot**2 * nsig + nbkg**3/(4*tot**3) + 9*nbkg**2*np.sqrt(nsig*nbkg)/(4*tot**5) )

#                 BDT_FOM.append(round(FOM,2))
#                 BDT_FOM_err.append(round(FOM_err,2))

#                 sigEff = nsig / sig_tot
#                 sigEff_err = sigEff * np.sqrt(1/nsig + 1/sig_tot)
#                 bkgEff = nbkg / bkg_tot
#                 bkgEff_err = bkgEff * np.sqrt(1/nbkg + 1/bkg_tot)
#                 BDT_sigEff.append(round(sigEff,2))
#                 BDT_sigEff_err.append(round(sigEff_err,2))
#                 BDT_bkgEff.append(round(bkgEff,2))
#                 BDT_bkgEff_err.append(round(bkgEff_err,2))
#     print(f'{BDT_FOM=}')
#     print(f'{BDT_sigEff=}')
#     print(f'{BDT_bkgEff=}')


# # +
############################### PID corrections #########################
import sys
import os
from datetime import date
import warnings
from sysvar import add_weights_to_dataframe

class PID_corrections:
    def __init__(self):
        self.e_efficiency = '~/B2SW/2024_OleMiss/systematics_framework/correction-tables/MC15/run_independent/PID/coarse_theta_binning/efficiency/e_efficiency_table.csv'
        self.pi_e_fake = '~/B2SW/2024_OleMiss/systematics_framework/correction-tables/MC15/run_independent/PID/coarse_theta_binning/fakeRate/pi_e_fakeRate_table.csv'
        self.mu_efficiency = ''
        self.pi_mu_fake = ''
        self.K_efficiency = 'tables/K_efficiency_kaonIDNN_0.9_2024-10-31.csv'
        self.pi_K_fake = 'tables/pi_K_fake_kaonIDNN_0.9_2024-10-31.csv'
        self.sys_path = '/group/belle2/dataprod/Systematics/systematic_corrections_framework/scripts/'
        
    def plot_table(self, table, table_name):
        fig, axs = plt.subplots(1,2, figsize=(8, 4), dpi=120)
        fig.suptitle(table_name)
        axs[0].plot(table[['p_min', 'p_max']].values.T);
        axs[1].plot(table[['theta_min', 'theta_max']].values.T);
        axs[0].set_xticks([0,1], ['min', 'max'])
        axs[1].set_xticks([0,1], ['min', 'max'])
        axs[0].set_xlabel('p_bin')
        axs[1].set_xlabel('theta_bin')
        axs[0].set_ylabel('p')
        axs[1].set_ylabel('theta')
        axs[0].grid()
        axs[1].grid();
        
    def get_lepton_tables(self, lepton='e', var="pidChargedBDTScore_e",
                          thres=0.9, exclude_bins='p_min>-1'):
        final_query = f'is_best_available == True and variable == "{var}" and \
        threshold =={thres}'
        
        if lepton=='e':
            ell_efficiency_table = pd.read_csv(self.e_efficiency).query(final_query).query(exclude_bins)
            pi_ell_fake_table = pd.read_csv(self.pi_e_fake).query(final_query).query(exclude_bins)
        
        elif lepton=='mu':
            ell_efficiency_table = pd.read_csv(self.mu_efficiency).query(final_query).query(exclude_bins)
            pi_ell_fake_table = pd.read_csv(self.pi_mu_fake).query(final_query).query(exclude_bins)
        
        self.plot_table(table=ell_efficiency_table, table_name=f'{lepton} efficiency')
        self.plot_table(table=pi_ell_fake_table, table_name=f'pi {lepton} fake rate')
        
        return ell_efficiency_table, pi_ell_fake_table
        
    def get_hadron_tables(self, new_table=False, hadron='K', var='kaonIDNN', thres=0.9):
        
        if new_table:
            sys.path.insert(1, self.sys_path)
            import weight_table as wm
            
            # efficiency table
            ratio_cfg = {
                "cut": f"{var} > {thres}",
                "particle_type": hadron,
                "data_collection": "proc13+prompt",
                "mc_collection": "MC15ri",
                "track_variables": ["p", "cosTheta", "charge"],
                "apply_std_constraints": False,
                "precut": "abs(dz)<2 and dr<0.5 and thetaInCDCAcceptance and nPXDHits>0 and nCDCHits>0",
                "binning": [np.linspace(0.2, 4, 11),
                           [-0.866, -0.682, -0.4226, -0.1045, 0.225, 0.5, 0.766, 0.8829, 0.9563],
                           [-2, 0, 2]]
            }
            efficiency_obj = wm.produce_data_mc_ratio(**ratio_cfg)
            efficiency_obj.plot()
            efficiency_table = efficiency_obj.create_weights()
            
            os.makedirs('tables/', exist_ok=True)
            efficiency_table.to_csv(f'tables/{hadron}_efficiency_{var}_{thres}_{date.today()}.csv', index=None)
            
            # fake rate table
            ratio_cfg = {
                "cut": f"{var} > {thres}",
                "particle_type": "pi",
                "data_collection": "proc13+prompt",
                "mc_collection": "MC15ri",
                "track_variables": ["p", "cosTheta", "charge"],
                "apply_std_constraints": False,
                "precut": "abs(dz)<2 and dr<0.5 and thetaInCDCAcceptance and nPXDHits>0 and nCDCHits>0",
                "binning": [np.linspace(0.2, 4, 11),
                           [-0.866, -0.682, -0.4226, -0.1045, 0.225, 0.5, 0.766, 0.8829, 0.9563],
                           [-2, 0, 2]]
            }
            pi_fake_obj = wm.produce_data_mc_ratio(**ratio_cfg)
            pi_fake_obj.plot()
            pi_fake_table = pi_fake_obj.create_weights()
            
            pi_fake_table.to_csv(f'tables/pi_{hadron}_fake_{var}_{thres}_{date.today()}.csv', index=None)
        
        else:
            # load the existing CSV table:
            efficiency_table = pd.read_csv(self.K_efficiency, index_col=None)
            pi_fake_table = pd.read_csv(self.pi_K_fake, index_col=None)
            
        return efficiency_table, pi_fake_table
    
    def apply_corrections(self, eff_table, fake_table, df, plots=True,
                          p='e', var='pidChargedBDTScore_e', thres=0.9):
        if p=='e':
            p_tables = {(11, 11): eff_table,
                        (11, 211): fake_table}
            p_thresholds = {11: (var, thres)}
            p_prefix = 'ell'
        
        elif p=='K':
            p_tables = {(321, 321): eff_table,
                        (321, 211): fake_table}
            p_thresholds = {321: (var, thres)}
            p_prefix = 'D_K'
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            add_weights_to_dataframe(p_prefix,
                                     df,
                                     systematic='custom_PID',
                                     custom_tables=p_tables,
                                     custom_thresholds=p_thresholds,
                                     show_plots=plots,
                                     sys_seed=0)



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
        std = np.sqrt(variance)
        
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
        c.mask = (x < 1.822) | (1.92 < x)
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
        c.mask = (x_re < 1.822) | (1.92 < x_re)
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
    def __init__(self, toy_temp, fit_temp, fit_inits = [0.4]*12,
                 toy_pars=[1]*12, par_fix=[False]*12):
        
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
            model_toy.config.par_map[par_name]['paramset'].suggested_init=[toy_pars[i]]
            model_toy.config.par_map[par_name]['paramset'].suggested_fixed=fix_mask[i]
            model_fit.config.par_map[par_name]['paramset'].suggested_init=[fit_inits[i]]
            model_fit.config.par_map[par_name]['paramset'].suggested_fixed=fix_mask[i]

        # setup init attributes
        self.model_toy = model_toy
        self.model_fit = model_fit
        # Set the pars for generating toys
        self.toy_pars = cabinetry.model_utils.asimov_parameters(model_toy)
        # Create a list of parameter names for samples that are not fixed
        self.minos_pars = [par for par in norm_parameter_names if par not in par_fix]
        
    def toy_generator(self, n_toys):
        pdf_toy = self.model_toy.make_pdf(pyhf.tensorlib.astensor(self.toy_pars))
        toys = pdf_toy.sample((n_toys,))

        return toys
        
    def fit_scipy_minuit(self, data):
        try: # fit with scipy for an initial guess
            pyhf.set_backend('jax', 'scipy')
            init_pars = pyhf.infer.mle.fit(pdf=self.model_fit, data=data).tolist()

        except Exception as e: # use the suggested init
            init_pars = self.model_fit.config.suggested_init()

        # fit with minuit
        pyhf.set_backend('jax', 'minuit')
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
        
    def generate_fit_toys(self,toy_pars,fit_inits,n_toys):
        # initialize util tools for pyhf
        pyhf_tools = pyhf_utils(toy_temp=self.toy_workspace,
                                     par_fix=self.pars_toFix,
                                     fit_temp=self.fit_workspace,
                                     toy_pars=toy_pars,
                                     fit_inits = fit_inits
                                   )
        # generate toys
        toys = pyhf_tools.toy_generator(n_toys=n_toys)
        
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
            percent_error = np.where(diff > 0, minos_up / fitted, minos_down / fitted) # will show in the plot
        else:
            # hesse error
            hesse_error = np.array(merged_results['hesse_uncertainty'])
            pulls = diff / hesse_error
            percent_error = hesse_error / fitted # will show in the plot
        
        if not normalize:
            pulls = diff
            
        # save
        merged_dict['toy_results']['pulls']=pulls.tolist()
        merged_dict['toy_results']['percent_error']=percent_error.tolist()
        
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
                          xlabel: str = '$(\mu-\mu_{in}) /\sigma_{\mu}$',
                          figsize=(6, 6 / 1.618), show: bool = False):
        # set up the figure
        fig = plt.figure(figsize=figsize)
        bins = np.linspace(-5 * sigma.n, +5 * sigma.n, 101)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        bin_width = bins[1] - bins[0]
        
        # set up the fitted gaussian
        def gaussian(x, mu, sig): # user defined gauss for uncertainties
            return 1. / (((2. * np.pi)**0.5) * sig) * np.e**(-(((x - mu) / sig)**2) / 2)
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
        plt.errorbar(x=bin_centers,y=hist,yerr=np.sqrt(hist),fmt='.',color='black',
                     markeredgecolor='white',markeredgewidth=0.5)

        # set up reference line and text
        for v in vertical_lines:
            plt.axvline(v, color='gray', ls='--', zorder=-100)

        plt.text(0.95,0.95, fr'$\mu_{{G}}=${round(mu.n,3)}$\pm${round(mu.s,3)}',
                 va='top',ha='right',usetex=False, transform=plt.gca().transAxes)
        plt.text(0.95, 0.88, fr'$\sigma_{{G}}=${round(sigma.n,3)}$\pm${round(sigma.s,3)}',
                 va='top', ha='right', usetex=False, transform=plt.gca().transAxes)

        if title_info is not None:
            plt.title(title_info, loc='right')
        if extra_info is not None:
            plt.text(0.05, 0.95, extra_info, va='top', ha='left', usetex=False, 
                     transform=plt.gca().transAxes, fontsize=12)
        
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
        plt.errorbar(np.array(x) + x_offset, np.array(y), yerr=yerr, label=None, fmt='.', color='k')
        
        # set up extra reference line and text
        plt.plot(np.array([bonds[0], bonds[1]]) + x_offset, [bonds[0], bonds[1]], color='gray', label='Diagonal', lw=0.5, zorder=-100, ls='--')
        plusminus = '+' if intercept >= 0 else '-'
        eq = f"""({round(slope.n,3)}$\pm${round(slope.s,3)})$\mu_{{in}}$${plusminus}$({abs(round(intercept.n,3))}$\pm${round(intercept.s,3)})"""
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
##################################### Plotting #################################
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import gridspec

######## define my colormap ########
# Original tab20 colors
original_colors = plt.cm.tab20.colors
# New order for the colors
new_order_indices = [0,1,2,3,16,10,4,5,12,13,19,8,18,7,6]
# Create a new ordered list of colors
reordered_colors = [original_colors[i] for i in new_order_indices]
# Add the rest of the colors that are not explicitly ordered
remaining_indices = [i for i in range(len(original_colors)) if i not in new_order_indices]
reordered_colors.extend([original_colors[i] for i in remaining_indices])
# Create a new colormap
my_cmap = mcolors.ListedColormap(reordered_colors, name='reordered_tab20')

class mpl:
    def __init__(self, mc_samples, data=None):
        self.samples = mc_samples
        self.data = data
        self.colors = my_cmap.colors
        # sort the components to plot in order of fitted templates_project size
        self.sorted_order = ['bkg_fakeD',    'bkg_continuum',    'bkg_combinatorial',
                             'bkg_TDFl',     'bkg_fakeTracks',
                             'bkg_singleBbkg',                   
                             r'$D\ell\nu$_gap_pi',               r'$D\ell\nu$_gap_eta',
                             r'$D^{\ast\ast}\ell\nu$_narrow',    r'$D^{\ast\ast}\ell\nu$_broad',      
                             r'$D^{\ast\ast}\tau\nu$',
                             r'$D^\ast\ell\nu$',                 r'$D\ell\nu$',
                             r'$D^\ast\tau\nu$',                 r'$D\tau\nu$']
        self.bkg = self.sorted_order[:6]
        self.norm = [r'$D\ell\nu$_gap_pi', r'$D\ell\nu$_gap_eta',
                     r'$D^{\ast\ast}\ell\nu$_narrow', r'$D^{\ast\ast}\ell\nu$_broad',
                     r'$D^\ast\ell\nu$',r'$D\ell\nu$']
        self.sig = [r'$D^{\ast\ast}\tau\nu$',r'$D^\ast\tau\nu$',r'$D\tau\nu$']
       
    
    def statistics(self, df=None, hist=None, count_only=False):
        if df is not None:
            counts = df.count()
            mean = df.mean()
            std = df.std()
        
        if hist is not None:
            bin_counts, bin_edges = hist
            counts = np.sum(bin_counts)
            
            # Step 1: Calculate bin centers
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            
            # Step 2: Calculate the weighted mean
            mean = np.average(bin_centers, weights=bin_counts)
            
            # Step 3: Calculate the weighted variance
            variance = np.average((bin_centers - mean)**2, weights=bin_counts)
            
            # Step 4: Use the uncertainties package's sqrt if variance has uncertainties
            std = unp.sqrt(variance)
        if count_only:
            return f'{counts=:d}'
        else:
            return f'''{counts=:d} \n{mean=:.3f} \n{std=:.3f}'''
    
    def plot_pie(self, cut='1.855<D_M<1.885'):
        # Plotting the pie chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        sizes = [len(self.samples[comp].query(cut)) for comp in self.sorted_order]
        ax1.pie(sizes, labels=self.sorted_order, autopct='%1.1f%%', startangle=140, colors=self.colors)
        ax1.set_title(f'All components in the region {cut=}')
        ax2.pie(sizes[:6], labels=self.sorted_order[:6], autopct='%1.1f%%', startangle=140, colors=self.colors)
        ax2.set_title(f'BKG components in the region {cut=}')
        plt.tight_layout()
        plt.show()
        
    
    def plot_data_1d(self, bins, ax, hist=None, sub_df=None, variable=None, cut=None, 
                 sig_mask=False, scale=1, name='Data', density=False):
        # Provide either variable or hist
        if variable:
            # Apply signal mask if requested
            if sig_mask:
                s_mask = 'D_M<1.855 or D_M>1.885'
                data = sub_df.query(s_mask) if sub_df is not None else self.data.query(s_mask)
            else:
                data = sub_df if sub_df is not None else self.data

            # Apply scaling if requested
            if scale:
                data.loc[:, '__weight__'] = scale

            var_col = data.query(cut)[variable] if cut else data[variable]

            # Compute histogram with weights
            counts, _ = np.histogram(var_col, bins=bins,
                                     weights=data.query(cut)['__weight__'] if cut else data['__weight__'])
            staterr_squared, _ = np.histogram(var_col, bins=bins,
                                              weights=(data.query(cut)['__weight__'] if cut else data['__weight__'])**2)
            staterror = np.sqrt(staterr_squared)

            # Normalize to density if requested
            if density:
                bin_widths = np.diff(bins)
                integral = np.sum(counts * bin_widths)
                if integral > 0:
                    factor = 1.0 / integral
                    counts *= factor
                    staterror *= factor

            label = f'{name} \n{self.statistics(df=var_col)}\n cut_eff={(len(var_col)/len(data)):.3f}'
            data_counts = unp.uarray(counts, staterror)

        else:
            # If hist is provided directly (data_counts as unp.uarray), we handle it similarly
            data_counts = hist

            if density and hist is not None:
                # Normalize the provided histogram to density if needed
                counts = unp.nominal_values(data_counts)
                staterror = unp.std_devs(data_counts)
                bin_widths = np.diff(bins)
                integral = np.sum(counts * bin_widths)
                if integral > 0:
                    factor = 1.0 / integral
                    counts *= factor
                    staterror *= factor
                    data_counts = unp.uarray(counts, staterror)

            label = f'{name} \n{self.statistics(hist=[data_counts,bins])}'

        bin_centers = (bins[:-1] + bins[1:]) / 2
        data_val = unp.nominal_values(data_counts)
        data_err = unp.std_devs(data_counts)

        if ax is not None:
            # Plot using errorbar to show data with uncertainties
            ax.errorbar(x=bin_centers, y=data_val, yerr=data_err, fmt='ko', label=label)

        return data_counts


    def plot_mc_1d(self, bins, ax, sub_df=None, sub_name=None, variable=None, cut=None,
               scale=1, correction=None, mask=[], legend_count=False, density=False):

        def normalize_to_density(counts, bins):
            # If density is True, normalize the counts so that the integral is 1
            if density:
                bin_widths = np.diff(bins)
                integral = np.sum(counts * bin_widths)
                if integral > 0:
                    counts = counts / integral
            return counts

        if correction:
            mc_combined = pd.concat(
                [df for name, df in self.samples.items() if name not in mask],
                ignore_index=True)
            if scale:
                mc_combined.loc[:, '__weight__'] = scale
            var_col = mc_combined.query(cut)[variable] if cut else mc_combined[variable]
            (stacked_counts, _) = np.histogram(var_col, bins=bins,
                                   weights=mc_combined.query(cut)['__weight__'] if cut else mc_combined['__weight__'])
            stacked_counts = normalize_to_density(stacked_counts, bins)
            
            if ax is not None:
                ax.hist(bins[:-1], bins, weights=stacked_counts, histtype='step', color='black',
                    label=(f'Unweighted MC \n{self.statistics(df=var_col,count_only=legend_count)} '
                           f'\n cut_eff={(len(var_col)/len(mc_combined)):.3f}'))

        if sub_df is not None:
            sample = sub_df
            if scale:
                sample.loc[:, '__weight__'] = scale
            var_col = sample.query(cut)[variable] if cut else sample[variable]
            (counts, _) = np.histogram(var_col, bins=bins,
                                       weights=sample.query(cut)['__weight__'] if cut else sample['__weight__'])
            (staterr_squared, _) = np.histogram(var_col, bins=bins,
                                 weights=sample.query(cut)['__weight__']**2 if cut else sample['__weight__']**2)
            staterror = np.sqrt(staterr_squared)

            counts = normalize_to_density(counts, bins)  # Normalize if density=True

            if ax is not None:
                ax.hist(bins[:-1], bins, weights=counts,
                    label=(f'{sub_name} \n{self.statistics(df=var_col,count_only=legend_count)} '
                           f'\n cut_eff={(len(var_col)/len(sample)):.3f}'))

            sample_counts = unp.uarray(counts, staterror)
            bottom = sample_counts

        else:
            bottom = unp.uarray(np.zeros(len(bins)-1), np.zeros(len(bins)-1))
            for i, name in enumerate(self.sorted_order):
                sample = self.samples[name]
                sample_size = len(sample.query(cut)) if cut else len(sample)
                if sample_size == 0 or name in mask:
                    continue
                if scale:
                    sample.loc[:, '__weight__'] = scale
                var_col = sample.query(cut)[variable] if cut else sample[variable]
                (counts, _) = np.histogram(var_col, bins=bins,
                                           weights=sample.query(cut)['__weight__'] if cut else sample['__weight__'])
                (staterr_squared, _) = np.histogram(var_col, bins=bins,
                                    weights=sample.query(cut)['__weight__']**2 if cut else sample['__weight__']**2)
                staterror = np.sqrt(staterr_squared)

                # Apply correction if needed
                if correction:
                    (counts, _) = np.histogram(var_col, bins=bins,
                                               weights=scale * sample.query(cut)['PIDWeight'] if cut else scale * sample['PIDWeight'])
                    (staterr_squared, _) = np.histogram(var_col, bins=bins,
                                                        weights=(scale * sample.query(cut)['PIDWeight'])**2 if cut else (scale * sample['PIDWeight'])**2)
                    staterror = np.sqrt(staterr_squared)

                # Normalize if density=True
                counts = normalize_to_density(counts, bins)
                b = unp.nominal_values(bottom)
                
                if ax is not None:
                    ax.hist(bins[:-1], bins, weights=counts, bottom=b, color=self.colors[i],
                        label=(f'{name} \n{self.statistics(df=var_col,count_only=legend_count)} '
                               f'\n cut_eff={(sample_size/len(sample)):.3f}'))

                sample_counts = unp.uarray(counts, staterror)
                bottom += sample_counts

        return bottom
    
    
    def plot_mc_1d_overlaid(self,variable,bins,cut=None,mask=[],show_only=None,density=False):
        if show_only is not None:
            # this will overwrite the mask argument
            if show_only == 'sig':
                mask = self.bkg + [r'$D^\ast\ell\nu$',r'$D\ell\nu$']
            elif show_only == 'norm':
                mask = self.bkg + self.sig
            elif show_only == 'bkg':
                mask = self.norm + self.sig
            else:
                print('Warning: show_only argument accepts sig, norm or bkg')
            
        fig,axs =plt.subplots(sharex=True, sharey=False,figsize=(8, 6))
        for i, name in enumerate(self.sorted_order):
            sample = self.samples[name]
            sample_size = len(sample.query(cut)) if cut else len(sample)
            if sample_size == 0 or name in mask:
                continue
            var_col= sample.query(cut)[variable] if cut else sample[variable]
            (counts, _) = np.histogram(var_col, bins=bins)

            axs.hist(bins[:-1], bins, weights=counts, density=density,histtype='step',lw=2,color=self.colors[i],
                    label=f'''{name} \n{self.statistics(var_col)} \n cut_eff={(sample_size/len(sample)):.3f}''')

        axs.set_title(f'Overlaid components ({cut=})', fontsize=14)
        axs.set_xlabel(f'{variable}', fontsize=14)
        axs.set_ylabel(f'# of events per bin {(bins[1]-bins[0]):.3f} GeV', fontsize=14)
        axs.grid()
        plt.legend(bbox_to_anchor=(1,1),ncol=3, fancybox=True, shadow=True,labelspacing=1.5)

    
    def plot_single_2d(self, df, variables, bins,fig, ax,name,hist=None,cut=None):
        # Compute 2d hist
        if hist is None:
            (counts, xedges, yedges) = np.histogram2d(
                            df.query(cut)[variables[0]] if cut else df[variables[0]], 
                            df.query(cut)[variables[1]] if cut else df[variables[1]],
                            bins=bins,
                    weights=df.query(cut)['__weight__'] if cut else df['__weight__'])

            (staterr_squared, _, _) = np.histogram2d(
                            df.query(cut)[variables[0]] if cut else df[variables[0]], 
                            df.query(cut)[variables[1]] if cut else df[variables[1]],
                            bins=bins,
                    weights=df.query(cut)['__weight__']**2 if cut else df['__weight__']**2)
            staterror = np.sqrt(staterr_squared)

            counts_err = unp.uarray(counts.round(0), staterror.round(0))
        else:
            xedges, yedges = bins
            counts = hist
            counts_err = hist.round(0)

        if fig is not None and ax is not None:
            # 2D Histogram
            im = ax.imshow(counts.T.round(0), origin='lower', aspect='auto', 
                             cmap='rainbow', norm=mcolors.LogNorm(),
                             extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
            fig.colorbar(im, ax=ax)
            ax.set_xlabel('$M_{miss}^2$')
            ax.set_ylabel('$|p_D| + |p_{\ell}|$')
            ax.set_title(name)
            ax.grid()
        
        return counts_err

    def plot_residuals(self, bins, data, model, ax, fig=None):
        if len(bins)>2:
            # Compute residuals (Data - Model) and their errors
            # at bins where data is not 0
            bin_centers = (bins[:-1] + bins[1:]) /2

            residuals = data - model
            residuals[unp.nominal_values(data) == 0] = 0  # Mask bins with 0 data

            res_val = unp.nominal_values(residuals)
            res_err = unp.std_devs(residuals)

            # Create a mask to exclude points where residual_errors are zero
            mask = res_err != 0
            # Compute chi-squared excluding those points
            chi2 = np.sum((res_val[mask] / res_err[mask]) ** 2)
            ndf = len(res_val[mask])
            label = f'reChi2 = {chi2:.3f} / {ndf} = {chi2/ndf:.3f}' if ndf else 'reChi2 not calculated'

            # Plot the residuals in ax
            ax.errorbar(bin_centers, res_val, yerr=res_err, fmt='ok',label=label)
            # Add a horizontal line at y=0 for reference
            ax.axhline(0, color='gray', linestyle='--')
            # Label the residual plot
            ax.set_ylabel('Residuals')
        elif len(bins)==2:
            residuals = data - model
            res_val = unp.nominal_values(residuals)
            res_err = unp.std_devs(residuals)
            
            # Create a mask to exclude points where residual_errors are zero
            mask = res_err != 0
            # Compute chi-squared excluding those points
            chi2 = np.sum((res_val[mask] / res_err[mask]) ** 2)
            ndf = len(res_val[mask])
            
            if ndf==0:
                label = 'reChi2 not calculated'
            else:
                label = f'reChi2 = {chi2:.3f} / {ndf} = {chi2/ndf:.3f}'
                
            # Plot the residuals in ax
            self.plot_single_2d(bins=bins, df=None, variables=None,
                                hist=res_val, fig=fig, ax=ax,name=label)
        
    def plot_ratios(self, bins, data, model, ax):
        # Compute ratios (Data / Model) and their errors
        bin_centers = (bins[:-1] + bins[1:]) /2
        mask_model = model != 0
        ratios = unp.uarray(np.ones_like(model), np.zeros_like(model))
        ratios[mask_model] = data[mask_model] / model[mask_model]
        rat_val = unp.nominal_values(ratios)
        rat_err = unp.std_devs(ratios)

        # Plot the ratios in ax
        ax.errorbar(bin_centers, rat_val, yerr=rat_err, fmt='ok')
        # Add a horizontal line at y=0 for reference
        ax.axhline(1, color='gray', linestyle='--')
        # Label the residual plot
        ax.set_ylabel('Ratios')
    
    def plot_data_mc_stacked(self,variable,bins,cut=None,scale=[1,1],
                             data_sig_mask=False, density=False,
                             correction=False,mask=[],figsize=(8,5),
                             ratio=False, legend_nc=2,legend_fs=12):
        # Create a figure with two subplots: one for the histogram, one for the residual plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [5, 1]})
            
        # MC
        mc_counts = self.plot_mc_1d(bins=bins, variable=variable, ax=ax1, cut=cut,
                                    density=density, 
                                    scale=scale[1],correction=correction,mask=mask)
        # Data
        if self.data is None:
            data_counts = unp.uarray(np.zeros_like(mc_counts), np.zeros_like(mc_counts))
        else:
            data_counts = self.plot_data_1d(bins=bins, variable=variable, 
                                            sig_mask=data_sig_mask,
                                            ax=ax1, cut=cut, scale=scale[0])
        
        if ratio:
            self.plot_ratios(bins=bins, data=data_counts, model=mc_counts, ax=ax2)
        else:
            # Residuals (Data - Model)
            self.plot_residuals(bins=bins, data=data_counts, model=mc_counts, ax=ax2)
            ax2.legend(bbox_to_anchor=(1,1),fancybox=True, shadow=True, fontsize=legend_fs)
        
        ax1.set_title(f'Overlaid Data vs MC ({cut=})')
        ax1.set_ylabel(f'# of events per bin {(bins[1]-bins[0]):.3f} GeV')
        ax1.grid()
        ax1.legend(bbox_to_anchor=(1,1),ncol=legend_nc, fancybox=True, shadow=True,labelspacing=1.5, fontsize=legend_fs)
        ax2.set_xlabel(f'{variable}')
        
        
        # Adjust the layout to avoid overlapping of the subplots
        plt.tight_layout()
        # Show the plot
        plt.show()

        return data_counts, mc_counts
        
        
    def plot_mc_sig_control(self,variable,bins,cut=None,correction=False,scale={},mask=[],
                            bkg_name='bkg_fakeD',merge_sidebands=False,samples_sig=None,
                            norm_tail_subt=False,figsize=(8,5),legend_nc=2,legend_fs=12):
        if type(variable)==str:
            # Create a figure with two subplots: one for the histogram, one for the residual plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [5, 1]})
        elif type(variable)==list:
            # Create a figure with two subplots: one for sig, one for the control
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        if bkg_name=='bkg_fakeD':
            # norm in sig region, for tail removal
            Dellnu = self.samples[r'$D\ell\nu$'].query('1.84<D_M<1.9').copy()
            Dstellnu = self.samples[r'$D^\ast\ell\nu$'].query('1.84<D_M<1.9').copy()
                    
            # fakeD in the signal region
            fakeD = self.samples[bkg_name]
            sig = fakeD.query('1.84<D_M<1.9').copy()
            
            # fakeD in sidebands, Concatenate all DataFrames into one
            df_concatenated = pd.concat(self.samples.values(), ignore_index=True)
            left = df_concatenated.query('D_M<1.83').copy()
            right = df_concatenated.query('D_M>1.91').copy()
                
            regions = {'left sideband': left,
                       'signal region': sig,
                       'right sideband': right}
            for region, df in regions.items():
                df.loc[:, '__weight__'] = scale[region]
            
            if merge_sidebands:
                sides = pd.concat([left, right])
                regions = {'sidebands': sides,
                       'signal region': sig,}
            
            sb_total = 0 # total counts in sidebands used in residual calculation
            sig_total = 0
            if type(variable)==str:
                D_counts = self.plot_mc_1d(bins=bins, sub_df=Dellnu, sub_name=region, variable=variable, 
                                        ax=None, cut=cut, scale=None,correction=correction,mask=mask)
                Dst_counts = self.plot_mc_1d(bins=bins, sub_df=Dstellnu, sub_name=region, variable=variable, 
                                        ax=None, cut=cut, scale=None,correction=correction,mask=mask)
                    
                for region, df in regions.items():
                    if region=='signal region':
                        counts = self.plot_data_1d(bins=bins, sub_df=df, variable=variable, ax=ax1, 
                                                cut=cut, scale=None,name=region)
                        sig_total += counts

                    elif region in ['sidebands','left sideband', 'right sideband']:
                        fakeD_counts = self.plot_mc_1d(bins=bins, sub_df=df, sub_name=region, variable=variable, 
                                        ax=None, cut=cut, scale=None,correction=correction,mask=mask)
                        
#                         fakeD_counts -= r_D * D_counts # if plot 2 sb separately, this subtraction will be accidentally done 2 times
#                         fakeD_counts -= r_Dst * Dst_counts
                        sb_total += fakeD_counts
                        
                        ax1.hist(bins[:-1], bins, weights=unp.nominal_values(fakeD_counts),histtype='step',
                    label=f'{region} \n{self.statistics(hist=[fakeD_counts, bins],count_only=False)} ')

                # Residuals (Data - Model) and their errors
                self.plot_residuals(bins=bins, data=sig_total, model=sb_total, ax=ax2)
                ax2.set_xlabel(f'{variable}')
                
            elif type(variable)==list:
                assert merge_sidebands==True, 'merge_sidebands must be True'
                for region, df in regions.items():
                    if region=='signal region':
                        sig_total = self.plot_single_2d(bins=bins, df=df, variables=variable, 
                                                        fig=fig, ax=ax1, cut=cut, name=region)
                    elif region in ['sidebands']:
                        sb_total = self.plot_single_2d(bins=bins, df=df, variables=variable, 
                                        fig=None, ax=None, cut=cut, name=region)
                        D_counts = self.plot_single_2d(bins=bins, df=Dellnu, variables=variable, 
                                        fig=None, ax=None, cut=cut, name=region)
                        Dst_counts = self.plot_single_2d(bins=bins, df=Dstellnu, variables=variable, 
                                        fig=None, ax=None, cut=cut, name=region)
#                         sb_total -= r_D * D_counts
#                         sb_total -= r_Dst * Dst_counts
                        
                        
                        # 2D Histogram
                        im = ax2.imshow(unp.nominal_values(sb_total).T.round(0), origin='lower', aspect='auto', 
                                         cmap='rainbow', norm=mcolors.LogNorm(),
                                         extent=[bins[0][0], bins[0][-1], bins[1][0], bins[1][-1]])
                        fig.colorbar(im, ax=ax2)
                        ax2.set_xlabel('$M_{miss}^2$')
                        ax2.set_ylabel('$|p_D| + |p_{\ell}|$')
                        ax2.set_title(region)
                        ax2.grid()
                        
                # Residuals (Data - Model) and their errors
                self.plot_residuals(bins=bins, data=sig_total, model=sb_total, fig=fig, ax=ax3)
            
        elif bkg_name in ['bkg_continuum','bkg_combinatorial']:
            sample_control = self.samples[bkg_name]
            if samples_sig is None:
                print(f'Error: samples_sig is required for {bkg_name}')
                return
            else:
                sample_sig = samples_sig[bkg_name]
                
            regions = {'control region': sample_control,
                       'signal region': sample_sig}
            for region, df in regions.items():
                df.loc[:, '__weight__'] = scale[region]
            
            sig_total = self.plot_data_1d(bins=bins, sub_df=sample_sig, variable=variable, 
                                          ax=ax1,cut=cut, scale=None,name='signal region')

            control_total = self.plot_mc_1d(bins=bins, sub_df=sample_control, sub_name='control region', 
                                            variable=variable,ax=ax1,cut=cut, scale=None,
                                            correction=correction,mask=mask)
            
            # Residuals (Data - Model) and their errors
            self.plot_residuals(bins=bins, data=sig_total, model=control_total, ax=ax2)
            ax2.set_xlabel(f'{variable}')
        
        if type(variable)==str:
            ax1.set_title(f'Overlaid signal region vs control region ({bkg_name=})')
            ax1.set_ylabel(f'# of events per bin {(bins[1]-bins[0]):.3f} GeV')
            ax1.grid()
            ax1.legend(bbox_to_anchor=(1,1),ncol=legend_nc, fancybox=True, shadow=True,labelspacing=1.5, fontsize=legend_fs)
            ax2.legend(bbox_to_anchor=(1,1),fancybox=True, shadow=True, fontsize=legend_fs)
        elif type(variable)==list:
            fig.suptitle(f'signal region vs weighted control region ({bkg_name=})')
        # Adjust the layout to avoid overlapping of the subplots
        plt.tight_layout()
        plt.show()

    def plot_data_subtracted_and_mc(self,var_list,bin_list,cut=None,scale={},
                                    correction=False,mask=['bkg_fakeD'],figsize=(10,10)):
        # get data in sig and sidebands regions
        data_left = self.data.query('D_M<1.83').copy()
        data_sig = self.data.query('1.84<D_M<1.9').copy()
        data_right = self.data.query('D_M>1.91').copy()
        # get mc in sig region without fake D
        mc_sig = pd.concat([df.query('1.84<D_M<1.9').copy() for name, df in self.samples.items() if name != 'bkg_fakeD'], 
                           ignore_index=True)

        data_mc_regions = {'data left sideband': data_left,
                           'data signal region': data_sig,
                           'data right sideband': data_right,
                           'mc signal region': mc_sig}

        for region, df in data_mc_regions.items():
            df.loc[:, '__weight__'] = scale[region]
            
        # calculate the 2d hists
        data_sb_2d = 0 
        data_sig_2d = 0
        mc_sig_2d = 0
        variable_x, variable_y = var_list
        edges_x, edges_y = bin_list
        if var_list==['B0_CMS3_weMissM2','p_D_l']:
            var_x_label = '$M_{miss}^2$    [$GeV^2/c^4$]'
            var_y_label = '$|p_D| + |p_{\ell}|$    [GeV/c]'
        else:
            var_x_label = var_list[0]
            var_y_label = var_list[1]
        
        for region, df in data_mc_regions.items():
            (counts_2d, xe, ye) = np.histogram2d(
                df.query(cut)[variable_x] if cut else df[variable_x], 
                df.query(cut)[variable_y] if cut else df[variable_y],
                bins=[edges_x, edges_y],
                weights=df.query(cut)['__weight__'] if cut else df['__weight__'])
            (staterr_squared_2d, edges_x, edges_y) = np.histogram2d(
                df.query(cut)[variable_x] if cut else df[variable_x], 
                df.query(cut)[variable_y] if cut else df[variable_y],
                bins=[edges_x, edges_y],
                weights=df.query(cut)['__weight__']**2 if cut else df['__weight__']**2)
            staterr_2d = np.sqrt(staterr_squared_2d)

            # Ensure that both arrays have the same shape
            assert counts_2d.shape == staterr_2d.shape, \
                f"Shape mismatch between hist counts and staterror for 2d data in {region=}"
            # Combine the count values with their uncertainties
            counts_uncert_2d = unp.uarray(counts_2d.round(0), staterr_2d.round(0))
            
            if region=='data signal region':
                data_sig_2d += counts_uncert_2d

            elif region=='mc signal region':
                mc_sig_2d += counts_uncert_2d
                
            elif region in ['data left sideband', 'data right sideband']:
                data_sb_2d += counts_uncert_2d

        # subtract the sidebands from sig region
        data_subtracted_2d = data_sig_2d - data_sb_2d

        # get 2 projections
        data_subtracted_x = data_subtracted_2d.sum(axis=1)  # Sum along the y-axis
        data_subtracted_y = data_subtracted_2d.sum(axis=0)  # Sum along the x-axis
           
        
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
        im = ax1.imshow(abs(unp.nominal_values(data_subtracted_2d)).T, origin='lower', aspect='auto', 
                         cmap='rainbow', norm=mcolors.LogNorm(),
                         extent=[edges_x[0], edges_x[-1], edges_y[0], edges_y[-1]])
        fig.colorbar(im, ax=ax1)
        ax1.set_title('Data, D_M sidebands subtracted')
        ax1.set_xlabel(var_x_label)
        ax1.set_ylabel(var_y_label)
        ax1.grid()
    
        
        # Top-right: 1D histogram of p_D_l projection + residuals
        data_y = self.plot_data_1d(bins=edges_y, hist=data_subtracted_y, ax=ax2,cut=cut, name='Data')
        mc_y = self.plot_mc_1d(bins=edges_y, variable=variable_y, ax=ax2, scale=scale['mc signal region'],
                               cut='1.84<D_M<1.9', correction=correction,mask=mask,legend_count=True)
        if var_list==['B0_CMS3_weMissM2','p_D_l']:
            ax2.set_title('$|p_D| + |p_{\ell}|$ Projection')
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
        mc_x = self.plot_mc_1d(bins=edges_x, variable=variable_x, ax=ax4, scale=scale['mc signal region'],
                               cut='1.84<D_M<1.9', correction=correction,mask=mask,legend_count=True)
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
        im = ax6.imshow(unp.nominal_values(mc_sig_2d).T, origin='lower', aspect='auto', 
                         cmap='rainbow', norm=mcolors.LogNorm(),
                         extent=[edges_x[0], edges_x[-1], edges_y[0], edges_y[-1]])
        fig.colorbar(im, ax=ax6)
        ax6.set_title('MC, fakeD removed')
        ax6.set_xlabel(var_x_label)
        ax6.set_ylabel(var_y_label)
        ax6.grid()
        
        # Adjust layout to avoid overlap
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.4, wspace=0.4)
        plt.show()

    def plot_FOM(self, sigModes, bkgModes, variable, test_points, cut=None, reverse=False):
        # define signal / bkg sample
        sig = pd.concat([self.samples[i] for i in sigModes])
        bkg = pd.concat([self.samples[i] for i in bkgModes])
        # of events before cut
        sig_tot = ufloat( len(sig), np.sqrt(len(sig)) )
        bkg_tot = ufloat( len(bkg), np.sqrt(len(bkg)) )
        
        FOM_list = []
        sigEff_list = []
        bkgEff_list = []

        for i in test_points:
            # of events after cut
            if reverse:
                x = f'{variable}<{i}'
            else:
                x = f'{variable}>{i}'
            
            nsig_val = len(sig.query(f"{cut} and {x}" if cut else f"{x}"))
            nbkg_val = len(bkg.query(f"{cut} and {x}" if cut else f"{x}"))
            
            nsig = ufloat(nsig_val, np.sqrt(nsig_val))
            nbkg = ufloat(nbkg_val, np.sqrt(nbkg_val))
            ntot = nsig+nbkg
            # calculation
            if ntot==0:
                FOM = ufloat(0,0)
            else:
                FOM = nsig / ntot**0.5 # s / √(s+b)
            sigEff = nsig / sig_tot
            bkgEff = nbkg / bkg_tot

            FOM_list.append(FOM)
            sigEff_list.append(sigEff)
            bkgEff_list.append(bkgEff)


        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        color = 'tab:blue'
        ax1.set_ylabel('Efficiency', color=color)  # we already handled the x-label with ax1
        ax1.errorbar(x=test_points, y=[E.n for E in sigEff_list], yerr=[E.s for E in sigEff_list],
                     marker='o',label='Signal Efficiency',color=color)
        ax1.errorbar(x=test_points, y=[E.n for E in bkgEff_list], yerr=[E.s for E in bkgEff_list],
                     marker='o',label='Bkg Efficiency',color='green')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')
        ax1.grid()
        ax1.set_xlabel('Signal Probability')
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
        color = 'tab:red'
        ax2.set_ylabel('FOM', color=color)
        ax2.errorbar(x=test_points, y=[F.n for F in FOM_list], yerr=[F.s for F in FOM_list],
                     marker='o',label='FOM',color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        #ax2.grid()
        ax2.legend(loc='upper right')

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title(f'FOM for {variable=}')
        plt.xlim(0,1)
        plt.ylim(bottom=0)
        plt.show()
        
        
    def plot_all_2Dhist(self, bin_list:list, var_list=['B0_CMS3_weMissM2','p_D_l'], cut=None, mask=[1.6,1]):
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

        fig.suptitle(f'Generic MC 200/fb ({cut=})', y=0.92, fontsize=18)
        fig.supylabel(r'$|p^\ast_{D}|+|p^\ast_{\ell}| \ \ [GeV]$', x=0.05,fontsize=18)
        fig.supxlabel('$M_{miss}^2\ \ \ [GeV^2/c^4]$', y=0.08,fontsize=18)
        
       
        

        
        
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
            ax[0].set_ylabel('$|p_D| + |p_{\ell}|$', fontsize=14)
            ax[0].grid()

            # X Projection
            ax[1].bar(xedges[:-1], x_projection, width=np.diff(xedges), align='edge')
            ax[1].set_title('$M_{miss}^2$ Projection')
            ax[1].set_xlabel('$M_{miss}^2$')
            ax[1].set_ylabel('Counts')
            ax[1].grid()

            # Y Projection
            ax[2].barh(yedges[:-1], y_projection, height=np.diff(yedges), align='edge')
            ax[2].set_title('$|p_D| + |p_{\ell}|$ Projection')
            ax[2].set_xlabel('Counts')
            ax[2].set_ylabel('$|p_D| + |p_{\ell}|$')
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
        
    def plot_cut_efficiency(self, cut, variable='B0_CMS3_weQ2lnuSimple',bins=15,xlim=[2,12],comp=[r'$D\tau\nu$',r'$D\ell\nu$']):
        plt.title(f'Efficiency for {cut=}', y=1,fontsize=16);
        plt.xlabel(f'{variable}',fontsize=16)
        plt.ylabel('Efficiency',x=0.06,fontsize=16)
        plt.grid()
        plt.xlim(xlim);
        plt.ylim(0,1);
        #fig.supxlabel('$|\\vec{p_D}|\ +\ |\\vec{p_l}|$  [GeV/c]')
        #fig.supxlabel('$M_{miss}^2 \ [GeV^2/c^4]$')
        
        sub_samples={key:value for key, value in self.samples.items() if key in comp}
        for name, df in sub_samples.items():
            (bc, bins1) = np.histogram(df[variable], bins=bins)
            (ac, bins1) = np.histogram(df.query(cut)[variable], bins=bins1)
            bc+=1
            ac+=1
            efficiency = ac / bc
            efficiency_err = efficiency * np.sqrt(1/ac + 1/bc)
            bin_centers = (bins1[:-1] + bins1[1:]) /2
            plt.errorbar(x=bin_centers, y=efficiency, yerr=efficiency_err, label=name)
            plt.legend()

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
                        'bkg_TDFl',     'bkg_singleBbkg',   r'$D\ell\nu$_gap',
                        r'$D^{\ast\ast}\ell\nu$',           r'$D^{\ast\ast}\tau\nu$',
                        r'$D^\ast\ell\nu$',                 r'$D\ell\nu$',
                        r'$D^\ast\tau\nu$',                 r'$D\tau\nu$']
        
        # plot data and fitted values
        data_val = unp.nominal_values(data_1d)
        data_err = unp.std_devs(data_1d)
        ax1.errorbar(x=bin_centers, y=data_val, yerr=data_err, fmt='ko')
        
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
                        
        ax2.errorbar(x=bin_centers, y=residual_val, yerr=residual_err, fmt='ko')
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
        other_axis_label = '$|p_D|\ +\ |p_l|$'
        other_axis_unit = '[GeV]'
    elif direction == 'p_D_l':
        axis = 1
        axis_label = '$|p_D|\ +\ |p_l|$'
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
    if direction not in ['mm2', 'p_D_l'] or plot_with not in ['mplhep', 'pltbar']:
        raise ValueError('direction in [mm2, p_D_l] and plot_with in [mplhep, pltbar]')
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
        fitted_err = np.sqrt(np.sum(np.array(template_err)**2, axis=0)) # assuming the correlations between each template are 0
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
        residual_err = np.sqrt(data_project + np.sum(np.array(templates_project_err)**2, axis=0))
        residual_err_signal = np.sqrt(residual_err**2 - np.array(templates_project_err[signal_index]))

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
        data_err = np.sqrt(data_project)

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
        ax1.errorbar(x=bin_centers, y=data_project, yerr=data_err, fmt='ko')
        # plot the residual
        residual = data_project - np.sum(templates_project, axis=0)
        # Error assuming the correlations between data and templates, between each template, are 0
        residual_err = np.sqrt(data_project + np.sum(np.array(templates_project_err)**2, axis=0))

        pull = [0 if residual_err[i]==0 else (residual[i]/residual_err[i]) for i in range(len(residual))]
        ax2.errorbar(x=bin_centers, y=residual, yerr=residual_err, fmt='ko')
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
#         residual_err_signal = np.sqrt(residual_err**2 - np.array(templates_project_err[signal_index]))
#         pull_signal = [0 if residual_err_signal[i]==0 else (residual_signal[i]/residual_err_signal[i]) for i in range(len(residual_signal))]

    if direction=='mm2':
        direction_label = '$M_{miss}^2$'
        direction_unit = '$[GeV^2/c^4]$'
        other_direction_label = '$|p_D|\ +\ |p_l|$'
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
        fitted_project_slice1_err = [np.sqrt((err**2)[:first_slice_index,:].sum(axis=axis_to_be_summed_over)) for err in fitted_templates_err]
        fitted_project_slice2_err = [np.sqrt((err**2)[second_slice_index:,:].sum(axis=axis_to_be_summed_over)) for err in fitted_templates_err]
        data_slice1 = data_2d[:first_slice_index,:]
        data_slice2 = data_2d[second_slice_index:,:]

    elif direction=='p_D_l':
        direction_label = '$|p_D|\ +\ |p_l|$'
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
        fitted_project_slice1_err = [np.sqrt((err**2)[:,:first_slice_index].sum(axis=axis_to_be_summed_over)) for err in fitted_templates_err]
        fitted_project_slice2_err = [np.sqrt((err**2)[:,second_slice_index:].sum(axis=axis_to_be_summed_over)) for err in fitted_templates_err]
        data_slice1 = data_2d[:,:first_slice_index]
        data_slice2 = data_2d[:,second_slice_index:]

    else:
        raise ValueError('Current version only supports projection to either MM2 or p_D_l')

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
#         other_direction_label = '$|p_D|\ +\ |p_l|$'
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
#         direction_label = '$|p_D|\ +\ |p_l|$'
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
