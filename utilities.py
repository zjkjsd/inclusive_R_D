# -*- coding: utf-8 -*-
# +
# define BDT training variables
spectators = ['__weight__', 'D_CMS_p', 'ell_CMS_p', 'B0_CMS3_weMissM2', 'B0_CMS3_weQ2lnuSimple']

CS_variables = ["B0_R2",       "B0_thrustOm",   "B0_cosTBTO",    "B0_cosTBz",
                "B0_KSFWV3",   "B0_KSFWV4",     "B0_KSFWV5",     "B0_KSFWV6",
                "B0_KSFWV7",   "B0_KSFWV8",     "B0_KSFWV9",     "B0_KSFWV10",
                "B0_KSFWV13",  "B0_KSFWV14",    "B0_KSFWV15",    "B0_KSFWV16",
                "B0_KSFWV17",  "B0_KSFWV18",    "B0_CC1",        "B0_CC2",
                "B0_CC3",      "B0_CC4",        "B0_CC5",        "B0_CC6",
                "B0_CC7",      "B0_CC8",        "B0_CC9",]
                #"B0_thrustBm", "B0_KSFWV1",     "B0_KSFWV2",     "B0_KSFWV11",
                #"B0_KSFWV12",] correlates with mm2 or p_D_l

DTC_variables = ['D_K_kaonID_binary_noSVD',    'D_pi1_kaonID_binary_noSVD',
                 'D_pi2_kaonID_binary_noSVD',  'D_K_dr', 
                 'D_pi1_dr',                   'D_pi2_dr',
                 'D_K_dz',                     'D_pi1_dz', 
                 'D_pi2_dz',                   'D_K_pValue', 
                 'D_pi1_pValue',               'D_pi2_pValue',#?
                 'D_vtxReChi2',                #'D_BFInvM', D mass will suppress the sideband
                 'D_A1FflightDistanceSig_IP',  'D_daughterInvM_0_1',
                 'D_daughterInvM_1_2',         'B0_vtxDDSig',]

B_variables = ['B0_Lab5_weMissPTheta',
               'B0_vtxReChi2',               'B0_flightDistanceSig',
               'B0_nROE_Tracks_my_mask',     'B0_nROE_NeutralHadrons_my_mask', # can remove
               'B0_roel_DistanceSig_dis',    'B0_roeDeltae_my_mask',
               'B0_roeEextra_my_mask',       'B0_roeMbc_my_mask',
               'B0_nROE_Photons_my_mask',    'B0_roeCharge_my_mask', # can remove
               'B0_nROE_K',                  'B0_TagVReChi2IP',]

training_variables = CS_variables + DTC_variables + B_variables
mva_variables = training_variables + spectators

analysis_variables=['__experiment__',     '__run__',       '__event__',      '__production__',
                    'B0_isContinuumEvent','B0_mcPDG',      'B0_mcErrors',    'B0_mcDaughter_0_PDG',
                    'B0_mcDaughter_1_PDG','D_mcErrors',    'D_genGMPDG',     'D_genMotherPDG',
                    'D_mcPDG',            'D_BFM',         'D_M',            'D_px',
                    'D_py',               'D_pz',          'D_p',            'D_CMS_p',
                    'D_K_mcErrors',       'D_pi1_mcErrors','D_pi2_mcErrors', 'D_K_pValue', 
                    'D_pi1_pValue',       'D_pi2_pValue',
                    'ell_genMotherPDG',   'ell_mcPDG',     'ell_mcErrors',   'ell_genGMPDG',
                    'ell_px',             'ell_py',        'ell_pz',         'ell_p',
                    'ell_CMS_p',          'ell_pSig',      'ell_pValue',
#                     'ell_GMdaughter_0_PDG',                'ell_GMdaughter_1_PDG',
#                     'ell_Mdaughter_0_PDG',                 'ell_Mdaughter_1_PDG'
                    'mode',               'p_D_l',         'B_D_ReChi2',
                    'D_daughter_pValue_min',           'D_daughter_pValue_mean',
                    'signal_prob', 'fakeD_prob', 'continuum_prob', 'fakeB_prob']

all_relevant_variables = mva_variables + analysis_variables


DecayMode = {'bkg_fakeD':0,           'bkg_continuum':1,    'bkg_combinatorial':2,
             'bkg_Odecay':3,          'bkg_fakeTC':4,       r'$D\tau\nu$':5,
             r'$D^\ast\tau\nu$':6,    r'$D\ell\nu$':7,      r'$D^\ast\ell\nu$':8,
             r'$D^{\ast\ast}\tau\nu$_mixed':9,              r'$D^{\ast\ast}\tau\nu$_charged':10,
             r'res_$D^{\ast\ast}\ell\nu$_mixed':11,         r'nonres_$D^{\ast\ast}\ell\nu$_mixed':12,
             r'gap_$D^{\ast\ast}\ell\nu$_mixed':13,         r'res_$D^{\ast\ast}\ell\nu$_charged':14,
             r'nonres_$D^{\ast\ast}\ell\nu$_charged':15,    'bkg_others':16}

DecayMode_new = {'bkg_fakeTracks':0,         'bkg_FakeD':1,           'bkg_TDFl':2,
                 'bkg_continuum':3,          'bkg_combinatorial':4,   'bkg_singleBbkg':5,
                 'bkg_other_TDTl':6,         'bkg_other_signal':7,
                 r'$D\tau\nu$':8,            r'$D^\ast\tau\nu$':9,    r'$D\ell\nu$':10,
                 r'$D^\ast\ell\nu$':11,                r'$D^{\ast\ast}\tau\nu$':12,
                 r'$D^{\ast\ast}\ell\nu$':13,          r'$D\ell\nu$_gap':14}

# +
## dataframe samples
import numpy as np
import pandas as pd
import lightgbm as lgb

def apply_mva_bcs(df, features, cut):
    # load model
    bst_lgb = lgb.Booster(model_file='/home/belle/zhangboy/inclusive_R_D/BDTs/LightGBM/lgbm_multiclass.txt')
    # predict
    pred = bst_lgb.predict(df[features], num_iteration=50) #bst_lgb.best_iteration
    lgb_out = pd.DataFrame(pred, columns=['signal_prob','continuum_prob','fakeD_prob','fakeB_prob'])
    # combine the predict result
    df_lgb = pd.concat([df, lgb_out], axis=1)
    df_lgb['largest_prob'] = df_lgb[['signal_prob','continuum_prob','fakeD_prob','fakeB_prob']].max(axis=1)
    del pred,lgb_out
    
    # apply the MVA cut and BCS
    df_cut=df_lgb.query(cut)
    df_bestSelected=df_cut.loc[df_cut.groupby(['__experiment__','__run__','__event__','__production__'])['B_D_ReChi2'].idxmin()]
    
    return df_bestSelected


def get_dataframe_samples_new(df, mode, template=True) -> dict:
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

    Dstst_list = [10413, 10411, 20413, 415, 10423, 10421, 20423, 425,
                 -10413, -10411, -20413, -415, -10423, -10421, -20423, -425]
    B2Dstst_tau = f'{signals} and B0_mcDaughter_0_PDG in @Dstst_list and abs(B0_mcDaughter_1_PDG)==15'
    B2Dstst_ell = f'{signals} and B0_mcDaughter_0_PDG in @Dstst_list and abs(B0_mcDaughter_1_PDG)=={lepton_PDG[mode]}'

    D_Dst_list = [411, 413, -411, -413]
    Pi_eta_list = [111, 211, -211, 221]
    B2D_ell_gap = f'{signals} and B0_mcDaughter_0_PDG in @D_Dst_list and B0_mcDaughter_1_PDG in @Pi_eta_list'
    
    ######################### Apply selection ###########################
    
    # Fake bkg components
    bkg_FakeD = df.query(FD).copy()
    bkg_TDFl = df.query(TDFl).copy()
    bkg_fakeTracks = df.query(fakeTracks).copy()
    samples['bkg_FakeD'] = bkg_FakeD
    samples['bkg_TDFl'] = bkg_TDFl
    samples['bkg_fakeTracks'] = bkg_fakeTracks
    
    # True Dl bkg components
    bkg_continuum = df.query(continuum).copy()
    bkg_combinatorial = df.query(combinatorial).copy()
    bkg_singleBbkg = df.query(singleBbkg).copy()
    signals_all = df.query(signals).copy()
    bkg_other_TDTl = pd.concat([df.query(TDTl).copy(),
                                bkg_singleBbkg,
                                bkg_combinatorial,
                                bkg_continuum,
                                signals_all]).drop_duplicates(
        subset=['__experiment__','__run__','__event__','__production__'],keep=False)
    
    samples['bkg_continuum'] = bkg_continuum
    samples['bkg_combinatorial'] = bkg_combinatorial
    samples['bkg_singleBbkg'] = bkg_singleBbkg
    samples['bkg_other_TDTl'] = bkg_other_TDTl
    
    # True Dl Signal components
    D_tau_nu=df.query(B2D_tau).copy()
    D_l_nu=df.query(B2D_ell).copy()
    Dst_tau_nu=df.query(B2Dst_tau).copy()
    Dst_l_nu=df.query(B2Dst_ell).copy()
    Dstst_tau_nu=df.query(B2Dstst_tau).copy()
    Dstst_l_nu=df.query(B2Dstst_ell).copy()
    D_l_nu_gap=df.query(B2D_ell_gap).copy()
    
    bkg_other_signal = pd.concat([signals_all,
                                  D_tau_nu,
                                  Dst_tau_nu,
                                  D_l_nu,
                                  Dst_l_nu,
                                  Dstst_tau_nu,
                                  Dstst_l_nu,
                                  D_l_nu_gap]).drop_duplicates(
        subset=['__experiment__','__run__','__event__','__production__'],keep=False)
    
    samples[r'$D\tau\nu$'] = D_tau_nu
    samples[r'$D^\ast\tau\nu$'] = Dst_tau_nu
    samples[r'$D\ell\nu$'] = D_l_nu
    samples[r'$D^\ast\ell\nu$'] = Dst_l_nu
    samples[r'$D^{\ast\ast}\tau\nu$'] = Dstst_tau_nu
    samples[r'$D^{\ast\ast}\ell\nu$'] = Dstst_l_nu
    samples[r'$D\ell\nu$_gap'] = D_l_nu_gap
    samples['bkg_other_signal'] = bkg_other_signal
    
    for name, df in samples.items():
        df['mode']=DecayMode_new[name]
    return samples


# +
# Function to rebin a histogram
def rebin_histogram(counts, bin_edges, threshold):
    new_counts = []
    new_edges = [bin_edges[0]]
    
    i = 0
    while i < len(counts):
        bin_count = counts[i]
        start_edge = bin_edges[i]
        end_edge = bin_edges[i + 1]
        
        # Merge bins until bin_count is above the threshold
        while bin_count < threshold and i < len(counts) - 1:
            i += 1
            bin_count += counts[i]
            end_edge = bin_edges[i + 1]
        
        new_counts.append(bin_count)
        new_edges.append(end_edge)
        
        i += 1
    
    return np.array(new_counts), np.array(new_edges)

# new_counts, new_bin_edges = rebin_histogram(counts, bin_edges, threshold)

# Function to rebin another histogram using new bin edges
def rebin_histogram_with_new_edges_and_uncertainties(counts, uncertainties, old_bin_edges, new_bin_edges):
    # Rebin the counts using np.histogram
    new_counts, _ = np.histogram(np.repeat(old_bin_edges[:-1], counts), bins=new_bin_edges)
    
    # Initialize new uncertainties array
    new_uncertainties = np.zeros_like(new_counts, dtype=float)
    
    # Combine uncertainties in the new bins
    for i in range(len(old_bin_edges) - 1):
        bin_value = old_bin_edges[i]
        new_bin_index = np.digitize(bin_value, new_bin_edges) - 1
        new_uncertainties[new_bin_index] += uncertainties[i] ** 2  # Sum uncertainties in quadrature

    # Take the square root of the sum of squares
    new_uncertainties = np.sqrt(new_uncertainties)
    return new_counts, new_uncertainties

# new_counts_B, new_uncertainties_B = rebin_histogram_with_new_edges_and_uncertainties(counts_B, uncertainties_B, bin_edges_B, new_bin_edges_A)

def create_templates(samples:dict, bins:list, scale_lumi=False,
                     variables=['B0_CMS3_weMissM2','p_D_l'],
                     bin_threshold=1, merge_threshold=10,
                     sample_to_exclude=['bkg_fakeTracks','bkg_other_TDTl','bkg_other_signal']):
    #################### Create template 2d histograms ################
    histograms = {}
    staterr = {}
    for name, df in samples.items():
        if name in sample_to_exclude:
            continue

        (counts, xedges, yedges) = np.histogram2d(df[variables[0]], 
                                                  df[variables[1]],
                                                  bins=bins,
                                                  weights=df['__weight__'])

        (staterr_squared, xedges, yedges) = np.histogram2d(df[variables[0]], 
                                                           df[variables[1]],
                                                           bins=bins,
                                                           weights=df['__weight__']**2)
        histograms[name] = counts.T
        if scale_lumi:
            staterr[name] = np.sqrt(counts.T)
        else:
            staterr[name] = np.sqrt(staterr_squared.T)

    ################### Trimming and flattening ###############
    # remove bins with count smaller than bin_threshold
    indices_threshold = np.where(np.sum(list(histograms.values()),axis=0) >= bin_threshold)
    template_flat = {name:t[indices_threshold].tolist() for name,t in histograms.items()}
    staterr_flat = {name:se[indices_threshold].tolist() for name,se in staterr.items()}

    asimov_data = np.sum(list(template_flat.values()),axis=0).tolist()

    ################## Create a new set of templates whose adjacent bins are merged if <10 counts

    dummy_bin_edges = np.arange(len(asimov_data)+1)
    new_counts, new_dummy_bin_edges = rebin_histogram(counts=asimov_data, 
                                                      bin_edges=dummy_bin_edges, 
                                                      threshold=merge_threshold)
    print(f'original template length = {len(asimov_data)}')
    print(f'new template length = {len(new_counts)}')

    template_flat_merged = {}
    staterr_flat_merged = {}
    for name, t in template_flat.items():
        new_c, new_err = rebin_histogram_with_new_edges_and_uncertainties(counts=t,
                                                                        uncertainties=staterr_flat[name], 
                                                                        old_bin_edges=dummy_bin_edges,
                                                                        new_bin_edges=new_dummy_bin_edges)
        template_flat_merged[name] = new_c.tolist()
        staterr_flat_merged[name] = new_err.tolist()

    asimov_data_merged = np.sum(list(template_flat_merged.values()),axis=0).tolist()
    
    return indices_threshold,(template_flat,staterr_flat,asimov_data),(template_flat_merged,staterr_flat_merged,asimov_data_merged)

def create_2d_template(template_flat: dict, staterror_flat: dict, data_flat,indices_threshold: tuple, bins: list):
    """
    Convert a flattened 1D template back to a 2D histogram based on provided binning.

    Parameters:
        template_flat (dict): Flattened 1D templates for different samples.
        indices_threshold (tuple): The indices that correspond to the bins retained after applying the threshold.
        bins (list): The bin edges for the 2D histogram.

    Returns:
        histograms (dict): 2D histograms for each sample.
        staterr (dict): Statistical uncertainties for each 2D histogram.
    """
    histograms = {}
    staterr = {}
    
    xbins, ybins = bins
    for name, flat_data in template_flat.items():
        # Initialize an empty 2D array for histogram and statistical error
        hist_2d = np.zeros((len(xbins) - 1, len(ybins) - 1)).T
        # Assign the flattened data back into the appropriate indices
        hist_2d[indices_threshold] = flat_data
        histograms[name] = hist_2d
        
    for name, flat_stat in staterror_flat.items():
        # Initialize an empty 2D array for histogram and statistical error
        staterr_2d = np.zeros((len(xbins) - 1, len(ybins) - 1)).T
        # Assign the flattened data back into the appropriate indices
        staterr_2d[indices_threshold] = flat_stat
        staterr[name] = staterr_2d
        
    data_2d = np.zeros((len(xbins) - 1, len(ybins) - 1)).T
    data_2d[indices_threshold] = data_flat
    
    return histograms, staterr, data_2d

def update_workspace(workspace: dict, temp_asimov_sets: list, staterror: bool = True) -> dict:
    names = list(temp_asimov_sets[0][0].keys())
    
    for ch_index in range(len(temp_asimov_sets)):
        template_flat = temp_asimov_sets[ch_index][0]
        staterr_flat = temp_asimov_sets[ch_index][1]
        asimov_data = temp_asimov_sets[ch_index][2]
        
        # Update the number of samples to match the new names
        current_samples = workspace['channels'][ch_index]['samples']
        if len(current_samples) < len(names):
            # Add missing samples
            for _ in range(len(names) - len(current_samples)):
                current_samples.append({'name': '', 'data': [], 'modifiers': []})
        elif len(current_samples) > len(names):
            # Remove extra samples
            workspace['channels'][ch_index]['samples'] = current_samples[:len(names)]
        
        # Update samples
        for samp_index, sample in enumerate(workspace['channels'][ch_index]['samples']):
            sample['name'] = names[samp_index]
            sample['data'] = template_flat[names[samp_index]]
            
            if 'staterror' not in [m['type'] for m in sample['modifiers']] and staterror:
                sample['modifiers'].append({'type': 'staterror'})
            
            for mod_index, m in enumerate(sample['modifiers']):
                if m['type'] == 'staterror':
                    if staterror:
                        m['data'] = staterr_flat[names[samp_index]]
                        m['name'] = f'staterror_{ch_index}_channel'
                    else:
                        # Safely remove the 'staterror' modifier if not needed
                        sample['modifiers'] = [mod for mod in sample['modifiers'] if mod['type'] != 'staterror']
                elif m['type'] == "normfactor":
                    m['name'] = names[samp_index] + '_norm'
                else:
                    print(m['type'], 'is turned on')
                    
        workspace['observations'][ch_index]['data'] = asimov_data

    # Update measurement parameters
    current_parameters = workspace["measurements"][0]["config"]["parameters"]
    if len(current_parameters) < len(names):
        # Add missing parameters
        for _ in range(len(names) - len(current_parameters)):
            current_parameters.append({'name': '', 'bounds': [[0,2]], 'inits': [1]})
    elif len(current_parameters) > len(names):
        # Remove extra parameters
        workspace["measurements"][0]["config"]["parameters"] = current_parameters[:len(names)]
    for i, par in enumerate(workspace["measurements"][0]["config"]["parameters"]):
        par['name'] = names[i] + '_norm'
        if par['name'].startswith('bkg'):
            par['bounds'] = [[0,2]]
        else:
            par['bounds'] = [[-5,5]]
    
    workspace["measurements"][0]["config"]['poi'] = "$D\\tau\\nu$_norm"

    return workspace

# for samp_index, sample in enumerate(workspace['channels'][ch_index]['samples']):
#     sample = {'name': 'new_sample'}  # This would not update the list in `workspace`
# Using sample as a reference to the list element is perfectly fine and will not cause bugs 
# as long as you're modifying the contents of sample (like updating values or appending to a list). 
# However, be cautious when assigning a completely new value to sample itself, 
# as that won't update the original list.



# +
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
    
def calculate_FOM3d(sig_data, bkg_data, variables, test_points):
    sig = pd.concat(sig_data)
    bkg = pd.concat(bkg_data)
    sig_tot = len(sig)
    bkg_tot = len(bkg)
    BDT_FOM = []
    BDT_FOM_err = []
    BDT_sigEff = []
    BDT_sigEff_err = []
    BDT_bkgEff = []
    BDT_bkgEff_err = []
    for i in test_points[0]:
        for j in test_points[1]:
            for k in test_points[2]:
                nsig = len(sig.query(f"{variables[0]}>{i} and {variables[1]}>{j} and {variables[2]}>{k}"))
                nbkg = len(bkg.query(f"{variables[0]}>{i} and {variables[1]}>{j} and {variables[2]}>{k}"))
                tot = nsig+nbkg
                tot_err = np.sqrt(tot)
                FOM = nsig / tot_err # s / √(s+b)
                FOM_err = np.sqrt( (tot_err - FOM/2)**2 /tot**2 * nsig + nbkg**3/(4*tot**3) + 9*nbkg**2*np.sqrt(nsig*nbkg)/(4*tot**5) )

                BDT_FOM.append(round(FOM,2))
                BDT_FOM_err.append(round(FOM_err,2))

                sigEff = nsig / sig_tot
                sigEff_err = sigEff * np.sqrt(1/nsig + 1/sig_tot)
                bkgEff = nbkg / bkg_tot
                bkgEff_err = bkgEff * np.sqrt(1/nbkg + 1/bkg_tot)
                BDT_sigEff.append(round(sigEff,2))
                BDT_sigEff_err.append(round(sigEff_err,2))
                BDT_bkgEff.append(round(bkgEff,2))
                BDT_bkgEff_err.append(round(bkgEff_err,2))
    print(f'{BDT_FOM=}')
    print(f'{BDT_sigEff=}')
    print(f'{BDT_bkgEff=}')


# +
## 1d fit
from iminuit import cost, Minuit
from scipy.stats import norm
from scipy.integrate import quad
from uncertainties import ufloat, correlated_values
import uncertainties.unumpy as unp

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

        

class fit_iminuit:
    def __init__(self,x_edges, hist):
        self.x_edges = x_edges
        self.y_val = unp.nominal_values(hist)
        self.y_err = unp.std_devs(hist)
        self.x_min = min(x_edges)
        self.x_max = max(x_edges)
        # self.x_mask_bkg = (x < 1.82) | (x > 1.921)
        # self.x_mask_sig_bkg = (x < 1.82) | (x > 1.921) | ((1.857 < x) & (x < 1.885))

    # np.polynomial.Polynomial.fit and np.polyval handle the order of polynomial coefficients differently.
    # np.polyval expects the coefficients in decreasing order of powers, i.e., from the highest degree term to the constant term.
    # np.polynomial.Polynomial stores the coefficients in increasing order of powers (from the constant term to the highest degree).
        
    # fit polynomial
    def gauss_polyno(self, x, par):
        return par[0] * norm.pdf(x, par[1], par[2]) + np.polyval(par[3:], x)# for len(par) == 2, this is a line
    
    def gauss_poly_cdf(self, x, *par):
        sig_gauss = par[0] * norm.cdf(x, par[1], par[2])
        bkg_poly = par[3] * polynomial(par[4:],self.x_min,self.x_max).cdf(x)
        return sig_gauss + bkg_poly
        
    def estimate_init(self, x, y, deg):
        # polynomial
        p = np.polynomial.Polynomial.fit(x, y, deg=deg)
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
            x = self.x_edges[:-1] + np.diff(self.x_edges)[0]
        if y_val is None:
            y_val = self.y_val
            y_err = self.y_err
        g_mean, g_std, p_init = self.estimate_init(x,y_val,deg)
        init = np.array([round(y_val.sum()/5000,1), g_mean, g_std, *p_init])
        print('initial parameters=', init)
        
        # cost function and minuit
        c = cost.LeastSquares(x,y_val,y_err,model=self.gauss_polyno,loss=loss)
        m = Minuit(c, init)

        # fit the bkg in sideband first
        m.limits["x0", "x1", "x2"] = (0, None)
        m.fixed["x0", "x1", "x2"] = True
        # temporarily mask out the signal
        c.mask = (x < 1.82) | (x > 1.921)
        m.simplex().migrad()

        # fit the signal with the bkg fixed
        c.mask = (x < 1.82) | (x > 1.921) | ((1.857 < x) & (x < 1.885)) # include the signal
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
        g_mean, g_std, p_init = self.estimate_init(xe[:-1],hist,deg)
        init = np.array([round(hist.sum()/10,1), g_mean, g_std, round(hist.sum(),1),*p_init])
        print('initial parameters=', init)
            
        # cost function and minuit
        c = cost.ExtendedBinnedNLL(n=hist,xe=xe,scaled_cdf=self.gauss_poly_cdf) 
        m = Minuit(c, *init)
        
        # fit the bkg in sideband first
        m.limits["x0", "x1", "x2"] = (0, None)
        m.fixed["x0", "x1", "x2"] = True
        # we temporarily mask out the signal
        cx = 0.5 * (xe[1:] + xe[:-1])
        c.mask = (cx < 1.819) | (cx > 1.921)
        m.simplex().migrad()

        # fit the signal with the bkg fixed
        c.mask = (cx < 1.819) | (cx > 1.921) | ((1.856 < cx) & (cx < 1.884)) # include the signal
        m.fixed = False  # release all parameters
        m.fixed["x3","x4","x5"] = True  # fix background amplitude
        m.simplex().migrad()

        # fit everything together to get the correct uncertainties
        m.fixed = False
        m.migrad()
        
        result = correlated_values(m.values, m.covariance)
        return m, c, result
        
        
    def plot_result(self, x, y, yerr, result):
        # Generate x, y values for plotting the fitted function
        x_plot = np.linspace(min(x), max(x), 500)
        y_plot = self.polyno(x_plot, result)

        # Calculate y and residual for plotting the residual plot
        y_fit = self.polyno(x, result)
        y_data = unp.uarray(y, yerr)
        residual = y_data - y_fit

        # Create a figure with two subplots: one for the histogram, one for the residual plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), gridspec_kw={'height_ratios': [5, 1]})
    
        # Plot data points and fitted function in ax1
        ax1.errorbar(x, y, yerr, fmt='o', label='Data')
        ax1.plot(x_plot, unp.nominal_values(y_plot), label='Fitted polynomial', color='red')
        ax1.grid()
        ax1.legend()
        ax1.set_ylabel('n Events per bin')
        #plt.ylim(0,1)

        # Plot the residuals in ax2
        ax2.errorbar(x, unp.nominal_values(residual), yerr=unp.std_devs(residual), fmt='o', color='black')
        # Add a horizontal line at y=0 for reference
        ax2.axhline(0, color='gray', linestyle='--')
        # Label the residual plot
        ax2.set_ylabel('Residuals')
        # ax2.set_xlabel(f'{variable}')
        
        # Adjust the layout to avoid overlapping of the subplots
        plt.tight_layout()
        # Show the plot
        plt.show()

# +
## Plotting
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib import gridspec

# import mplhep as hep
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20.colors)

class mpl:
    def __init__(self, mc_samples, data=None):
        self.samples = mc_samples
        self.data = data
        self.kwarg={'histtype':'step','lw':2}
        self.colors = plt.cm.tab20.colors
        # sort the components to plot in order of fitted templates_project size
        self.sorted_order = ['bkg_FakeD',    'bkg_continuum',    'bkg_combinatorial',
                             'bkg_singleBbkg','bkg_TDFl',        r'$D\ell\nu$_gap',
                             r'$D^{\ast\ast}\ell\nu$',           r'$D^\ast\ell\nu$',
                             r'$D\ell\nu$',                 r'$D^{\ast\ast}\tau\nu$',
                             r'$D^\ast\tau\nu$',                 r'$D\tau\nu$',]
    
    def statistics(self, df=None, hist=None):
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
        
        return f'''{counts=:d} \n{mean=:.3f} \n{std=:.3f}'''

    def plot_data_1d(self, bins, ax, hist=None, sub_df=None, variable=None, cut=None, scale=1, name='Data'):
        # Provide either variable or hist
        if variable:
            data = sub_df if sub_df is not None else self.data
            if scale:
                data['__weight__'] = scale
            var_col= data.query(cut)[variable] if cut else data[variable]
            (counts, _) = np.histogram(var_col, bins=bins,
                                    weights=data.query(cut)['__weight__'] if cut else data['__weight__'])
            (staterr_squared, _) = np.histogram(var_col, bins=bins,
                                    weights=data.query(cut)['__weight__']**2 if cut else data['__weight__']**2)
            staterror = np.sqrt(staterr_squared)
            label=f'''{name} \n{self.statistics(df=var_col)} \n cut_eff={(len(var_col)/len(data)):.3f}'''
            data_counts = unp.uarray(counts, staterror)
        else:
            data_counts = hist
            label=f'''{name} \n{self.statistics(hist=[data_counts,bins])}'''

        bin_centers = (bins[:-1] + bins[1:]) /2
        data_val = unp.nominal_values(data_counts)
        data_err = unp.std_devs(data_counts)
        ax.errorbar(x=bin_centers, y=data_val, yerr=data_err, fmt='ko',label=label)

        return data_counts

    def plot_mc_1d(self, bins, ax, sub_df=None,sub_name=None, variable=None, cut=None, scale=1,correction=None,mask=[]):
        if correction:
            mc_combined = pd.concat(self.samples.values(), ignore_index=True)
            if scale:
                mc_combined['__weight__'] = scale
            var_col= mc_combined.query(cut)[variable] if cut else mc_combined[variable]
            (stacked_counts, _) = np.histogram(var_col, bins=bins,
                                    weights=mc_combined.query(cut)['__weight__'] if cut else mc_combined['__weight__'])
            ax.hist(bins[:-1], bins, weights=stacked_counts,histtype='step',color='black',
                    label=f'''Unweighted MC \n{self.statistics(df=var_col)} \n cut_eff={(len(var_col)/len(mc_combined)):.3f}''')
        
        if sub_df is not None:
            sample = sub_df
            if scale:
                sample['__weight__'] = scale
            var_col= sample.query(cut)[variable] if cut else sample[variable]
            (counts, _) = np.histogram(var_col, bins=bins,
                                      weights=sample.query(cut)['__weight__'] if cut else sample['__weight__'])
            (staterr_squared, _) = np.histogram(var_col, bins=bins,
                                    weights=sample.query(cut)['__weight__']**2 if cut else sample['__weight__']**2)
            staterror = np.sqrt(staterr_squared)
            ax.hist(bins[:-1], bins, weights=counts, **self.kwarg,
                    label=f'''{sub_name} \n{self.statistics(df=var_col)} \n cut_eff={(len(var_col)/len(sample)):.3f}''')
            sample_counts = unp.uarray(counts, staterror)
            bottom = sample_counts
        
        else:
            bottom = unp.uarray(np.zeros(len(bins)-1), np.zeros(len(bins)-1))
            for i,name in enumerate(self.sorted_order):
                sample = self.samples[name]
                sample_size = len(sample.query(cut)) if cut else len(sample)
                if sample_size==0 or name in mask:
                    continue
                if scale:
                    sample['__weight__'] = scale
                var_col= sample.query(cut)[variable] if cut else sample[variable]
                (counts, _) = np.histogram(var_col, bins=bins,
                                          weights=sample.query(cut)['__weight__'] if cut else sample['__weight__'])
                (staterr_squared, _) = np.histogram(var_col, bins=bins,
                                        weights=sample.query(cut)['__weight__']**2 if cut else sample['__weight__']**2)
                staterror = np.sqrt(staterr_squared)
                
                if correction:
                    (counts, _) = np.histogram(var_col, bins=bins,
                                    weights=scale*sample.query(cut)['PIDWeight'] if cut else scale*sample['PIDWeight'])
                    (staterr_squared, _) = np.histogram(var_col, bins=bins,
                            weights=(scale*sample.query(cut)['PIDWeight'])**2 if cut else (scale*sample['PIDWeight'])**2)
                    staterror = np.sqrt(staterr_squared)
    
                b = unp.nominal_values(bottom)
                ax.hist(bins[:-1], bins, weights=counts, bottom=b,color = self.colors[i],
                        label=f'''{name} \n{self.statistics(df=var_col)} \n cut_eff={(sample_size/len(sample)):.3f}''')
                
                sample_counts = unp.uarray(counts, staterror)
                bottom += sample_counts

        return bottom

    def plot_residuals(self, bins, data, model, ax):
        # Compute residuals (Data - Model) and their errors
        bin_centers = (bins[:-1] + bins[1:]) /2
        residuals = data - model
        res_val = unp.nominal_values(residuals)
        res_err = unp.std_devs(residuals)

        # Create a mask to exclude points where residual_errors are zero
        mask = res_err != 0
        # Compute chi-squared excluding those points
        chi2 = np.sum((res_val[mask] / res_err[mask]) ** 2)
        ndf = len(res_val[mask])

        # Plot the residuals in ax
        ax.errorbar(bin_centers, res_val, yerr=res_err, fmt='ok',
                   label=f'rechi2 = {chi2:.3f} / {ndf} = {chi2/ndf:.3f}')
        # Add a horizontal line at y=0 for reference
        ax.axhline(0, color='gray', linestyle='--')
        # Label the residual plot
        ax.set_ylabel('Residuals')
        
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
                             correction=False,mask=[],figsize=(8,5),
                             ratio=False):
        # Create a figure with two subplots: one for the histogram, one for the residual plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [5, 1]})
        # Data
        data_counts = self.plot_data_1d(bins=bins, variable=variable, ax=ax1, cut=cut, scale=scale[0])
            
        # MC
        mc_counts = self.plot_mc_1d(bins=bins, variable=variable, ax=ax1, cut=cut, 
                                    scale=scale[1],correction=correction,mask=mask)
        
        if ratio:
            self.plot_ratios(bins=bins, data=data_counts, model=mc_counts, ax=ax2)
        else:
            # Residuals (Data - Model)
            self.plot_residuals(bins=bins, data=data_counts, model=mc_counts, ax=ax2)
            ax2.legend(bbox_to_anchor=(1,1),fancybox=True, shadow=True)
        
        ax1.set_title(f'Overlaid Data vs MC ({cut=})')
        ax1.set_ylabel(f'# of events per bin {(bins[1]-bins[0]):.3f} GeV')
        ax1.grid()
        ax1.legend(bbox_to_anchor=(1,1),ncol=2, fancybox=True, shadow=True,labelspacing=1.5)
        ax2.set_xlabel(f'{variable}')
        
        
        # Adjust the layout to avoid overlapping of the subplots
        plt.tight_layout()
        # Show the plot
        plt.show()

        return data_counts, mc_counts
        
        
    def plot_mc_sig_control(self,variable,bins,bkg_name='bkg_FakeD',merge_sidebands=False,
                            cut=None,scale={},correction=False,mask=[],figsize=(8,5)):
        # Create a figure with two subplots: one for the histogram, one for the residual plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [5, 1]})
        
        if bkg_name=='bkg_FakeD':
            sample = self.samples[bkg_name]
            left = sample.query('D_M<1.83').copy()
            sig = sample.query('1.84<D_M<1.9').copy()
            right = sample.query('D_M>1.91').copy()
                
            regions = {'left sideband': left,
                       'signal region': sig,
                       'right sideband': right}
            for region, df in regions.items():
                df['__weight__'] = scale[region]
            
            if merge_sidebands:
                sides = pd.concat([left, right])
                regions = {'sidebands': sides,
                       'signal region': sig,}
            
            sb_total = 0 # total counts in sidebands used in residual calculation
            sig_total = 0
            for region, df in regions.items():
                if region=='signal region':
                    counts = self.plot_data_1d(bins=bins, sub_df=df, variable=variable, ax=ax1, 
                                            cut=cut, scale=None,name=region)
                    sig_total += counts
                    
                elif region in ['sidebands','left sideband', 'right sideband']:
                    counts = self.plot_mc_1d(bins=bins, sub_df=df, sub_name=region, variable=variable, 
                                    ax=ax1, cut=cut, scale=None,correction=correction,mask=mask)
                    sb_total += counts
            
            # Residuals (Data - Model) and their errors
            self.plot_residuals(bins=bins, data=sig_total, model=sb_total, ax=ax2)
            ax2.set_xlabel(f'{variable}')
                    
        ax1.set_title(f'Overlaid signal region vs control region ({bkg_name=})')
        ax1.set_ylabel(f'# of events per bin {(bins[1]-bins[0]):.3f} GeV')
        ax1.grid()
        ax1.legend(bbox_to_anchor=(1,1),ncol=2, fancybox=True, shadow=True,labelspacing=1.5)
        ax2.legend(bbox_to_anchor=(1,1),fancybox=True, shadow=True)
        
        # Adjust the layout to avoid overlapping of the subplots
        plt.tight_layout()
        plt.show()

    def plot_data_subtracted_and_mc(self,var_list,bin_list,cut=None,scale={},
                                    correction=False,mask=['bkg_FakeD'],figsize=(10,10)):
        # get data in sig and sidebands regions
        data_left = self.data.query('D_M<1.83').copy()
        data_sig = self.data.query('1.84<D_M<1.9').copy()
        data_right = self.data.query('D_M>1.91').copy()
        # get mc in sig region without fake D
        mc_sig = pd.concat([df.query('1.84<D_M<1.9').copy() for name, df in self.samples.items() if name != 'bkg_FakeD'], 
                           ignore_index=True)

        data_mc_regions = {'data left sideband': data_left,
                           'data signal region': data_sig,
                           'data right sideband': data_right,
                           'mc signal region': mc_sig}

        for region, df in data_mc_regions.items():
                df['__weight__'] = scale[region]
            
        # calculate the 2d hists
        data_sb_2d = 0 
        data_sig_2d = 0
        mc_sig_2d = 0
        variable_x, variable_y = var_list
        edges_x, edges_y = bin_list
        var_x_label = '$M_{miss}^2$    [$GeV^2/c^4$]'
        var_y_label = '$|p_D| + |p_{\ell}|$    [GeV/c]'
        
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
            counts_uncert_2d = unp.uarray(counts_2d.round(3), staterr_2d.round(3))
            
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
                         cmap='rainbow', norm=colors.LogNorm(),
                         extent=[edges_x[0], edges_x[-1], edges_y[0], edges_y[-1]])
        fig.colorbar(im, ax=ax1)
        ax1.set_title('Data, background subtracted')
        ax1.set_xlabel(var_x_label)
        ax1.set_ylabel(var_y_label)
        ax1.grid()
    
        
        # Top-right: 1D histogram of p_D_l projection + residuals
        data_y = self.plot_data_1d(bins=edges_y, hist=data_subtracted_y, ax=ax2,cut=cut, name='Data')
        mc_y = self.plot_mc_1d(bins=edges_y, variable=variable_y, ax=ax2, scale=scale['mc signal region'],
                               cut='1.84<D_M<1.9', correction=correction,mask=mask)
        ax2.set_title('$|p_D| + |p_{\ell}|$ Projection')
        ax2.grid()
        ax2.legend(ncol=1, framealpha=0, shadow=False,labelspacing=1.5,fontsize=8)
        # Residual plot below
        self.plot_residuals(bins=edges_y, data=data_y, model=mc_y, ax=ax3)
        ax3.set_xlabel(var_y_label)
        ax3.legend(bbox_to_anchor=(0.8,-1.2),ncol=2, framealpha=0, shadow=False,labelspacing=1.5)
        
        
        # Bottom-left: 1D histogram of mm2 projection + residuals
        data_x = self.plot_data_1d(bins=edges_x, hist=data_subtracted_x, ax=ax4,cut=cut, name='Data')
        mc_x = self.plot_mc_1d(bins=edges_x, variable=variable_x, ax=ax4, scale=scale['mc signal region'],
                               cut='1.84<D_M<1.9', correction=correction,mask=mask)
        ax4.set_title('$M_{miss}^2$ Projection')
        ax4.grid()
        ax4.legend(ncol=1, framealpha=0, shadow=False,labelspacing=1.5,fontsize=8)
        # Residual plot below
        self.plot_residuals(bins=edges_x, data=data_x, model=mc_x, ax=ax5)
        ax5.set_xlabel(var_x_label)
        ax5.legend(bbox_to_anchor=(0.8,12.5),ncol=2, framealpha=0, shadow=False,labelspacing=1.5)
        
        
        # Bottom-right: 2D histogram of MC (B0_CMS3_weMissM2 vs p_D_l)
        im = ax6.imshow(unp.nominal_values(mc_sig_2d).T, origin='lower', aspect='auto', 
                         cmap='rainbow', norm=colors.LogNorm(),
                         extent=[edges_x[0], edges_x[-1], edges_y[0], edges_y[-1]])
        fig.colorbar(im, ax=ax6)
        ax6.set_title('MC, background removed')
        ax6.set_xlabel(var_x_label)
        ax6.set_ylabel(var_y_label)
        ax6.grid()
        
        # Adjust layout to avoid overlap
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.4, wspace=0.4)
        plt.show()


    def plot_data_mc_all(self, bins, variables, cut=None, scale=[1,1], figsize=(30, 100), fontsize=12):
        fig = plt.figure(figsize=figsize)
        dfs = [self.data, self.mc_samples]
        names = ['Data', 'MC']

#         for i in range(len(variables)):
#             ax = fig.add_subplot(len(variables)//3 + 1, 3, i+1)
#             bin1 = bins
#             for j in range(len(names)):
#                 dfs[j]['__weight__'] = scale[j]
#                 var_col = dfs[j][variables[i]]
#                 if cut is not None:
#                     var_col = var_col.query(cut)
#                 (counts, bin1) = np.histogram(var_col, bins=bin1,
#                                            weights=dfs[j].query(cut)['__weight__'] if cut else dfs[j]['__weight__'])
#                 kwarg={'histtype':'step','lw':2}

#                 if names[j]=='MC':
#                     ax.hist(bin1[:-1], bin1, weights=counts,
#                         label=f'{names[j]} \n{statistics(var_col)} \n cut_eff={(len(var_col)/len(dfs[j])):.3f}',**kwarg)

#                 elif names[j]=='Data':
#                     bin_centers = (bin1[:-1] + bin1[1:]) /2
#                     ax.errorbar(x=bin_centers, y=counts, yerr=np.sqrt(counts), fmt='ko',
#                                 label=f'{names[j]} \n{statistics(var_col)} \n cut_eff={(len(var_col)/len(dfs[j])):.3f}')

#             ax.set_ylabel(f'# of events per bin {(bin1[1]-bin1[0]):.3f} GeV',fontsize=fontsize)
#             ax.set_xlabel(variables[i],fontsize=fontsize)
#             ax.grid()
#     #         ax.legend()
#         fig.suptitle(f'Overlaid Data vs MC ({cut=})', fontsize=fontsize*2)
#         plt.tight_layout()
    
    
    
    
    
        
        
    def plot_all_mc_overlaid(self,variable,bins,cut=None,mask=[],density=False):
            
        fig,axs =plt.subplots(sharex=True, sharey=False,figsize=(8, 6))
        for name, sample in self.samples.items():
            sample_size = len(sample.query(cut)) if cut else len(sample)
            if sample_size==0 or name in mask:
                continue
            var_col= sample.query(cut)[variable] if cut else sample[variable]
            (counts, _) = np.histogram(var_col, bins=bins)

            kwarg=self.kwarg

            axs.hist(bins[:-1], bins, weights=counts, density=density,
                    label=f'''{name} \n{self.statistics(var_col)} \n cut_eff={(sample_size/len(sample)):.3f}''',**kwarg)

        axs.set_title(f'Overlaid components ({cut=})', fontsize=14)
        axs.set_xlabel(f'{variable}', fontsize=14)
        axs.set_ylabel(f'# of events per bin {(bins[1]-bins[0]):.3f} GeV', fontsize=14)
        axs.grid()
        plt.legend(bbox_to_anchor=(1,1),ncol=3, fancybox=True, shadow=True,labelspacing=1.5)
        

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
        for i, (name, sample) in enumerate(self.samples.items()):
            sample_size = len(sample.query(cut)) if cut else len(sample)
            if sample_size==0:
                continue
            ax = fig.add_subplot(5,3,i+1)
            (counts, xe, ye) = np.histogram2d(
                            sample.query(cut)[variable_x] if cut else sample[variable_x], 
                            sample.query(cut)[variable_y] if cut else sample[variable_y],
                            bins=[xedges, yedges])

            im = ax.imshow(counts.T, origin='lower', aspect='auto', 
                     cmap='rainbow', norm=colors.LogNorm(),alpha=mask_arr,
                     extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
            fig.colorbar(im, ax=ax)
#             X, Y = np.meshgrid(xedges, yedges)
#             im=ax.pcolormesh(X, Y, counts, cmap='rainbow', norm=colors.LogNorm(), alpha=mask_arr)
            ax.grid()
            ax.set_xlim(xedges.min(),xedges.max())
            ax.set_ylim(yedges.min(),yedges.max())
            ax.set_title(name,fontsize=14)

        fig.suptitle(f'Signal MC ({cut=})', y=0.95, fontsize=18)
        fig.supylabel(r'$|p^\ast_{D}|+|p^\ast_{\ell}| \ \ [GeV]$', x=0.05,fontsize=18)
        fig.supxlabel('$M_{miss}^2\ \ \ [GeV^2/c^4]$', y=0.05,fontsize=18)
        
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
                             cmap='rainbow', norm=colors.LogNorm(),
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
                      bins=30,cmap='rainbow', norm=colors.LogNorm())
            ax.set_ylabel(target,fontsize=30)
            ax.set_xlabel(variables[i],fontsize=30)
            ax.grid()

    def plot_FOM(self, sigModes, bkgModes, variable, test_points, cut=None):
        sig = pd.concat([self.samples[i] for i in sigModes])
        bkg = pd.concat([self.samples[i] for i in bkgModes])
        sig_tot = len(sig)
        bkg_tot = len(bkg)
        BDT_FOM = []
        BDT_FOM_err = []
        BDT_sigEff = []
        BDT_sigEff_err = []
        BDT_bkgEff = []
        BDT_bkgEff_err = []
        for i in test_points:
            nsig = len(sig.query(f"{cut} and {variable}>{i}" if cut else f"{variable}>{i}"))
            nbkg = len(bkg.query(f"{cut} and {variable}>{i}" if cut else f"{variable}>{i}"))
            tot = nsig+nbkg
            tot_err = np.sqrt(tot)
            FOM = nsig / tot_err # s / √(s+b)
            FOM_err = np.sqrt( (tot_err - FOM/2)**2 /tot**2 * nsig + nbkg**3/(4*tot**3) + 9*nbkg**2*np.sqrt(nsig*nbkg)/(4*tot**5) )

            BDT_FOM.append(FOM)
            BDT_FOM_err.append(FOM_err)

            sigEff = nsig / sig_tot
            sigEff_err = sigEff * np.sqrt(1/nsig + 1/sig_tot)
            bkgEff = nbkg / bkg_tot
            bkgEff_err = bkgEff * np.sqrt(1/nbkg + 1/bkg_tot)
            BDT_sigEff.append(sigEff)
            BDT_sigEff_err.append(sigEff_err)
            BDT_bkgEff.append(bkgEff)
            BDT_bkgEff_err.append(bkgEff_err)

        fig, ax1 = plt.subplots(figsize=(8, 6))
        

        color = 'tab:red'
        ax1.set_ylabel('Efficiency', color=color)  # we already handled the x-label with ax1
        ax1.errorbar(x=test_points, y=BDT_sigEff, yerr=BDT_sigEff_err,marker='o',label='Signal Efficiency',color=color)
        ax1.errorbar(x=test_points, y=BDT_bkgEff, yerr=BDT_bkgEff_err,marker='o',label='Bkg Efficiency',color='green')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')
        ax1.grid()
        ax1.set_xlabel('Signal Probability')
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
        color = 'tab:blue'
        ax2.set_ylabel('FOM', color=color)
        ax2.errorbar(x=test_points, y=BDT_FOM, yerr=BDT_FOM_err,marker='o',label='FOM',color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        #ax2.grid()
        ax2.legend(loc='upper right')

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title(f'FOM for {variable=}')
        plt.xlim(0,1)
        plt.ylim(bottom=0)
        plt.show()
        
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
    
            


# +
def fit_project_cabinetry(fit_result, templates_2d,staterror_2d,data_2d, 
                                      edges_list, direction='mm2', slice_thresholds=None):
    assert direction in ['mm2', 'p_D_l'], 'direction must be mm2 or p_D_l'

    def plot(bins, fitted_1d, data_1d, ax1, ax2, ax3, legend=True):        

        bin_width = np.diff(bins)
        bin_centers = (bins[:-1] + bins[1:]) /2
        # plot the templates with defined colors
        c = plt.cm.tab20.colors
        # sort the components to plot in order of fitted templates_project size
        sorted_order = ['bkg_FakeD',    'bkg_continuum',    'bkg_combinatorial',
                        'bkg_singleBbkg','bkg_TDFl',        r'$D\ell\nu$_gap',
                        r'$D^{\ast\ast}\ell\nu$',      r'$D^{\ast\ast}\tau\nu$',
                        r'$D^\ast\ell\nu$',                 r'$D\ell\nu$',
                        r'$D^\ast\tau\nu$',                 r'$D\tau\nu$',]
        
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



# # plotting version: two residual plots, residual_signal = data - all_temp
# def mpl_projection_residual_iMinuit(Minuit, templates_2d, data_2d, edges, slices=[1.6,1],direction='mm2', plot_with='pltbar'):
#     if direction not in ['mm2', 'p_D_l'] or plot_with not in ['mplhep', 'pltbar']:
#         raise ValueError('direction in [mm2, p_D_l] and plot_with in [mplhep, pltbar]')
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

#     def extend(x):
#         return np.append(x, x[-1])

#     def errorband(bins, template_sum, template_err, ax):
#         fitted_sum = np.sum(template_sum, axis=0)
#         fitted_err = np.sqrt(np.sum(np.array(template_err)**2, axis=0)) # assuming the correlations between each template are 0
#         ax.fill_between(bins, extend(fitted_sum - fitted_err), extend(fitted_sum + fitted_err),
#         step="post", color="black", alpha=0.3, linewidth=0, zorder=100,)   

#     def plot_with_hep(bins, templates_project, templates_project_err, data, signal_name, ax1, ax2, ax3):
#         data_project = data.sum(axis=axis_to_be_summed_over)
#         # plot the templates and data
#         hep.histplot(templates_project, bin_edges, stack=True, histtype='fill', sort='yield_r', label=fitted_components_names, ax=ax1)
#         # errorband(bin_edges, templates_project, templates_project_err, ax1)
#         hep.histplot(data_project, bin_edges, histtype='errorbar', color='black', w2=data_project, ax=ax1)
#         # plot the residual
#         signal_index = fitted_components_names.index(signal_name)
#         residual = data_project - np.sum(templates_project, axis=0)
#         residual_signal = residual + templates_project[signal_index]
#         # Error assuming the correlations between data and templates, between each template, are 0
#         residual_err = np.sqrt(data_project + np.sum(np.array(templates_project_err)**2, axis=0))
#         residual_err_signal = np.sqrt(residual_err**2 - np.array(templates_project_err[signal_index]))

#         pull = [0 if residual_err[i]==0 else (residual[i]/residual_err[i]) for i in range(len(residual))]
#         pull_signal = [0 if residual_err_signal[i]==0 else (residual_signal[i]/residual_err_signal[i]) for i in range(len(residual_signal))]
#         #hep.histplot(residual, bin_edges, histtype='errorbar', color='black', yerr=residual_err, ax=ax2)
#         hep.histplot(residual, bin_edges, histtype='errorbar', color='black', ax=ax2)
#         ax2.axhline(y=0, linestyle='-', linewidth=1, color='r')
#         #hep.histplot(residual_signal, bin_edges, histtype='errorbar', color='black', yerr=residual_err_signal, ax=ax3)
#         hep.histplot(pull, bin_edges, histtype='errorbar', color='black', ax=ax3)
#         ax3.axhline(y=0, linestyle='-', linewidth=1, color='r')

#         ax1.grid()
#         ax1.set_ylabel('# of counts per bin',fontsize=16)
#         ax1.set_xlim(bin_edges.min(), bin_edges.max())
#         ax1.set_ylim(0, data_project.max()*1.2)
#         ax2.set_ylabel('pull',fontsize=14)
#         ax2.set_xlim(bin_edges.min(), bin_edges.max())
#         ax3.set_ylabel('pull + signal',fontsize=10)
#         ax3.set_xlim(bin_edges.min(), bin_edges.max())
#         ax1.legend(bbox_to_anchor=(1,1),ncol=1, fancybox=True, shadow=True,labelspacing=1)

#     def plot_with_bar(bins, templates_project, templates_project_err, data, ax1, ax2, ax3,signal_name=None):        
#         # calculate the arguments for plotting
#         bin_width = bins[1]-bins[0]
#         bin_centers = (bins[:-1] + bins[1:]) /2
#         data_project = data.sum(axis=axis_to_be_summed_over)
#         data_err = np.sqrt(data_project)

#         # plot the templates with defined colors
#         c = plt.cm.tab20.colors
#         # sort the components to plot in order of fitted templates_project size
#         sorted_indices = sorted(range(len(templates_2d)), key=lambda i: np.sum(templates_project[i]), reverse = True)
#         bottom_hist = np.zeros(data.shape[1-axis_to_be_summed_over])
#         for i in sorted_indices:
#             binned_counts = templates_project[i]
#             ax1.bar(x=bins[:-1], height=binned_counts, bottom=bottom_hist, color = c[i],
#                     width=bin_width, align='edge', label=fitted_components_names[i])
#             bottom_hist = bottom_hist + binned_counts
#         # errorband(bin_edges, templates_project, templates_project_err, ax1)

#         # plot the data
#         ax1.errorbar(x=bin_centers, y=data_project, yerr=data_err, fmt='ko')
#         # plot the residual
#         residual = data_project - np.sum(templates_project, axis=0)
#         # Error assuming the correlations between data and templates, between each template, are 0
#         residual_err = np.sqrt(data_project + np.sum(np.array(templates_project_err)**2, axis=0))
                        
#         pull = [0 if residual_err[i]==0 else (residual[i]/residual_err[i]) for i in range(len(residual))]
#         ax2.errorbar(x=bin_centers, y=residual, yerr=residual_err, fmt='ko')
#         ax2.axhline(y=0, linestyle='-', linewidth=1, color='r')
#         ax3.scatter(x=bin_centers, y=pull, c='black')
#         ax3.axhline(y=0, linestyle='-', linewidth=1, color='r')            

#         ax1.grid()
#         ax1.set_ylabel('# of counts per bin',fontsize=16)
#         ax1.set_xlim(bin_edges.min(), bin_edges.max())
#         ax1.set_ylim(0, data_project.max()*1.2)
#         ax2.set_ylabel('residual',fontsize=14)
#         ax2.set_xlim(bin_edges.min(), bin_edges.max())
#         ax3.set_ylabel('pull',fontsize=14)
#         ax3.set_xlim(bin_edges.min(), bin_edges.max())
#         ax1.legend(bbox_to_anchor=(1,1),ncol=1, fancybox=True, shadow=True,labelspacing=1)
        
# #         signal_index = fitted_components_names.index(signal_name)
# #         residual_signal = residual + templates_project[signal_index]
# #         residual_err_signal = np.sqrt(residual_err**2 - np.array(templates_project_err[signal_index]))
# #         pull_signal = [0 if residual_err_signal[i]==0 else (residual_signal[i]/residual_err_signal[i]) for i in range(len(residual_signal))]

#     if direction=='mm2':
#         direction_label = '$M_{miss}^2$'
#         direction_unit = '$[GeV^2/c^4]$'
#         other_direction_label = '$|p_D|\ +\ |p_l|$'
#         other_direction_unit = '[GeV]'
#         axis_to_be_summed_over = 0

#         bin_edges = edges[axis_to_be_summed_over] #xedges
#         slice_position = slices[1-axis_to_be_summed_over] #p_D_l
#         slice_index, = np.asarray(np.isclose(edges[1-axis_to_be_summed_over],slice_position,atol=0.2)).nonzero()
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
#         slice_index, = np.asarray(np.isclose(edges[1-axis_to_be_summed_over],slice_position,atol=0.2)).nonzero()
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
#         raise ValueError('Current version only supports projection to either MM2 or p_D_l')


#     if not slices:
#         fig = plt.figure(figsize=(6.4,6.4))
#         gs = gridspec.GridSpec(3,1, height_ratios=[0.7,0.15,0.15])
#         ax1 = fig.add_subplot(gs[0])
#         ax2 = fig.add_subplot(gs[1])
#         ax3 = fig.add_subplot(gs[2])
#         gs.update(hspace=0.3) 
#         fitted_project = [temp.sum(axis=axis_to_be_summed_over) for temp in fitted_templates]
#         fitted_project_err = [temp.sum(axis=axis_to_be_summed_over) for temp in fitted_templates_err]

#         # plot the templates and data and templates_err
#         if plot_with=='mplhep':
#             plot_with_hep(bin_edges, fitted_project, fitted_project_err, counts, '$D\\tau\\nu$', ax1, ax2,ax3)
#         elif plot_with=='pltbar':
#             plot_with_bar(bin_edges, fitted_project, fitted_project_err, counts, '$D\\tau\\nu$', ax1, ax2,ax3)
#         ax1.set_title(f'Fitting projection to {direction_label}')
#         ax3.set_xlabel(direction_label)

#     elif slices:
#         fig = plt.figure(figsize=(16,9))
#         spec = gridspec.GridSpec(6,2, figure=fig, wspace=0.4, hspace=0.5)
#         ax1 = fig.add_subplot(spec[:-2, 0])
#         ax2 = fig.add_subplot(spec[:-2, 1])
#         ax3 = fig.add_subplot(spec[-2, 0])
#         ax4 = fig.add_subplot(spec[-2, 1])
#         ax5 = fig.add_subplot(spec[-1, 0])
#         ax6 = fig.add_subplot(spec[-1, 1])
#         #gs.update(hspace=0) 

#         # plot the templates and data and template_err
#         if plot_with=='mplhep':
#             plot_with_hep(bin_edges, fitted_project_slice1, fitted_project_slice1_err, data_slice1, slice1_signal, ax1, ax3, ax5)
#             plot_with_hep(bin_edges, fitted_project_slice2, fitted_project_slice2_err, data_slice2, slice2_signal, ax2, ax4, ax6)
#         elif plot_with=='pltbar':
#             plot_with_bar(bin_edges, fitted_project_slice1, fitted_project_slice1_err, data_slice1, ax1, ax3, ax5)
#             plot_with_bar(bin_edges, fitted_project_slice2, fitted_project_slice2_err, data_slice2, ax2, ax4, ax6)

#         ax1.set_title(f'{other_direction_label} < {slice_position}  {other_direction_unit}',fontsize=14)
#         ax2.set_title(f'{other_direction_label} > {slice_position}  {other_direction_unit}',fontsize=14)
#         fig.suptitle(f'Fitted projection to {direction_label} in slices of {other_direction_label}',fontsize=16)
#         fig.supxlabel(direction_label + '  ' + direction_unit,fontsize=16)

# +
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

class ply:
    def __init__(self, df):
        self.df = df
        
    def hist(self, variable='B0_CMS3_weMissM2', cut=None, facet=False):
        # Create a histogram
        fig=px.histogram(self.df.query(cut) if cut else self.df, 
                         x=variable, color='mode', nbins=60, 
                         marginal='box', #opacity=0.5, barmode='overlay',
                         color_discrete_sequence=px.colors.qualitative.Plotly,
                         template='simple_white', title='Signal MC',
                         facet_col='p_D_l_region' if facet else None)

        # Manage the layout
        fig.update_layout(font_family='Rockwell', hovermode='closest',
                          legend=dict(orientation='h',title='',x=1,y=1,xanchor='right',yanchor='bottom'))

        # Manage the hover labels
        count_by_color = self.df.groupby('mode')['__event__'].count()
        for i, (color, count) in enumerate(count_by_color.items()):
            fig.update_traces(hovertemplate='Bin_Count: %{y}<br>Overall_Count: '+str(count),selector={'name':color})
            #fig.add_annotation(x=1+i*3, y=8000, text=f'Total Count ({color}): {count}', showarrow=True)

        # Update axes labels
        if variable=='B0_CMS3_weMissM2':
            fig.update_xaxes(title_text="$M_{miss}^2\ \ [GeV^2/c^4]$", row=1)

        # Show the plot
        fig.show()
        
    def hist2d(self, cut=None, facet=False):
        # Define number of colors to generate
        color_sequence = ['rgb(255,255,255)'] + px.colors.sequential.Rainbow[1:]
        num_colors = 9
        # Generate colors with uniform spacing and Rainbow color scale
        my_colors = [[i/(num_colors-1), color_sequence[i]] for i in range(num_colors)]

        # Create a 2d histogram
        fig = px.density_heatmap(self.df.query(cut) if cut else self.df, 
                                 x="B0_CMS3_weMissM2", y="p_D_l",
                                 marginal_x='histogram', marginal_y='histogram',
                                 nbinsx=40,nbinsy=40,color_continuous_scale=my_colors,
                                 template='simple_white', title='Signal MC',
                                 facet_col='mode' if facet else None,
                                 facet_col_wrap=3 if facet else None,)

        # Update axes labels
        fig.update_xaxes(title_text="$M_{miss}^2\ \ [GeV^2/c^4]$", row=1)
        fig.update_yaxes(title_text="$|p_D|+|p_l|\ \ [GeV/c]$",row=1, col=1)

        fig.show()
        
    def plot_FOM(self, sigModes, bkgModes, variable, test_points,cut=None):
        # calculate the FOM, efficiencies
        sig = self.df.loc[self.df['mode'].isin(sigModes)]
        bkg = self.df.loc[self.df['mode'].isin(bkgModes)]
        sig_tot = len(sig)
        bkg_tot = len(bkg)
        BDT_FOM = []
        BDT_FOM_err = []
        BDT_sigEff = []
        BDT_sigEff_err = []
        BDT_bkgEff = []
        BDT_bkgEff_err = []
        for i in test_points:
            nsig = len(sig.query(f"{cut} and {variable}>{i}" if cut else f"{variable}>{i}"))
            nbkg = len(bkg.query(f"{cut} and {variable}>{i}" if cut else f"{variable}>{i}"))
            tot = nsig+nbkg
            tot_err = np.sqrt(tot)
            FOM = nsig / tot_err # s / √(s+b)
            FOM_err = np.sqrt( (tot_err - FOM/2)**2 /tot**2 * nsig + nbkg**3/(4*tot**3) + 9*nbkg**2*np.sqrt(nsig*nbkg)/(4*tot**5) )

            BDT_FOM.append(FOM)
            BDT_FOM_err.append(FOM_err)

            sigEff = nsig / sig_tot
            sigEff_err = sigEff * np.sqrt(1/nsig + 1/sig_tot)
            bkgEff = nbkg / bkg_tot
            bkgEff_err = bkgEff * np.sqrt(1/nbkg + 1/bkg_tot)
            BDT_sigEff.append(sigEff)
            BDT_sigEff_err.append(sigEff_err)
            BDT_bkgEff.append(bkgEff)
            BDT_bkgEff_err.append(bkgEff_err)
        

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig.add_trace(
            go.Scatter(x=test_points, y=BDT_FOM, name="FOM",
                       error_y=dict(type='data',array=BDT_FOM_err,visible=True)),
            secondary_y=True,
        )

        fig.add_trace(
            go.Scatter(x=test_points, y=BDT_sigEff, name="sig_eff",
                       error_y=dict(type='data',array=BDT_sigEff_err,visible=True)),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=test_points, y=BDT_bkgEff, name="bkg_eff",
                       error_y=dict(type='data',array=BDT_bkgEff_err,visible=True)),
            secondary_y=False,
        )

        # Add figure title
        fig.update_layout(
            title_text="MVA Performance",
            template='simple_white',
            hovermode='x',
            legend=dict(orientation='h',title='',x=1,y=1.1,xanchor='right',yanchor='bottom')
        )

        # Set x-axis title
        fig.update_xaxes(title_text=variable)

        # Set y-axes titles
        fig.update_yaxes(title_text="<b>FOM</b>", secondary_y=True)
        fig.update_yaxes(title_text="Efficiency", secondary_y=False)

        fig.show()
        
    def plot_cut_efficiency(self, cut, variable='B0_CMS3_weQ2lnuSimple',bins=15):
        # Create figure with secondary y-axis
        fig = make_subplots()
        
        for mode in self.df['mode'].unique():
            if mode in ['bkg_continuum','bkg_fakeDTC','bkg_fakeB','bkg_others']:
                continue
            comp=self.df.loc[self.df['mode']==mode]
            (bc, bins1) = np.histogram(comp[variable], bins=bins)
            (ac, bins1) = np.histogram(comp.query(cut)[variable], bins=bins1)
            bc+=1
            ac+=1
            efficiency = ac / bc
            factor = [i if i<1 else 0 for i in 1/ac + 1/bc] # mannually set the uncertainty to 0 if bin count==0
            efficiency_err = efficiency * np.sqrt(factor)
            bin_centers = (bins1[:-1] + bins1[1:]) /2
            
            # Add traces
            fig.add_trace(
                go.Scatter(x=bin_centers, y=efficiency, name=mode,
                           error_y=dict(type='data',array=efficiency_err,visible=True))
            )
        
        
        # Add figure title
        fig.update_layout(
            title_text=f'Efficiency for {cut=}',
            template='simple_white',
            hovermode='closest',
            legend=dict(orientation='h',title='',x=1,y=1,xanchor='right',yanchor='bottom')
        )

        # Set x-axis title
        fig.update_xaxes(title_text=variable)

        # Set y-axes titles
        fig.update_yaxes(title_text="<b>Efficiency</b>")

        fig.show()
        
    
# plotting version: residual = data - all_temp
def ply_projection_residual(Minuit, templates_2d, data_2d, edges, slices=[1.6,1],direction='mm2'):
    if direction not in ['mm2', 'p_D_l']:
        raise ValueError('direction in [mm2, p_D_l]')
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

    if direction=='mm2':
        direction_label = '$M_{miss}^2$'
        direction_unit = '$[GeV^2/c^4]$'
        other_direction_label = '$|p_D|\ +\ |p_l|$'
        other_direction_unit = '[GeV]'
        axis_to_be_summed_over = 0

        bin_edges = edges[axis_to_be_summed_over] #xedges
        slice_position = slices[1-axis_to_be_summed_over] #p_D_l
        slice_index, = np.asarray(np.isclose(edges[1-axis_to_be_summed_over],slice_position,atol=0.1)).nonzero()
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
        slice_index, = np.asarray(np.isclose(edges[1-axis_to_be_summed_over],slice_position,atol=0.1)).nonzero()
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
        raise ValueError('Current version only supports projection to either mm2 or p_D_l')

        
    def plot(bins, templates_project, templates_project_err, data, column):        
        # calculate the arguments for plotting
        bin_width = bins[1]-bins[0]
        bin_centers = (bins[:-1] + bins[1:]) /2
        data_project = data.sum(axis=axis_to_be_summed_over)
        data_err = np.sqrt(data_project)

        # sort the components to plot in order of fitted templates_project size
        c = px.colors.qualitative.Light24
        sorted_indices = sorted(range(len(templates_2d)), key=lambda i: np.sum(templates_project[i]), reverse = True)
        bottom_hist = np.zeros(data.shape[1-axis_to_be_summed_over])
        for i in sorted_indices:
            binned_counts = templates_project[i]
            fig.add_trace(go.Bar(x=bins[:-1], y=binned_counts, width=bin_width,
                                 alignmentgroup=1, name=fitted_components_names[i],
                                 legendgroup=fitted_components_names[i],
                                 marker=dict(color=c[i]),
                                 showlegend=True if column==1 else False), 
                          row=1, col=column)

        # plot the data
        fig.add_trace(go.Scatter(x=bin_centers, y=data_project, name='data',mode='markers',
                                error_y=dict(type='data',array=data_err,visible=True),
                                legendgroup='data',marker=dict(color=c[11]),
                                showlegend=True if column==1 else False),
                      row=1, col=column)

        # plot the residual
        residual = data_project - np.sum(templates_project, axis=0)
        # Error assuming the correlations between data and templates, between each template, are 0
        residual_err = np.sqrt(data_project + np.sum(np.array(templates_project_err)**2, axis=0))
                        
        pull = [0 if residual_err[i]==0 else (residual[i]/residual_err[i]) for i in range(len(residual))]
        fig.add_trace(go.Scatter(x=bin_centers, y=residual,name='residual',mode='markers',
                        error_y=dict(type='data',array=residual_err,visible=True),
                                legendgroup='residual',marker=dict(color=c[12]),
                                showlegend=True if column==1 else False),
              row=2, col=column)
        fig.add_trace(go.Scatter(x=bin_centers, y=pull,name='pull',mode='markers',
                                legendgroup='pull',marker=dict(color=c[13]),
                                showlegend=True if column==1 else False),
              row=3, col=column)
        

    # create subplots
    fig = make_subplots(rows=3, cols=2, row_heights=[0.7, 0.15,0.15],vertical_spacing=0.05,
                    subplot_titles=(f'{other_direction_label} < {slice_position}  {other_direction_unit}',
                                    f'{other_direction_label} > {slice_position}  {other_direction_unit}',
                                    '','','',''))

    # plot the templates and data and template_err
    plot(bin_edges, fitted_project_slice1, fitted_project_slice1_err, data_slice1, column=1)
    plot(bin_edges, fitted_project_slice2, fitted_project_slice2_err, data_slice2, column=2)
    
    # Set x/y-axis title
    fig.update_xaxes(title_text=direction_label + direction_unit,row=3)
    fig.update_yaxes(title_text='# of counts per bin', row=1, col=1)
    fig.update_yaxes(title_text='residual', row=2, col=1)
    fig.update_yaxes(title_text='pull', row=3, col=1)

    # Add figure title
    fig.update_layout(
        width=850,height=650,
        title_text=f'Fitted projection to {direction_label} in slices of {other_direction_label}',
        template='simple_white',
        hovermode='closest',
        barmode='stack',
        legend=dict(orientation='h',title='',x=1,y=1.1,xanchor='right',yanchor='bottom'),
        shapes=[dict(type='line', y0=0, y1=0, xref='paper', 
                     x0=bin_edges.min(), x1=bin_edges.max())],
    )

    fig.show()

# +
# from enum import Enum

# # define DecayModes from DecayHash
# class DecayMode(Enum):
#     bkg = 0
#     sig_D_tau_nu = 1
#     sig_D_l_nu = 2
#     sig_Dst_tau_nu = 3
#     sig_Dst_l_nu = 4
#     Dstst_tau_nu_mixed = 5
#     Dstst_tau_nu_charged = 6
#     res_Dstst_l_nu_mixed = 7
#     nonres_Dstst_l_nu_mixed = 8
#     gap_Dstst_l_nu_mixed = 9
#     res_Dstst_l_nu_charged = 10
#     nonres_Dstst_l_nu_charged = 11

# DecayMode = Enum('DecayMode', ['bkg_fakeD',           'bkg_continuum',    'bkg_combinatorial',
#                                'bkg_Odecay',          'bkg_fakeTC',       r'$D\tau\nu$',
#                                r'$D^\ast\tau\nu$',    r'$D\ell\nu$',      r'$D^\ast\ell\nu$',
#                                r'$D^{\ast\ast}\tau\nu$_mixed',            r'$D^{\ast\ast}\tau\nu$_charged',
#                                r'res_$D^{\ast\ast}\ell\nu$_mixed',        r'nonres_$D^{\ast\ast}\ell\nu$_mixed',
#                                r'gap_$D^{\ast\ast}\ell\nu$_mixed',        r'res_$D^{\ast\ast}\ell\nu$_charged',
#                                r'nonres_$D^{\ast\ast}\ell\nu$_charged'],
#                  start=0)

# DecayMode(0).name

#     if not new:
#         # Sig components
#         sig_D_tau_nu=df.query(f'DecayMode=={DecayMode["sig_D_tau_nu"].value} and \
#         {true_B0} and {true_D_tau} and {DecayErrors["D_tau_errors"]}').copy()

#         sig_D_l_nu=df.query(f'DecayMode=={DecayMode[f"sig_D_l_nu"].value} and \
#         {true_B0} and {true_D_ell} and {DecayErrors["D_l_errors"]}').copy()

#         sig_Dst_tau_nu=df.query(f'DecayMode=={DecayMode["sig_Dst_tau_nu"].value} and \
#         {true_B0} and {true_D_tau} and {DecayErrors["Dst_Dstst_mixed_errors"]}').copy()

#         sig_Dst_l_nu=df.query(f'DecayMode=={DecayMode[f"sig_Dst_l_nu"].value} and \
#         {true_B0} and {true_D_ell} and {DecayErrors["Dst_l_errors"]}').copy()

#         Dstst_tau_nu_mixed=df.query(f'DecayMode=={DecayMode["Dstst_tau_nu_mixed"].value} and \
#         {true_B0} and {true_D_tau} and {DecayErrors["Dst_Dstst_mixed_errors"]}').copy()

#         res_Dstst_l_nu_mixed=df.query(f'DecayMode=={DecayMode[f"res_Dstst_l_nu_mixed"].value} and \
#         {true_B0} and {true_D_ell} and {DecayErrors["Dst_Dstst_mixed_errors"]}').copy()

#         nonres_Dstst_l_nu_mixed=df.query(f'DecayMode=={DecayMode[f"nonres_Dstst_l_nu_mixed"].value} and \
#         {true_B0} and {true_D_ell} and {DecayErrors["Dst_Dstst_mixed_errors"]}').copy()

#         gap_Dstst_l_nu_mixed=df.query(f'DecayMode=={DecayMode[f"gap_Dstst_l_nu_mixed"].value} and \
#         {true_B0} and {true_D_ell} and {DecayErrors["Dst_Dstst_mixed_errors"]}').copy()

#         Dstst_tau_nu_charged=df.query(f'DecayMode=={DecayMode["Dstst_tau_nu_charged"].value} and \
#         {B_charged} and {true_D_tau} and {DecayErrors["Bcharged_errors"]}').copy()

#         res_Dstst_l_nu_charged=df.query(f'DecayMode=={DecayMode[f"res_Dstst_l_nu_charged"].value} and \
#         {B_charged} and {true_D_ell} and {DecayErrors["Bcharged_errors"]}').copy()

#         nonres_Dstst_l_nu_charged=df.query(f'DecayMode=={DecayMode[f"nonres_Dstst_l_nu_charged"].value} and \
#         {B_charged} and {true_D_ell} and {DecayErrors["Bcharged_errors"]}').copy()

# +
# ## DecayHash

# from collections import OrderedDict

# # the order of keys might be important, try to keep the muon modes at the bottom for e reconstruction
# # the e modes will be kept at the bottom for a muon reconstruction
# mode_dict = {}
# mode_dict['e'] = OrderedDict()
# mode_dict['e']['sig_D_tau_nu']=[
#     '511 (-> -411 (-> 321 -211 -211) -15 (-> -11 12 -16) 16)',
#     '-511 (-> 411 (-> -321 211 211) 15 (-> 11 -12 16) -16)']

# mode_dict['e']['sig_D_l_nu']=[
#     '511 (-> -411 (-> 321 -211 -211) -11 12)',
#     '-511 (-> 411 (-> -321 211 211) 11 -12)']

# mode_dict['e']['sig_Dst_tau_nu']=[
#     '511 (-> -413 (-> -411 (-> 321 -211 -211) 111) -15 (-> -11 12 -16) 16)',
#     '511 (-> -413 (-> -411 (-> 321 -211 -211) 22) -15 (-> -11 12 -16) 16)',
#     '-511 (-> 413 (-> 411 (-> -321 211 211) 111) 15 (-> 11 -12 16) -16)',
#     '-511 (-> 413 (-> 411 (-> -321 211 211) 22) 15 (-> 11 -12 16) -16)']

# mode_dict['e']['sig_Dst_l_nu']=[
#     '511 (-> -413 (-> -411 (-> 321 -211 -211) 111) -11 12)',
#     '511 (-> -413 (-> -411 (-> 321 -211 -211) 22) -11 12)',
#     '-511 (-> 413 (-> 411 (-> -321 211 211) 111) 11 -12)',
#     '-511 (-> 413 (-> 411 (-> -321 211 211) 22) 11 -12)']

# mode_dict['e']['Dstst_tau_nu_mixed']=[
#     '511 (-> -10413 -15 (-> -11 12 -16) 16)','-511 (-> 10413 15 (-> 11 -12 16) -16)',
#     '511 (-> -10411 -15 (-> -11 12 -16) 16)','-511 (-> 10411 15 (-> 11 -12 16) -16)',
#     '511 (-> -20413 -15 (-> -11 12 -16) 16)','-511 (-> 20413 15 (-> 11 -12 16) -16)',
#     '511 (-> -415 -15 (-> -11 12 -16) 16)',  '-511 (-> 415 15 (-> 11 -12 16) -16)']

# mode_dict['e']['Dstst_tau_nu_charged']=[
#     '521 (-> -10423 -15 (-> -11 12 -16) 16)','-521 (-> 10423 15 (-> 11 -12 16) -16)',
#     '521 (-> -10421 -15 (-> -11 12 -16) 16)','-521 (-> 10421 15 (-> 11 -12 16) -16)',
#     '521 (-> -20423 -15 (-> -11 12 -16) 16)','-521 (-> 20423 15 (-> 11 -12 16) -16)',
#     '521 (-> -425 -15 (-> -11 12 -16) 16)',  '-521 (-> 425 15 (-> 11 -12 16) -16)']

# mode_dict['e']['res_Dstst_l_nu_mixed']=[
#     '511 (-> -10413 -11 12)','-511 (-> 10413 11 -12)',
#     '511 (-> -10411 -11 12)','-511 (-> 10411 11 -12)',
#     '511 (-> -20413 -11 12)','-511 (-> 20413 11 -12)',
#     '511 (-> -415 -11 12)',  '-511 (-> 415 11 -12)']

# mode_dict['e']['res_Dstst_l_nu_charged']=[
#     '521 (-> -10423 -11 12)','-521 (-> 10423 11 -12)',
#     '521 (-> -10421 -11 12)','-521 (-> 10421 11 -12)',
#     '521 (-> -20423 -11 12)','-521 (-> 20423 11 -12)',
#     '521 (-> -425 -11 12)',  '-521 (-> 425 11 -12)']

# mode_dict['e']['nonres_Dstst_l_nu_mixed']=[
#     '511 (-> -411 111 -11 12)',     '-511 (-> 411 111 11 -12)',
#     '511 (-> -411 111 111 -11 12)', '-511 (-> 411 111 111 11 -12)',
#     '511 (-> -411 211 -211 -11 12)','-511 (-> 411 211 -211 11 -12)',
#     '511 (-> -411 -211 211 -11 12)','-511 (-> 411 -211 211 11 -12)',
#     '511 (-> -413 111 -11 12)',     '-511 (-> 413 111 11 -12)',
#     '511 (-> -413 111 111 -11 12)', '-511 (-> 413 111 111 11 -12)',
#     '511 (-> -413 211 -211 -11 12)','-511 (-> 413 211 -211 11 -12)',
#     '511 (-> -413 -211 211 -11 12)','-511 (-> 413 -211 211 11 -12)']

# mode_dict['e']['nonres_Dstst_l_nu_charged']=[
#     '521 (-> -411 211 -11 12)',    '-521 (-> 411 -211 11 -12)',
#     '521 (-> -411 211 111 -11 12)','-521 (-> 411 -211 111 11 -12)',
#     '521 (-> -411 111 211 -11 12)','-521 (-> 411 111 -211 11 -12)',
#     '521 (-> -413 211 -11 12)',    '-521 (-> 413 -211 11 -12)',
#     '521 (-> -413 211 111 -11 12)','-521 (-> 413 -211 111 11 -12)',
#     '521 (-> -413 111 211 -11 12)','-521 (-> 413 111 -211 11 -12)']

# mode_dict['e']['gap_Dstst_l_nu_mixed']=[
#     '511 (-> -411 221 -11 12)','-511 (-> 411 221 11 -12)',
#     '511 (-> -413 221 -11 12)','-511 (-> 413 221 11 -12)']


# #################################

# mode_dict['mu'] = OrderedDict()
# mode_dict['mu']['sig_D_tau_nu']=[
#     '511 (-> -411 (-> 321 -211 -211) -15 (-> -13 14 -16) 16)',
#     '-511 (-> 411 (-> -321 211 211) 15 (-> 13 -14 16) -16)']

# mode_dict['mu']['sig_D_l_nu']=[
#     '511 (-> -411 (-> 321 -211 -211) -13 14)',
#     '-511 (-> 411 (-> -321 211 211) 13 -14)']

# mode_dict['mu']['sig_Dst_tau_nu']=[
#     '511 (-> -413 (-> -411 (-> 321 -211 -211) 111) -15 (-> -13 14 -16) 16)',
#     '511 (-> -413 (-> -411 (-> 321 -211 -211) 22) -15 (-> -13 14 -16) 16)',
#     '-511 (-> 413 (-> 411 (-> -321 211 211) 111) 15 (-> 13 -14 16) -16)',
#     '-511 (-> 413 (-> 411 (-> -321 211 211) 22) 15 (-> 13 -14 16) -16)']

# mode_dict['mu']['sig_Dst_l_nu']=[
#     '511 (-> -413 (-> -411 (-> 321 -211 -211) 111) -13 14)',
#     '511 (-> -413 (-> -411 (-> 321 -211 -211) 22) -13 14)',
#     '-511 (-> 413 (-> 411 (-> -321 211 211) 111) 13 -14)',
#     '-511 (-> 413 (-> 411 (-> -321 211 211) 22) 13 -14)']

# mode_dict['mu']['Dstst_tau_nu_mixed']=[
#     '511 (-> -10413 -15 (-> -13 14 -16) 16)','-511 (-> 10413 15 (-> 13 -14 16) -16)',
#     '511 (-> -10411 -15 (-> -13 14 -16) 16)','-511 (-> 10411 15 (-> 13 -14 16) -16)',
#     '511 (-> -20413 -15 (-> -13 14 -16) 16)','-511 (-> 20413 15 (-> 13 -14 16) -16)',
#     '511 (-> -415 -15 (-> -13 14 -16) 16)',  '-511 (-> 415 15 (-> 13 -14 16) -16)']

# mode_dict['mu']['Dstst_tau_nu_charged']=[
#     '521 (-> -10423 -15 (-> -13 14 -16) 16)','-521 (-> 10423 15 (-> 13 -14 16) -16)',
#     '521 (-> -10421 -15 (-> -13 14 -16) 16)','-521 (-> 10421 15 (-> 13 -14 16) -16)',
#     '521 (-> -20423 -15 (-> -13 14 -16) 16)','-521 (-> 20423 15 (-> 13 -14 16) -16)',
#     '521 (-> -425 -15 (-> -13 14 -16) 16)',  '-521 (-> 425 15 (-> 13 -14 16) -16)']

# mode_dict['mu']['res_Dstst_l_nu_mixed']=[
#     '511 (-> -10413 -13 14)','-511 (-> 10413 13 -14)',
#     '511 (-> -10411 -13 14)','-511 (-> 10411 13 -14)',
#     '511 (-> -20413 -13 14)','-511 (-> 20413 13 -14)',
#     '511 (-> -415 -13 14)',  '-511 (-> 415 13 -14)',]

# mode_dict['mu']['res_Dstst_l_nu_charged']=[
#     '521 (-> -10423 -13 14)','-521 (-> 10423 13 -14)',
#     '521 (-> -10421 -13 14)','-521 (-> 10421 13 -14)',
#     '521 (-> -20423 -13 14)','-521 (-> 20423 13 -14)',
#     '521 (-> -425 -13 14)',  '-521 (-> 425 13 -14)',]

# mode_dict['mu']['nonres_Dstst_l_nu_mixed']=[
#     '511 (-> -411 111 -13 14)',     '-511 (-> 411 111 13 -14)',
#     '511 (-> -411 111 111 -13 14)', '-511 (-> 411 111 111 13 -14)',
#     '511 (-> -411 211 -211 -13 14)','-511 (-> 411 211 -211 13 -14)',
#     '511 (-> -411 -211 211 -13 14)','-511 (-> 411 -211 211 13 -14)',
#     '511 (-> -413 111 -13 14)',     '-511 (-> 413 111 13 -14)',
#     '511 (-> -413 111 111 -13 14)', '-511 (-> 413 111 111 13 -14)',
#     '511 (-> -413 211 -211 -13 14)','-511 (-> 413 211 -211 13 -14)',
#     '511 (-> -413 -211 211 -13 14)','-511 (-> 413 -211 211 13 -14)',]

# mode_dict['mu']['nonres_Dstst_l_nu_charged']=[
#     '521 (-> -411 211 -13 14)',    '-521 (-> 411 -211 13 -14)',
#     '521 (-> -411 211 111 -13 14)','-521 (-> 411 -211 111 13 -14)',
#     '521 (-> -413 211 -13 14)',    '-521 (-> 413 -211 13 -14)',
#     '521 (-> -413 211 111 -13 14)','-521 (-> 413 -211 111 13 -14)']

# mode_dict['mu']['gap_Dstst_l_nu_mixed']=[
#     '511 (-> -411 221 -13 14)','-511 (-> 411 221 13 -14)',
#     '511 (-> -413 221 -13 14)','-511 (-> 413 221 13 -14)']

# +
# def get_dataframe_samples_new_inclusiveD(df, mode, template=True):
#     samples = {}
#     lepton_PDG = {'e':11, 'mu':13}
    
#     ################## Define lepton #################
#     truel = f'abs(ell_mcPDG)=={lepton_PDG[mode]}'
#     fakel = f'abs(ell_mcPDG)!={lepton_PDG[mode]} and ell_mcErrors!=512'
#     fakeTrack = 'ell_mcErrors==512'
    
#     ################# Define B ####################
    
#     D_Dst_list = [411, 413, -411, -413]
#     Dstst_list = [10413, 10411, 20413, 415, 10423, 10421, 20423, 425,
#                  -10413, -10411, -20413, -415, -10423, -10421, -20423, -425]
#     D_list = D_Dst_list + Dstst_list
#     Pi_eta_list = [111, 211, -211, 221]
    
    
#     # more categories with truel
#     continuum = f'{truel} and B0_isContinuumEvent==1'
    
#     signals = f'{truel} and (abs(ell_genGMPDG)==511 or abs(ell_genGMPDG)==521) and \
#     abs(ell_genMotherPDG)==15 and (ell_GMdaughter_0_PDG in @D_list)'
    
#     norms = f'{truel} and (abs(ell_genMotherPDG)==511 or abs(ell_genMotherPDG)==521) and \
#     (ell_Mdaughter_0_PDG in @D_list)'
    
#     BBbkg = f'{truel} and B0_isContinuumEvent==0 and \
#     ( (abs(ell_genGMPDG)!=511 and abs(ell_genGMPDG)!=521) or abs(ell_genMotherPDG)!=15 or (ell_GMdaughter_0_PDG not in @D_list) ) and \
#     ( (abs(ell_genMotherPDG)!=511 and abs(ell_genMotherPDG)!=521)) or (ell_Mdaughter_0_PDG not in @D_list)'
# #     combinatorial = f'{TDTl} and B0_mcPDG==300553'
# #     singleBbkg = f'{TDTl} and B0_isContinuumEvent==0 and B0_mcPDG!=300553 and \
# #     ( (abs(B0_mcPDG)!=511 and abs(B0_mcPDG)!=521) or \
# #     ( ell_genMotherPDG!=B0_mcPDG and (ell_genGMPDG!=B0_mcPDG or abs(ell_genMotherPDG)!=15) ) )'
    
#     # more categories with signals and norms
#     B2D_tau = f'{signals} and ell_GMdaughter_0_PDG*ell_GMdaughter_1_PDG==411*15'
#     B2D_ell = f'{norms} and ell_Mdaughter_0_PDG*ell_Mdaughter_1_PDG==411*{lepton_PDG[mode]}'
#     B2Dst_tau = f'{signals} and ell_GMdaughter_0_PDG*ell_GMdaughter_1_PDG==413*15'
#     B2Dst_ell = f'{norms} and ell_Mdaughter_0_PDG*ell_Mdaughter_1_PDG==413*{lepton_PDG[mode]}'

#     B2Dstst_tau = f'{signals} and (ell_GMdaughter_0_PDG in @Dstst_list) and abs(ell_GMdaughter_1_PDG)==15'
#     B2Dstst_ell_res = f'{norms} and (ell_Mdaughter_0_PDG in @Dstst_list) and abs(ell_Mdaughter_1_PDG)=={lepton_PDG[mode]}'

#     B2Dstst_ell_gap_non = f'{norms} and (ell_Mdaughter_0_PDG in @D_Dst_list) and ell_Mdaughter_1_PDG in @Pi_eta_list'

#     ######################### Apply selection ###########################
    
#     # Fake bkg components
#     bkg_fakel = df.query(fakel).copy()
#     bkg_fakeTrack = df.query(fakeTrack).copy()
#     samples[r'bkg_fakel'] = bkg_fakel
#     samples[r'bkg_fakeTrack'] = bkg_fakeTrack
    
#     # True Dl bkg components
#     bkg_continuum = df.query(continuum).copy()
#     bkg_BB = df.query(BBbkg).copy()
#     signals_all = df.query(signals).copy()
#     norms_all = df.query(norms).copy()
#     bkg_other_truel = pd.concat([df.query(truel).copy(),
#                                  bkg_continuum,
#                                  bkg_BB,
#                                  signals_all,
#                                  norms_all]).drop_duplicates(
#         subset=['__experiment__','__run__','__event__','__production__'],keep=False)
    
#     samples[r'bkg_continuum'] = bkg_continuum
#     samples[r'bkg_BB'] = bkg_BB
#     samples[r'bkg_other_truel'] = bkg_other_truel
    
#     # True Dl Signal components
#     D_tau_nu=df.query(B2D_tau).copy()
#     D_l_nu=df.query(B2D_ell).copy()
#     Dst_tau_nu=df.query(B2Dst_tau).copy()
#     Dst_l_nu=df.query(B2Dst_ell).copy()
#     Dstst_tau_nu=df.query(B2Dstst_tau).copy()
#     Dstst_l_nu_res=df.query(B2Dstst_ell_res).copy()
#     Dstst_l_nu_gap_non=df.query(B2Dstst_ell_gap_non).copy()
    
#     bkg_other_signal = pd.concat([signals_all,
#                                   norms_all,
#                                   D_tau_nu,
#                                   Dst_tau_nu,
#                                   D_l_nu,
#                                   Dst_l_nu,
#                                   Dstst_tau_nu,
#                                   Dstst_l_nu_res,
#                                   Dstst_l_nu_gap_non]).drop_duplicates(
#         subset=['__experiment__','__run__','__event__','__production__'],keep=False)
    
#     samples[r'$D\tau\nu$'] = D_tau_nu
#     samples[r'$D^\ast\tau\nu$'] = Dst_tau_nu
#     samples[r'$D\ell\nu$'] = D_l_nu
#     samples[r'$D^\ast\ell\nu$'] = Dst_l_nu
#     samples[r'$D^{\ast\ast}\tau\nu$'] = Dstst_tau_nu
#     samples[r'$D^{\ast\ast}\ell\nu$_res'] = Dstst_l_nu_res
#     samples[r'$D^{\ast\ast}\ell\nu$_gap_non'] = Dstst_l_nu_gap_non
#     samples['bkg_other_signal'] = bkg_other_signal
    
#     for name, df in samples.items():
#         df['mode']=DecayMode_inclusiveD[name]
#     return samples
    
# def get_dataframe_samples_old(df, mode, template=True):
#     samples = {}
#     lepton_PDG = {'e':11, 'mu':13}
    
#     # Define Truth matching criteria
#     true_D_tau = f'D_mcErrors<8 and D_mcPDG*ell_mcPDG==411*{lepton_PDG[mode]} and ell_genGMPDG==B0_mcPDG and abs(ell_genMotherPDG)==15'
    
#     true_D_ell = f'D_mcErrors<8 and D_mcPDG*ell_mcPDG==411*{lepton_PDG[mode]} and ell_genMotherPDG==B0_mcPDG'
    
#     true_B0 = 'B0_mcPDG*D_mcPDG==-511*411'
#     B_charged = 'B0_mcPDG*D_mcPDG==-521*411'
    
#     lepton_misID = f'abs(ell_mcPDG)!={lepton_PDG[mode]}'
#     lepton_wrong_mother = f'ell_genMotherPDG!=B0_mcPDG and \
#     (abs(ell_genMotherPDG)!=15 or abs(ell_genMotherPDG)==15 and ell_genGMPDG!=B0_mcPDG)'
    
# ###################### mcErrors need to be handled carefully
# ###################### apparently correct e mcPDG==11 could have large mcErrors, 128, 2048; mcSecPhysProc need to be checked later
#     DecayErrors = {}
#     # norm modes, signal modes, D** mixed modes
#     for key, value in {'D_l':[8,16],'Dst_l':[8,64],'D_tau':[8,32],'Dst_Dstst_mixed':[8,64]}.items():
#                     # tau to e has a 1% radiative mode, thus 32
#         correct_decay = f'({value[0]}+ell_mcErrors)<=B0_mcErrors<({value[1]}+ell_mcErrors)'
#         missing_photon = f'({value[0]+1024}+ell_mcErrors)<=B0_mcErrors<({value[1]+1024}+ell_mcErrors)'
        
# #         wrongBremsDaughter = f'{value[0]+2048}<=B0_mcErrors<{value[1]+2048+128}'
# #         missing_photon_wrongBrems = f'{value[0]+2048+1024}<=B0_mcErrors<{value[1]+2048+1024+128}'
        
#         DecayErrors[f'{key}_errors'] = f'({correct_decay} or {missing_photon})'
#         # note that the parentheses in the `or` statement is very important when using `and` in front
#         if template and (key in ['D_l', 'Dst_l']):
#             DecayErrors[f'{key}_errors'] = correct_decay
        
            
#     # D** charged modes
#     Bcharged_errors = f'({8+256}+ell_mcErrors)<=B0_mcErrors<({64+256}+ell_mcErrors)'
#     missing_photon_Bcharged = f'({8+1024+256}+ell_mcErrors)<=B0_mcErrors<({64+1024+256}+ell_mcErrors)'
    
#     # a charged particle (e.g. pi,e,mu) is added as a Brems daughter by the correctBrem module
#     # however, this mistake doesn't change the MM2 and p_D_l much, still within the signal window
#     # B0_mcErrors is offset by 128 due to the misID of the Brems daughter
# #     wrongBremsDaughter_Bcharged = f'{8+2048+256}<=B0_mcErrors<{64+2048+256+128}'
# #     missing_photon_wrongBrems_Bcharged = f'{8+2048+1024+256}<=B0_mcErrors<{64+2048+1024+256+128}'

#     DecayErrors[f'Bcharged_errors'] = f'({Bcharged_errors} or {missing_photon_Bcharged})'
# #         if template:
# #             DecayErrors[f'Bcharged_errors'] = Bcharged_errors


#     B2D_tau = 'B0_mcDaughter_0_PDG*B0_mcDaughter_1_PDG==411*15'
#     B2D_ell = f'B0_mcDaughter_0_PDG*B0_mcDaughter_1_PDG==411*{lepton_PDG[mode]}'
#     B2Dst_tau = 'B0_mcDaughter_0_PDG*B0_mcDaughter_1_PDG==413*15'
#     B2Dst_ell = f'B0_mcDaughter_0_PDG*B0_mcDaughter_1_PDG==413*{lepton_PDG[mode]}'

#     Dstst_list = [10413, 10411, 20413, 415, 10423, 10421, 20423, 425,
#                  -10413, -10411, -20413, -415, -10423, -10421, -20423, -425]
#     B2Dstst_tau = 'B0_mcDaughter_0_PDG in @Dstst_list and abs(B0_mcDaughter_1_PDG)==15'
#     B2Dstst_ell_res = f'B0_mcDaughter_0_PDG in @Dstst_list and abs(B0_mcDaughter_1_PDG)=={lepton_PDG[mode]}'

#     D_Dst_list = [411, 413, -411, -413]
#     Pi_list = [111, 211, -211]
#     B2Dstst_ell_non = 'B0_mcDaughter_0_PDG in @D_Dst_list and B0_mcDaughter_1_PDG in @Pi_list'
#     B2Dstst_ell_gap = 'B0_mcDaughter_0_PDG in @D_Dst_list and B0_mcDaughter_1_PDG==221'


#     # Sig components
#     sig_D_tau_nu=df.query(f'{true_B0} and {true_D_tau} and {B2D_tau} and \
#     {DecayErrors["D_tau_errors"]}').copy()

#     sig_D_l_nu=df.query(f'{true_B0} and {true_D_ell} and {B2D_ell} and \
#     {DecayErrors["D_l_errors"]}').copy()

#     sig_Dst_tau_nu=df.query(f'{true_B0} and {true_D_tau} and {B2Dst_tau} and \
#     {DecayErrors["Dst_Dstst_mixed_errors"]}').copy()

#     sig_Dst_l_nu=df.query(f'{true_B0} and {true_D_ell} and {B2Dst_ell} and \
#     {DecayErrors["Dst_l_errors"]}').copy()

#     Dstst_tau_nu_mixed=df.query(f'{true_B0} and {true_D_tau} and {B2Dstst_tau} and \
#     {DecayErrors["Dst_Dstst_mixed_errors"]}').copy()

#     res_Dstst_l_nu_mixed=df.query(f'{true_B0} and {true_D_ell} and {B2Dstst_ell_res} and \
#     {DecayErrors["Dst_Dstst_mixed_errors"]}').copy()

#     nonres_Dstst_l_nu_mixed=df.query(f'{true_B0} and {true_D_ell} and {B2Dstst_ell_non} and \
#     {DecayErrors["Dst_Dstst_mixed_errors"]}').copy()

#     gap_Dstst_l_nu_mixed=df.query(f'{true_B0} and {true_D_ell} and {B2Dstst_ell_gap} and \
#     {DecayErrors["Dst_Dstst_mixed_errors"]}').copy()

#     Dstst_tau_nu_charged=df.query(f'{B_charged} and {true_D_tau} and {B2Dstst_tau} and \
#     {DecayErrors["Bcharged_errors"]}').copy()

#     res_Dstst_l_nu_charged=df.query(f'{B_charged} and {true_D_ell} and {B2Dstst_ell_res} and \
#     {DecayErrors["Bcharged_errors"]}').copy()

#     nonres_Dstst_l_nu_charged=df.query(f'{B_charged} and {true_D_ell} and {B2Dstst_ell_non} and \
#     {DecayErrors["Bcharged_errors"]}').copy()
    
#     samples[r'$D\tau\nu$'] = sig_D_tau_nu
#     samples[r'$D^\ast\tau\nu$'] = sig_Dst_tau_nu
#     samples[r'$D^{\ast\ast}\tau\nu$_mixed'] = Dstst_tau_nu_mixed
#     samples[r'$D^{\ast\ast}\tau\nu$_charged'] = Dstst_tau_nu_charged
#     samples[r'$D\ell\nu$'] = sig_D_l_nu
#     samples[r'$D^\ast\ell\nu$'] = sig_Dst_l_nu
#     samples[r'res_$D^{\ast\ast}\ell\nu$_mixed'] = res_Dstst_l_nu_mixed
#     samples[r'nonres_$D^{\ast\ast}\ell\nu$_mixed'] = nonres_Dstst_l_nu_mixed
#     samples[r'gap_$D^{\ast\ast}\ell\nu$_mixed'] = gap_Dstst_l_nu_mixed
#     samples[r'res_$D^{\ast\ast}\ell\nu$_charged'] = res_Dstst_l_nu_charged
#     samples[r'nonres_$D^{\ast\ast}\ell\nu$_charged'] = nonres_Dstst_l_nu_charged
   
    
#     #Bkg components
#     bkg_fakeTracksClusters = df.query('B0_mcErrors==512 and B0_isContinuumEvent!=1').copy()
#     samples[r'bkg_fakeTC'] = bkg_fakeTracksClusters
    
#     bkg_continuum = df.query('B0_isContinuumEvent==1').copy()
#     samples[r'bkg_continuum'] = bkg_continuum
    
#     bkg_BBbar = df.query('B0_mcErrors!=512 and B0_isContinuumEvent!=1')
#     bkg_fakeD = bkg_BBbar.query('(abs(D_mcPDG)!=411 or D_mcErrors>=8)').copy()
#     samples[r'bkg_fakeD'] = bkg_fakeD

#     bkg_trueD = bkg_BBbar.query('abs(D_mcPDG)==411 and D_mcErrors<8')
#     bkg_combinatorial = bkg_trueD.query('B0_mcPDG==300553').copy()
#     samples[r'bkg_combinatorial'] = bkg_combinatorial
    
#     bkg_sigOtherBDTaudecay = bkg_trueD.query(f'B0_mcPDG!=300553 and \
#                 (abs(B0_mcPDG)!=511 and abs(B0_mcPDG)!=521 or \
#                 {lepton_misID} or {lepton_wrong_mother})').copy()
#                 # reconstruct a non-B particle or lepton_misID or lepton from B daughter decay
#     samples[r'bkg_Odecay'] = bkg_sigOtherBDTaudecay
    
    
#     bkg_others = pd.concat([df,
#                             sig_D_tau_nu,
#                             sig_D_l_nu,
#                             sig_Dst_tau_nu,
#                             sig_Dst_l_nu,
#                             Dstst_tau_nu_mixed,
#                             Dstst_tau_nu_charged,
#                             res_Dstst_l_nu_mixed,
#                             nonres_Dstst_l_nu_mixed,
#                             gap_Dstst_l_nu_mixed,
#                             res_Dstst_l_nu_charged,
#                             nonres_Dstst_l_nu_charged,
#                             bkg_fakeTracksClusters,
#                             bkg_fakeD,
#                             bkg_sigOtherBDTaudecay,
#                             bkg_combinatorial,
#                             bkg_continuum]).drop_duplicates(
#                 subset=['__experiment__','__run__','__event__','__production__'],keep=False)
    
#     samples['bkg_others'] = bkg_others

    
#     for name, df in samples.items():
#         df['mode']=DecayMode[name]

#     df = pd.concat([df for df in samples.values()],ignore_index=True).reset_index(drop=True)
#     df['p_D_l_region'] = np.where(df['p_D_l']>2.5,1,0)
    
#     return df, samples

#     # the bkg_others contains events with wrong bremsstralung corrected electrons
#     # The added daughter to the electron are pions, electrons, muons
#     # so the 130<B0_mcErrors<160, e_mcErrors==128, 2176, 2180
