# -*- coding: utf-8 -*-
import numpy as np
############################## define relevant variables ########################

spectators = ['__weight__', 'D_CMS_p', 'ell_CMS_p', ]

CS_variables = ["B0_cosTBTO",    "B0_KSFWV3",    "B0_KSFWV4",     "B0_KSFWV5",     "B0_KSFWV6",
                "B0_KSFWV7",     "B0_KSFWV9",    "B0_KSFWV10",    "B0_KSFWV13",    "B0_KSFWV17",]   
                # "B0_thrustBm", "B0_KSFWV1", "B0_KSFWV2", "B0_KSFWV11", "B0_KSFWV12" correlates with mm2 or p_D_l
                # "B0_R2", "B0_thrustOm","B0_cosTBz","B0_KSFWV8","B0_KSFWV14","B0_KSFWV15","B0_KSFWV16","B0_KSFWV18", data/mc mismodeling

DTC_variables = ['D_A1FflightDistanceSig_IP',   'D_daughterInvM_1_2',  'D_daughterInvM_0_1',] # 'D_vtxReChi2', data/mc mismodeling

B_variables = ['B0_CMS_cos_angle_0_1', 'B0_D_l_DisSig', 'B0_roeMbc_my_mask', 'B0_roeDeltae_my_mask',]
                # 'B0_vtxReChi2',  'B0_TagVReChi2IP',  data/mc mismodeling

# CS_variables = ["B0_cosTBTO",    "B0_KSFWV3",    "B0_KSFWV4",     "B0_KSFWV5",     "B0_KSFWV6",
#                 "B0_KSFWV7",     "B0_KSFWV9",    "B0_KSFWV10",    "B0_KSFWV13",    "B0_KSFWV17",
#                "B0_R2", "B0_thrustOm","B0_cosTBz","B0_KSFWV8","B0_KSFWV14","B0_KSFWV15","B0_KSFWV16","B0_KSFWV18",]   
#                 # "B0_thrustBm", "B0_KSFWV1", "B0_KSFWV2", "B0_KSFWV11", "B0_KSFWV12" correlates with mm2 or p_D_l
#                 # "B0_R2", "B0_thrustOm","B0_cosTBz","B0_KSFWV8","B0_KSFWV14","B0_KSFWV15","B0_KSFWV16","B0_KSFWV18", data/mc mismodeling

# DTC_variables = ['D_A1FflightDistanceSig_IP',   'D_daughterInvM_1_2',  'D_daughterInvM_0_1', 'D_vtxReChi2'] #, data/mc mismodeling

# B_variables = ['B0_CMS_cos_angle_0_1', 'B0_D_l_DisSig', 'B0_roeMbc_my_mask', 'B0_roeDeltae_my_mask', 'B0_vtxReChi2',  'B0_TagVReChi2IP',]
#                 # 'B0_vtxReChi2',  'B0_TagVReChi2IP',  data/mc mismodeling

training_variables = CS_variables + DTC_variables + B_variables
mva_variables = training_variables + spectators

analysis_variables=['__experiment__',     '__run__',       '__event__',      '__production__',
                    'experiment',     'run',       'event',      'production',
                    'B0_isContinuumEvent','B0_mcPDG',      'B0_mcErrors',    'B0_mcDaughter_0_PDG',
                    'B0_mcDaughter_1_PDG','B0_deltaE',     'B0_Mbc',          'B0_dr',
                    
                    'D_mcErrors',         'D_genGMPDG',    'D_genMotherPDG', 'D_mcPDG',
                    'D_BFM',              'D_M',           'D_p',            'D_vtxReChi2',
                    
                    'ell_genMotherPDG',   'ell_pValue',    'ell_mcErrors',   'ell_genGMPDG',    'ell_mcSecPhysProc',
                    'ell_BFbrems_electronIDNN', 'ell_BFbrems_p', 'ell_BFbrems_cosTheta', 'ell_BFbrems_theta',
                    'ell_BFbrems_charge', 'ell_BFbrems_PDG', 'ell_BFbrems_mcPDG',               

                    'ell_p',    'ell_cosTheta',    'ell_theta',    'ell_charge',    'ell_PDG',    'ell_mcPDG',
                    'K_p',      'K_cosTheta',      'K_theta',      'K_charge',      'K_PDG',      'K_mcPDG',
                    'pi1_p',    'pi1_cosTheta',    'pi1_theta',    'pi1_charge',    'pi1_PDG',    'pi1_mcPDG',
                    'pi2_p',    'pi2_cosTheta',    'pi2_theta',    'pi2_charge',    'pi2_PDG',    'pi2_mcPDG',
                    
                    'mode',               'Ecms',          'p_D_l',          'B_D_ReChi2', 'B0_D_ReChi2',
                    'sig_prob',           'fakeD_prob',    'fakeB_prob',     'continuum_prob',
                    'combinatorial_prob', 'nPi0',          'pi0_p',          'pi0_cosTheta',
                    
                    'B0_recQ2Bh',         'B0_recQ2BhSimple',                       'B0_mcMomTransfer2',  
                    'B0_recMissM2',       'B0_missingMomentumOfEvent_theta',        'B0_CMS_roeP_my_mask',
                    'B0_roeEextra_my_mask',   'B0_roeCharge_my_mask',               'B0_CMS_roeE_my_mask',
                    'B0_nROE_Tracks_my_mask',  'nMC_K_L',           'B0_vtxReChi2',]
#                  'D_K_kaonIDNN',        'D_K_pionIDNN',  'D_pi2_kaonIDNN', 'D_pi2_pionIDNN',
#                  'D_pi1_kaonIDNN',      'D_pi1_pionIDNN',]
#                'B0_nROE_Tracks_my_mask',  'B0_nROE_Photons_my_mask',  'B0_nROE_NeutralHadrons_my_mask',


neutral_cols_D = [f"D_511_{i}_daughterPDG" for i in range(7)]   # 7 columns for neutral-B daughters
charged_cols_D = [f"D_521_{i}_daughterPDG" for i in range(7)]
neutral_cols_ell = [f"ell_511_{i}_daughterPDG" for i in range(7)]
charged_cols_ell = [f"ell_521_{i}_daughterPDG" for i in range(7)]
combinatorial_vars_D = neutral_cols_D + charged_cols_D
combinatorial_vars_ell = neutral_cols_ell + charged_cols_ell
combinatorial_vars = combinatorial_vars_D + combinatorial_vars_ell

veto_vars = ['DstVeto_massDiff_0']

all_relevant_variables = mva_variables + analysis_variables + combinatorial_vars + veto_vars

DecayMode_new = {'bkg_fakeTracks':0,         'bkg_fakeD':1,           'bkg_fakeL':2,
                 'bkg_continuum':3,          'bkg_combinatorial':4,   'bkg_hadronicB_secondaryL':5,
                 'bkg_other_TDTl':6,         'bkg_other_signal':7,
                 r'$D\tau\nu$':8,            r'$D^\ast\tau\nu$':9,    r'$D\ell\nu$':10,
                 r'$D^\ast\ell\nu$':11,                r'$D^{\ast\ast}\tau\nu$':12,
                 r'$D^{\ast\ast}\ell\nu$_narrow':13,   r'$D^{\ast\ast}\ell\nu$_broad':14,
                 r'$D\ell\nu$_gap_pi':15,              r'$D\ell\nu$_gap_eta':16,     r'$D\ell\nu$_gap':17}

lgb_tight = 'sig_prob>0.52 and fakeD_prob<0.13 and continuum_prob<0.11'
lgb_loose = 'sig_prob>0.2 and fakeD_prob<0.15 and continuum_prob<0.5 and combinatorial_prob<0.6'
lgb_comb = 'fakeD_prob<0.15 and continuum_prob<0.5 and combinatorial_prob>0.5'

offline_cut = '(5<B0_roeMbc_my_mask) & (-4<B0_roeDeltae_my_mask) & (B0_roeDeltae_my_mask<1) & (B0_dr<0.1) & (ell_p<4)'

Dst_veto_cut = '( (DstVeto_massDiff_0<0.135) | (0.145<DstVeto_massDiff_0) )'

# pyhf fit bin edges
MM2_bins_sr = np.concatenate((np.linspace(-2.5,2.5,11), np.linspace(2.8,10,31) ) )
p_D_l_bins_sr = np.concatenate(([0.4, 0.6, 0.8], np.linspace(1,3,41), 
                             [3.1, 3.2, 3.3, 3.4, 3.6, 3.8, 4, 4.2, 4.4, 4.6, 4.8]))

MM2_bins_sb = np.linspace(-2.5,10,21)
p_D_l_bins_sb = np.linspace(0.4,4.8,21)


FIT_VARIABLES = ['B0_recMissM2', 'p_D_l']
MVA_OUTPUTS = ['sig_prob', 'fakeD_prob', 'continuum_prob', 'combinatorial_prob']

FIT_VARIABLE_LABELS = {
    'B0_recMissM2': r'$M_{\mathrm{miss}}^2$ [GeV$^2$]',
    'p_D_l':        r'$|p^*_D| + |p^*_\ell|$ [GeV]',
}

D_MASS_SR = '1.855<D_M<1.885'       # same window as create_templates_new
D_MASS_SB = 'D_M<1.85 or 1.9<D_M'   # same sidebands as create_templates_new

############################## derived (post-loading) variables ########################

# name -> pandas.eval expression. Evaluated in insertion order, so a later
# entry may use an earlier one. The input branches must be present in the
# loaded columns (all four are already in all_relevant_variables).
derived_variables = {
    'B_D_ReChi2': 'B0_vtxReChi2 + D_vtxReChi2',
    'p_D_l':      'D_CMS_p + ell_CMS_p',
    # 'cos_D_l':  '(D_px*ell_px + D_py*ell_py + D_pz*ell_pz) / (D_p*ell_p)',
}


############################## define relevant constants ########################

# for event classification
pi_pdg = [111, 211, -211]
eta_pdg = [221]

charged_D_Dst_pdg = [411, 413, -411, -413]
neutral_D_Dst_pdg = [421, 423, -421, -423]
D_Dst_pdg = charged_D_Dst_pdg + neutral_D_Dst_pdg

Dstst_narrow_pdg = [10413, 10423, 415, 425, -10413, -10423, -415, -425]
Dstst_broad_pdg  = [10411, 10421, 20413, 20423, -10411, -10421, -20413, -20423]
Dstst_pdg   = Dstst_narrow_pdg + Dstst_broad_pdg

D_s_pdg = [431, 433, 10431, 20433, 10433, 435, -431, -433, -10431, -20433, -10433, -435]

# for combinatorial classification
D_mesons_pdg = D_Dst_pdg + Dstst_pdg + D_s_pdg
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
leptons = {11, -11, 12, -12, 
           13, -13, 14, -14,
           15, -15, 16, -16}
Bpdg = {511, -511, 521, -521}


measured_pdg_norad = [# 1 charm, mixed
    411*11*12,
    413*11*12,
    411*13*14,
    413*13*14,
    411*15*16,
    413*15*16,
    411*15*16,
    20213*411,
    413*15*16,
    413*20213,
    411*2212*2112,
    411*223*211,
    411*321*311,
    # 411*213*223,
    # 411*1114*2114,
    # 411*211*211*213,
    # 411*213*211*211,
    213*411,
    # 411*211*213*211,
    # 411*223*213,
    411*323*311,
    413*211*211*211*211*211,
    # 411*221*213,
    # 411*213*213*211,
    # 411*213*221,
    # 411*211*213*213,
    411*2112*2112*211,
    413*223*211,
    413*213,
    413*321*311,
    413*2212*2112,
    411*321*313,
    411*321*321*211,
    411*323*321*211,
    411*2212*2112*111,
    411*211,
    413*211,
    411*323,
    413*323,

                    # 1 charm, charged
    421*11*12,
    423*11*12,
    421*13*14,
    423*13*14,
    421*15*16,
    423*15*16,
    # 411*1114*2214,
    # 411*1114*2212,
    # 425*15*16,
    # 411*2114*2224,
    # 10421*15*16,
    10423*213,
    # 411*2112*2224,
    # 425*213,
    # 213*10421,
    # 413*1114*2214,
    10423*211,
    10421*211,
    411*211*211,

                    # 2 charm, mixed
    10431*411,      # correction mode 1
    433*411,
    413*10431,      # correction mode 2
    411*413*311,
    411*431,
    413*413*311,
    # 411*411*313,
    20433*411,
    411*423*321,
    413*20433,
    413*423*321,
    # 413*411*313,
    433*413,
    # 411*413*313,
    # 413*413*313,
    # 433*413,
    # 433*411*211*211,
    413*431,
    # 411*423*323,
    # 431*411*211*211,
    # 411*421*323,
    413*411*311,
    # 433*411*111*111,
    # 431*411*111*111,
    # 433*411*111,
    # 431*411*111,
    # 413*423*323,
    411*411*311,
    411*421*321,
    # 415*431,
    # 415*433,
    # 10413*433,
    413*421*321,
    # 413*421*323,
    413*411,
    413*413,
    411*411,
    411*411*321*211,
    411*411*311*111,
    30443*130,
    30443*310,

                    # 2 charm, charged
    423*413*311,
    # 421*411*313,
    # 423*411*313,
    423*411*311,
    # 433*411*211,
    # 431*411*211,
    # 431*411*211*111,
    # 423*413*313,
    421*411*311,
    # 433*411*211*111,
    421*413*311,
    # 425*433,
    # 425*431,
    # 421*413*313,
    # 411*411*323,
    413*413*321,
    413*411*321,
    423*411,
    30443*321,
    423*413,
    421*411,
    411*411*321,
    10431*421,    # mode 3: B+ -> D_s0*+ D0
    10431*423,    # mode 4: B+ -> D_s0*+ D*0
]

measured_pdg_rad = [pdg * 22 for pdg in measured_pdg_norad]

measured_pdg_list = measured_pdg_norad + measured_pdg_rad

# replace some non-resonant 3/4 body hadronic B decays by 2 body decays
hadronicB_replacement_map = {431 * 411 * 211 * 211: 431 * 10413,  # Ds D pi pi -> Ds D1
                             433 * 411 * 211 * 211: 433 * 10413,  # Ds* D pi pi -> Ds* D1
                             431 * 411 * 111 * 111: 431 * 20413,  # Ds D pi0 pi0 -> Ds D1'        415/10413    D2*/D1
                             433 * 411 * 111 * 111: 433 * 20413,  # Ds* D pi0 pi0 -> Ds* D1'      415/10413    D2*/D1
                             431 * 411 * 111: 431 * 415,          # Ds D pi0 -> Ds D2*
                             433 * 411 * 111: 433 * 415,          # Ds* D pi0 -> Ds* D2*
                             431 * 411 * 211: 431 * 425,          # Ds D pi -> Ds D2*
                             433 * 411 * 211: 433 * 425,          # Ds* D pi -> Ds* D2*
                             431 * 411 * 211 * 111: 431 * 20423,  # Ds D pi pi0 -> Ds D1'         425/10423    D2*/D1
                             433 * 411 * 211 * 111: 433 * 20423,  # Ds* D pi pi0 -> Ds* D1'       425/10423    D2*/D1
                            }


# ============================================================
# Truth-level category definitions (universal across BCS/PID/etc.)
# ============================================================

# --- D-side truth (mode-independent) ---
# NOTE (exhaustiveness): these three predicates cover D_mcErrors == 0,
# 0 < D_mcErrors < 512, and D_mcErrors == 512 only. Candidates with
# D_mcErrors > 512 (the clone/fake-track bit set together with any other
# mismatch bit, or any higher bit) match none of them and are silently
# dropped from every category. The true-D branch has catch-alls
# (bkg_other_TDTl, bkg_other_signal); the fake branch does not.
# Run scripts/validate_truth_categories.py to measure the leakage on a
# real ntuple before assuming it is negligible.
TRUE_D = 'D_mcErrors==0'
FAKE_D = '0<D_mcErrors<512'
FAKE_TRACKS = 'D_mcErrors==512'

# --- lepton-side truth (mode-dependent) ---
LEPTON_PDG = {'e': 11, 'mu': 13}

TRUE_LEPTON = {
    'e':  'abs(ell_BFbrems_mcPDG)==11',
    'mu': 'abs(ell_mcPDG)==13',
}
FAKE_LEPTON = {
    'e':  'abs(ell_BFbrems_mcPDG)!=11',
    'mu': 'abs(ell_mcPDG)!=13',
}


def get_truth_categories(mode: str) -> dict:
    """
    Build the full set of MC-truth category query strings for a given
    lepton mode ('e' or 'mu'). This is the single source of truth for
    the classification scheme used in classify_mc_dict, and can be
    imported by any other function (e.g. BCS or PID performance code)
    that needs the same truth categories.
    """
    truel = TRUE_LEPTON[mode]
    fakel = FAKE_LEPTON[mode]

    TDFl = f'{TRUE_D} and {fakel}'
    TDTl = f'{TRUE_D} and {truel}'

    continuum = f'{TDTl} and B0_isContinuumEvent==1'
    combinatorial = f'{TDTl} and B0_mcPDG==300553'
    signals = f'{TDTl} and (abs(B0_mcPDG)==511 or abs(B0_mcPDG)==521) and \
    (ell_genMotherPDG==B0_mcPDG or ell_genGMPDG==B0_mcPDG and abs(ell_genMotherPDG)==15)'
    hadronicB_secondaryL = f'{TDTl} and B0_isContinuumEvent==0 and B0_mcPDG!=300553 and \
    ( (abs(B0_mcPDG)!=511 and abs(B0_mcPDG)!=521) or \
    ( ell_genMotherPDG!=B0_mcPDG and (ell_genGMPDG!=B0_mcPDG or abs(ell_genMotherPDG)!=15) ) )'

    B2D_tau     = f'{signals} and B0_mcDaughter_0_PDG*B0_mcDaughter_1_PDG==411*15'
    B2D_ell     = f'{signals} and B0_mcDaughter_0_PDG*B0_mcDaughter_1_PDG==411*{LEPTON_PDG[mode]}'
    B2Dst_tau   = f'{signals} and B0_mcDaughter_0_PDG*B0_mcDaughter_1_PDG==413*15'
    B2Dst_ell   = f'{signals} and B0_mcDaughter_0_PDG*B0_mcDaughter_1_PDG==413*{LEPTON_PDG[mode]}'

    B2Dstst_tau        = f'{signals} and B0_mcDaughter_0_PDG in @Dstst_pdg and abs(B0_mcDaughter_1_PDG)==15'
    B2Dstst_ell_narrow = f'{signals} and B0_mcDaughter_0_PDG in @Dstst_narrow_pdg and abs(B0_mcDaughter_1_PDG)=={LEPTON_PDG[mode]}'
    B2Dstst_ell_broad  = f'{signals} and B0_mcDaughter_0_PDG in @Dstst_broad_pdg and abs(B0_mcDaughter_1_PDG)=={LEPTON_PDG[mode]}'

    B2D_ell_gap_pi  = f'{signals} and B0_mcDaughter_0_PDG in @charged_D_Dst_pdg and B0_mcDaughter_1_PDG in @pi_pdg'
    B2D_ell_gap_eta = f'{signals} and B0_mcDaughter_0_PDG in @charged_D_Dst_pdg and B0_mcDaughter_1_PDG in @eta_pdg'

    return {
        'trueD': TRUE_D, 'fakeD': FAKE_D, 'fakeTracks': FAKE_TRACKS,
        'truel': truel, 'fakel': fakel,
        'TDFl': TDFl, 'TDTl': TDTl,
        'continuum': continuum, 'combinatorial': combinatorial,
        'signals': signals, 'hadronicB_secondaryL': hadronicB_secondaryL,
        'B2D_tau': B2D_tau, 'B2D_ell': B2D_ell,
        'B2Dst_tau': B2Dst_tau, 'B2Dst_ell': B2Dst_ell,
        'B2Dstst_tau': B2Dstst_tau,
        'B2Dstst_ell_narrow': B2Dstst_ell_narrow,
        'B2Dstst_ell_broad': B2Dstst_ell_broad,
        'B2D_ell_gap_pi': B2D_ell_gap_pi,
        'B2D_ell_gap_eta': B2D_ell_gap_eta,
    }


########################### define known corrections ########################

# so far, NOT used anywhere
def create_naive_data_mc_correction( # so far, NOT used anywhere
    df_data,
    df_mc,
    cut,
    var,
    mc_weight=0.25,
    corr_col_name="naive_corr_w",
):
    """Add a binned data/MC ratio weight to a simulated sample.

    The correction is calculated after applying ``cut``.  A copy of the selected
    MC rows is returned, leaving the caller's DataFrame unchanged.  Empty MC bins
    receive the neutral weight of one.
    """
    if cut is not None:
        df_data = df_data.query(cut)
        df_mc = df_mc.query(cut)
    else:
        df_mc = df_mc.copy()
    
    # Count events per integer value
    counts_data = df_data[var].value_counts().sort_index()
    counts_mc   = mc_weight * df_mc[var].value_counts().sort_index()
    
    # Make sure both have same index
    all_bins = sorted(set(counts_data.index).union(set(counts_mc.index)))
    
    counts_data = counts_data.reindex(all_bins, fill_value=0)
    counts_mc   = counts_mc.reindex(all_bins, fill_value=0)
    
    # Compute ratio safely
    ratio = counts_data / counts_mc.replace(0, np.nan)
    
    ratio_df = pd.DataFrame({
        "N_data": counts_data,
        "N_mc": counts_mc,
        "ratio": ratio
    })
    
    print(ratio_df)
    df_mc[corr_col_name] = df_mc[var].map(ratio).fillna(1.0)
    return df_mc


def apply_event_by_event_weight(df, weights, weight_col):
    """Multiply *weights* by an event-weight column when it is available."""
    if weight_col in df.columns:
        return np.asarray(weights) * df[weight_col].to_numpy()

    print(f"Warning: column '{weight_col}' not found; skipping event by event weighting")
    return weights


# Backwards-compatible spelling used by existing notebooks.
apply_eventByEvent_weight = apply_event_by_event_weight


def pid_variation_columns(df, weight_col):
    """Variation columns `weight_col`_0, _1, ... made by apply_pid_corrections, in order."""
    found = []
    for col in df.columns:
        head, _, tail = str(col).rpartition('_')
        if head == weight_col and tail.isdigit():
            found.append((int(tail), col))
    return [col for _, col in sorted(found)]


def pid_toy_histograms(sample, variable, bins, base_weight, weight_col):
    """Histogram of `variable` filled once per PID-weight variation.

    base_weight: per-event weight WITHOUT the PID correction (the luminosity scale).
    Returns an array (n_bins, n_toys), or None if `sample` has no variation columns.
    Same binning convention as np.histogram (last bin includes its right edge).
    """
    cols = pid_variation_columns(sample, weight_col)
    if not cols:
        return None
    bins = np.asarray(bins, float)
    n_bins = len(bins) - 1
    x = sample[variable].to_numpy(dtype=float)
    idx = np.searchsorted(bins, x, side='right') - 1
    idx[x == bins[-1]] = n_bins - 1
    ok = (idx >= 0) & (idx < n_bins)
    w = np.broadcast_to(np.asarray(base_weight, float), x.shape)[ok, None] \
        * sample[cols].to_numpy(dtype=float)[ok]
    return np.stack([np.bincount(idx[ok], weights=w[:, k], minlength=n_bins)
                     for k in range(w.shape[1])], axis=1)


def draw_mc_band(ax, bins, nominal, err, label='MC unc.'):
    """Shaded +-err band on top of a stacked MC histogram."""
    lo, hi = nominal - err, nominal + err
    ax.fill_between(bins, np.append(lo, lo[-1]), np.append(hi, hi[-1]), step='post',
                    color='gray', alpha=0.4, linewidth=0, label=label, zorder=3)

def apply_pi0_eff_correction(df, corr_table, corr_col_name='pi0_eff_weight'):
    """Merge momentum/angular pi0-efficiency corrections into a DataFrame."""
    df = df.copy()
    table = pd.read_csv(corr_table)

    # --- Build bin edges ---
    p_bins = sorted(set(table['p_min']).union(table['p_max']))
    cos_bins = sorted(set(table['cosTheta_min']).union(table['cosTheta_max']))

    # --- Assign bins to df ---
    df['p_bin'] = pd.cut(df['pi0_p'], bins=p_bins, right=False)
    df['cos_bin'] = pd.cut(df['pi0_cosTheta'], bins=cos_bins, right=False)

    # --- Convert intervals to tuple form for matching ---
    table['p_bin'] = pd.IntervalIndex.from_arrays(
        table['p_min'], table['p_max'], closed='left'
    )
    table['cos_bin'] = pd.IntervalIndex.from_arrays(
        table['cosTheta_min'], table['cosTheta_max'], closed='left'
    )

    # --- Merge ---
    merged = df.merge(
        table[['p_bin', 'cos_bin', 'data_MC_ratio', 'data_MC_ratio_err']],
        on=['p_bin', 'cos_bin'],
        how='left',
        sort=False,
        validate='many_to_one',
    )

    # --- Rename output columns ---
    merged = merged.rename(columns={
        'data_MC_ratio': corr_col_name,
        'data_MC_ratio_err': f'{corr_col_name}_err'
    })
    
    mask_missing = merged[corr_col_name].isna()
    print(merged.loc[mask_missing, ['pi0_p', 'pi0_cosTheta']].describe())
    merged[corr_col_name] = merged[corr_col_name].fillna(1)
    merged[f'{corr_col_name}_err'] = merged[f'{corr_col_name}_err'].fillna(0)

    return merged


import warnings
import pandas as pd

def apply_pid_corrections(df, MC='MC16', run='run1', channel='e', corr_col_name='total_PIDweight',
                          n_variations=100, seed=1, keep_particle_variations=False):
    """Add the PID correction weight `corr_col_name` (product over the lepton, K, pi1, pi2
    weights) and, for n_variations > 0, its variations `corr_col_name`_0 ... _{n-1}.

    sysvar draws each variation from the stat + sys uncertainties of the PID tables
    (sys fully correlated within a table, stat independent between table bins), so a
    histogram filled with variation i is one toy of the PID uncertainty; the spread
    over toys is the PID uncertainty of that histogram, including all correlations.
    Toy i of the total weight is the product of toy i of every particle.

    seed: makes the variations reproducible and, more importantly, makes particles
        that use the SAME tables (pi1 and pi2) get the SAME variations, so their
        uncertainties add coherently. Without it pi1 and pi2 are varied independently
        and the PID uncertainty is underestimated. Needs a basf2 release whose sysvar
        supports `seed`; otherwise a warning is printed. Use the same seed for every
        MC sample that ends up in one plot.
    keep_particle_variations: keep the per-particle variation columns (4 x n_variations
        columns; dropped by default to save memory).
    """
    import sysvar

    e_eff_table = pd.read_csv(f'/home/belle/zhangboy/inclusive_R_D/{MC}_sys_tables/{MC}_pid_tables/e_efficiency_{run}_TwophotonEe.csv')
    pi_e_fake = pd.read_csv(f'/home/belle/zhangboy/inclusive_R_D/{MC}_sys_tables/{MC}_pid_tables/pi_e_fake_{run}.csv')
    K_e_fake = pd.read_csv(f'/home/belle/zhangboy/inclusive_R_D/{MC}_sys_tables/{MC}_pid_tables/K_e_fake_{run}.csv')

    mu_eff_table = pd.read_csv(f'/home/belle/zhangboy/inclusive_R_D/{MC}_sys_tables/{MC}_pid_tables/mu_efficiency_{run}_TwophotonMumu.csv')
    pi_mu_fake = pd.read_csv(f'/home/belle/zhangboy/inclusive_R_D/{MC}_sys_tables/{MC}_pid_tables/pi_mu_fake_{run}.csv')
    K_mu_fake = pd.read_csv(f'/home/belle/zhangboy/inclusive_R_D/{MC}_sys_tables/{MC}_pid_tables/K_mu_fake_{run}.csv')

    k_eff_table = pd.read_csv(f'/home/belle/zhangboy/inclusive_R_D/{MC}_sys_tables/{MC}_pid_tables/k_efficiency_{run}.csv')
    pi_k_fake = pd.read_csv(f'/home/belle/zhangboy/inclusive_R_D/{MC}_sys_tables/{MC}_pid_tables/pi_k_fake_{run}.csv')
    e_k_fake = pd.read_csv(f'/home/belle/zhangboy/inclusive_R_D/{MC}_sys_tables/{MC}_pid_tables/e_k_fake_{run}_TwophotonEe.csv')
    
    pi_eff_table = pd.read_csv(f'/home/belle/zhangboy/inclusive_R_D/{MC}_sys_tables/{MC}_pid_tables/pi_efficiency_{run}.csv')
    k_pi_fake = pd.read_csv(f'/home/belle/zhangboy/inclusive_R_D/{MC}_sys_tables/{MC}_pid_tables/k_pi_fake_{run}.csv')
    e_pi_fake = pd.read_csv(f'/home/belle/zhangboy/inclusive_R_D/{MC}_sys_tables/{MC}_pid_tables/e_pi_fake_{run}_TwophotonEe.csv')
    mu_pi_fake = pd.read_csv(f'/home/belle/zhangboy/inclusive_R_D/{MC}_sys_tables/{MC}_pid_tables/mu_pi_fake_{run}_TwophotonMumu.csv')
    
    e_tables = {(11, 11): e_eff_table,
                (11, 211): pi_e_fake,
                (11, 321): K_e_fake}
    e_thresholds = {11: ('electronIDNN', 0.9)}

    mu_tables = {(13, 13): mu_eff_table,
                 (13, 211): pi_mu_fake,
                 (13, 321): K_mu_fake}
    mu_thresholds = {13: ('muonIDNN', 0.9)}
    
    k_tables = {(321, 321): k_eff_table,
                (321, 211): pi_k_fake,
                (321, 11):  e_k_fake}
    k_thresholds = {321: ('kaonIDNN', 0.9)}
    
    pi_tables = {(211, 211): pi_eff_table,
                 (211, 321): k_pi_fake,
                 (211, 13):  mu_pi_fake,
                 (211, 11):  e_pi_fake}
    pi_thresholds = {211: ('pionIDNN', 0.1)}


    assert channel in ['e', 'mu'], 'channel must be either e or mu'
    # sysvar silently ignores keyword arguments it does not know, so check for `seed`
    seed_supported = hasattr(sysvar, '_derive_seed')
    if n_variations > 0 and seed is not None and not seed_supported:
        print("Warning: this sysvar version has no `seed` option: pi1 and pi2 get independent "
              "PID variations and the PID uncertainty is underestimated.")
    kw = dict(generate_variations=n_variations > 0, n_variations=n_variations)
    if seed is not None and seed_supported:
        kw['seed'] = seed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if channel=='e':
            sysvar.add_weights_to_dataframe('ell_BFbrems', df, systematic='custom_PID', custom_tables=e_tables, custom_thresholds=e_thresholds, **kw)
            weight_cols = ['ell_BFbrems_Weight', 'K_Weight', 'pi1_Weight', 'pi2_Weight',]
        elif channel=='mu':
            sysvar.add_weights_to_dataframe('ell', df, systematic='custom_PID', custom_tables=mu_tables, custom_thresholds=mu_thresholds, **kw)
            weight_cols = ['ell_Weight', 'K_Weight', 'pi1_Weight', 'pi2_Weight',]
        
        sysvar.add_weights_to_dataframe('K', df, systematic='custom_PID', custom_tables=k_tables, custom_thresholds=k_thresholds, **kw)
        sysvar.add_weights_to_dataframe('pi1', df, systematic='custom_PID', custom_tables=pi_tables, custom_thresholds=pi_thresholds, **kw)
        sysvar.add_weights_to_dataframe('pi2', df, systematic='custom_PID', custom_tables=pi_tables, custom_thresholds=pi_thresholds, **kw)
    
    # Consolidate the fragmented internal memory layout.
    df = df.copy()

    df[corr_col_name] = df[weight_cols].product(axis=1)

    if n_variations > 0:
        _check_shared_table_variations(df, 'pi1_Weight', 'pi2_Weight', n_variations)
        toys = np.ones((len(df), n_variations))
        for col in weight_cols:
            toys *= df[[f'{col}_{i}' for i in range(n_variations)]].to_numpy()
        if not keep_particle_variations:
            df = df.drop(columns=[f'{col}_{i}' for col in weight_cols for i in range(n_variations)])
        toy_df = pd.DataFrame(toys.astype(np.float32), index=df.index,
                              columns=[f'{corr_col_name}_{i}' for i in range(n_variations)])
        df = pd.concat([df, toy_df], axis=1)

    return df


def _check_shared_table_variations(df, col_a, col_b, n_variations, n_check=5):
    """pi1 and pi2 use the same tables, so tracks in the same table bin (same nominal
    weight) must get the same variations. Warn if they do not (no common seed)."""
    same_bin = (df[col_a] == df[col_b]) & (df[col_a] != 1.0)
    if not same_bin.any():
        return
    n = min(n_check, n_variations)
    a = df.loc[same_bin, [f'{col_a}_{i}' for i in range(n)]].to_numpy()
    b = df.loc[same_bin, [f'{col_b}_{i}' for i in range(n)]].to_numpy()
    if np.nanmax(np.abs(a - b)) > 1e-9:
        print(f"Warning: {col_a} and {col_b} come from the same PID tables but were varied "
              f"independently; their PID uncertainties will not add coherently.")


################################ PID performance evaluation ###########################
import numpy as np
import uproot


# ======================================================================
# Pure helper functions (stateless — operate only on their arguments,
# independently testable/reusable outside the class)
# ======================================================================

def attach_weights(perf_table, counts, p_bins, cosTheta_bins, decimals=4):
    df = perf_table.copy()

    p_bins_r = np.round(p_bins, decimals)
    cosTheta_bins_r = np.round(cosTheta_bins, decimals)
    p_min_r = np.round(df["p_min"].values, decimals)
    cosTheta_min_r = np.round(df["cosTheta_min"].values, decimals)

    p_idx = np.searchsorted(p_bins_r, p_min_r, side="left")
    c_idx = np.searchsorted(cosTheta_bins_r, cosTheta_min_r, side="left")

    assert np.allclose(p_bins_r[p_idx], p_min_r), \
        "p binning mismatch between p_bins and performance table"
    assert np.allclose(cosTheta_bins_r[c_idx], cosTheta_min_r), \
        "cosTheta binning mismatch between cosTheta_bins and performance table"

    df["weight"] = counts[p_idx, c_idx]
    return df


def weighted_average(df,
                      value_col="data_efficiency",
                      stat_col="data_uncertainty_stat_up",
                      syst_col="data_uncertainty_sys_up"):
    """
    Signal-MC-density-weighted average of a performance table, with
    stat uncertainty added in quadrature and syst uncertainty added
    linearly (conservative placeholder -- prefer the Belle II framework's
    own covariance-aware folding tool if available).
    """
    w = df["weight"].values
    v = df[value_col].values
    stat = df[stat_col].values
    syst = df[syst_col].values

    wsum = w.sum()
    if wsum == 0:
        raise ValueError(
            "No signal MC events fell into the table's phase space - "
            "check binning/units."
        )

    mean_val = np.sum(w * v) / wsum
    stat_err = np.sqrt(np.sum((w * stat) ** 2)) / wsum
    syst_err = np.sum(w * syst) / wsum
    spread = np.sqrt(np.sum(w * (v - mean_val) ** 2) / wsum)

    return {
        "mean": mean_val,
        "stat_err": stat_err,
        "syst_err": syst_err,
        "spread_over_populated_region": spread,
        "n_signal_used": wsum,
    }


# ======================================================================
# Stateful class: owns signal MC loading/caching and species -> track mapping
# ======================================================================

class PIDNNPerformanceEvaluator:
    """
    Computes signal-MC-phase-space-weighted PIDNN efficiency/fake-rate
    values for B -> D tau nu, folding the official Belle II PIDNN
    performance tables through the (p, cosTheta) density of the relevant
    track species in signal MC.

    Fixed analysis-specific settings (MC path, offline cut, branches to
    load) are hardcoded below since they aren't expected to change --
    update them here directly if they ever do.
    """

    MC_PATH = '/home/belle/zhangboy/inclusive_R_D/Samples/MC16rd_signals.root'
    OFFLINE_CUT = offline_cut
    COLUMNS = analysis_variables

    def __init__(self):
        self._mc_cache = {}

    # ------------------------------------------------------------------
    # MC loading (cached per instance)
    # ------------------------------------------------------------------
    def _load_signal_mc(self):
        if self._mc_cache:
            return self._mc_cache

        sigMC_e = uproot.concatenate(
            [f'{self.MC_PATH}:MC_e_loose'],
            library="pd", cut=self.OFFLINE_CUT,
            filter_branch=lambda branch: branch.name in self.COLUMNS)
        sigMC_mu = uproot.concatenate(
            [f'{self.MC_PATH}:MC_mu_loose'],
            library="pd", cut=self.OFFLINE_CUT,
            filter_branch=lambda branch: branch.name in self.COLUMNS)

        signal_e = classify_mc_dict(sigMC_e, 'e', template=False)[r'$D\tau\nu$']
        signal_mu = classify_mc_dict(sigMC_mu, 'mu', template=False)[r'$D\tau\nu$']

        self._mc_cache['e'] = signal_e
        self._mc_cache['mu'] = signal_mu
        return self._mc_cache

    def clear_cache(self):
        """Force the next call to reload signal MC from disk."""
        self._mc_cache = {}

    # ------------------------------------------------------------------
    # Species -> (p, cosTheta) track arrays
    # ------------------------------------------------------------------
    def _get_species_tracks(self, species):
        """
        species mapping:
          'k'  -> signal_e + signal_mu, K_p / K_cosTheta
          'pi' -> signal_e + signal_mu, pi1 and pi2 stacked
          'e'  -> signal_e only, ell_BFbrems_p / ell_BFbrems_cosTheta
          'mu' -> signal_mu only, ell_p / ell_cosTheta
        """
        mc = self._load_signal_mc()
        signal_e, signal_mu = mc['e'], mc['mu']

        if species == "k":
            combined = pd.concat([signal_e, signal_mu], ignore_index=True)
            p_values = combined["K_p"].values
            cosTheta_values = combined["K_cosTheta"].values

        elif species == "pi":
            combined = pd.concat([signal_e, signal_mu], ignore_index=True)
            p_values = np.concatenate(
                [combined["pi1_p"].values, combined["pi2_p"].values])
            cosTheta_values = np.concatenate(
                [combined["pi1_cosTheta"].values, combined["pi2_cosTheta"].values])

        elif species == "e":
            p_values = signal_e["ell_BFbrems_p"].values
            cosTheta_values = signal_e["ell_BFbrems_cosTheta"].values

        elif species == "mu":
            p_values = signal_mu["ell_p"].values
            cosTheta_values = signal_mu["ell_cosTheta"].values

        else:
            raise ValueError(
                f"Unknown species '{species}', expected one of 'k', 'pi', 'e', 'mu'")

        return p_values, cosTheta_values

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def compute(self, table_path, p_bins, cosTheta_bins, species):
        """
        Compute the signal-MC-weighted PIDNN efficiency/fake-rate for a
        given performance table.

        Parameters
        ----------
        table_path : str
            Path to the official PIDNN efficiency/fake-rate CSV table.
        p_bins, cosTheta_bins : list or array
            Bin edges matching the performance table's binning.
        species : str
            Which track(s) to build the phase-space density from.
            One of "k" (kaon), "pi" (pion), "e" (electron), "mu" (muon).
            Note: for a fake-rate table (e.g. "kaon fakes pion"), pass the
            *true* species the track actually is (e.g. species="k"), not
            the PIDNN cut's target species.

        Returns
        -------
        result : dict
            mean, stat_err, syst_err, spread_over_populated_region, n_signal_used
        weighted_table : pd.DataFrame
            Performance table with an extra "weight" column (signal MC
            counts per bin) -- useful for the density-overlay diagnostic plot.
        counts : np.ndarray
            Raw 2D signal MC histogram, shape
            (len(p_bins)-1, len(cosTheta_bins)-1).
        """
        p_bins = np.asarray(sorted(p_bins))
        cosTheta_bins = np.asarray(sorted(cosTheta_bins))

        p_values, cosTheta_values = self._get_species_tracks(species)

        counts, _, _ = np.histogram2d(
            p_values, cosTheta_values, bins=[p_bins, cosTheta_bins])

        n_total = len(p_values)
        n_in_range = counts.sum()
        if n_in_range < n_total:
            frac_out = 1 - n_in_range / n_total
            print(f"[PIDNNPerformanceEvaluator] WARNING: {frac_out:.2%} of "
                  f"'{species}' signal tracks fall outside the table's "
                  f"(p, cosTheta) coverage and were dropped from the "
                  f"weighted average ({table_path}).")

        perf_table = pd.read_csv(table_path)
        weighted_table = attach_weights(perf_table, counts, p_bins, cosTheta_bins)
        result = weighted_average(weighted_table)

        return result, weighted_table, counts

# usage
# evaluator = PIDNNPerformanceEvaluator()

# result_pi, weighted_table_pi, counts_pi = evaluator.compute(
#     "pion_eff_table.csv", p_bins, cosTheta_bins, species="pi")
# result_k, weighted_table_k, counts_k = evaluator.compute(
#     "kaon_eff_table.csv", p_bins, cosTheta_bins, species="k")

import matplotlib.pyplot as plt

def plot_density_over_efficiency(perf_table, counts, p_bins, cosTheta_bins,
                                  result=None,
                                  value_col="data_efficiency",
                                  title="electronIDNN>0.9 efficiency",
                                  cbar_label="Efficiency",
                                  contour_levels=5,
                                  log_density=True,
                                  decimals=4):
    """
    perf_table: the official table (with p_min, cosTheta_min, <value_col> columns),
                or the weighted_table returned by PIDNNPerformanceEvaluator.compute()
    counts:     2D array from np.histogram2d, same (p_bins, cosTheta_bins) binning
    p_bins, cosTheta_bins: bin edges
    result:     optional dict from weighted_average() / evaluator.compute(), with
                keys "mean", "stat_err", "syst_err", "spread_over_populated_region",
                "n_signal_used". If given, a summary text box is added to the plot.
    """
    p_bins = np.asarray(sorted(p_bins))
    cosTheta_bins = np.asarray(sorted(cosTheta_bins))
    nP, nC = len(p_bins) - 1, len(cosTheta_bins) - 1

    df = perf_table.copy()
    df["p_min"] = np.round(df["p_min"].values, decimals)
    df["cosTheta_min"] = np.round(df["cosTheta_min"].values, decimals)

    n_before = len(df)
    df = df.groupby(["p_min", "cosTheta_min"], as_index=False)[value_col].mean()
    if len(df) < n_before:
        print(f"[plot_density_over_efficiency] Note: averaged {n_before} rows down to "
              f"{len(df)} (p_min, cosTheta_min) cells -- table has multiple rows per "
              f"cell (e.g. split by charge); the map below shows their mean.")

    eff_grid = np.full((nP, nC), np.nan)
    p_bins_r = np.round(p_bins, decimals)
    cosTheta_bins_r = np.round(cosTheta_bins, decimals)
    p_idx = np.searchsorted(p_bins_r, df["p_min"].values, side="left")
    c_idx = np.searchsorted(cosTheta_bins_r, df["cosTheta_min"].values, side="left")
    eff_grid[p_idx, c_idx] = df[value_col].values

    fig, ax = plt.subplots(figsize=(7.5, 5.8))

    mesh = ax.pcolormesh(cosTheta_bins, p_bins, eff_grid, cmap="viridis",
                          shading="flat", vmin=0, vmax=1)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(cbar_label)

    p_centers = 0.5 * (p_bins[:-1] + p_bins[1:])
    cosTheta_centers = 0.5 * (cosTheta_bins[:-1] + cosTheta_bins[1:])

    density = counts.astype(float)
    if log_density:
        density = np.ma.masked_where(density <= 0, density)
        density_to_plot = np.ma.log10(density)
        contour_label = "log10(signal MC counts)"
    else:
        density_to_plot = density
        contour_label = "signal MC counts"

    cs = ax.contour(cosTheta_centers, p_centers, density_to_plot,
                     levels=contour_levels, colors="white", linewidths=1.2)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

    ax.contourf(cosTheta_centers, p_centers, density_to_plot,
                levels=contour_levels, cmap="Reds", alpha=0.15)

    ax.set_xlabel(r"$\cos\theta$")
    ax.set_ylabel("Momentum p [GeV/c]")
    ax.set_title(title)

    ax.plot([], [], color="white", lw=1.2, label=contour_label)
    ax.legend(loc="upper right", framealpha=0.6)

    # --- Summary text box: TOP-LEFT corner (high p, low cosTheta) ---
    if result is not None:
        summary_lines = [
            rf"$\varepsilon = {result['mean']:.4f}$",
            rf"$\pm {result['stat_err']:.4f}$ (stat)",
            rf"$\pm {result['syst_err']:.4f}$ (syst)",
            rf"spread = {result['spread_over_populated_region']:.4f}",
            rf"$N_{{\rm signal}} = {result['n_signal_used']:.3g}$",
        ]
        summary_text = "\n".join(summary_lines)

        ax.text(
            0.03, 0.97, summary_text,
            transform=ax.transAxes,
            fontsize=9,
            va="top", ha="left",
            color="black",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.85),
            zorder=10,
        )

    fig.tight_layout()
    return fig, ax
    

################################ dataframe samples ###########################
# from autogluon.tabular import TabularPredictor
import lightgbm as lgb

def add_derived_variables(dfs, definitions=None, overwrite=True):
    """Add derived columns to one or several DataFrames, in place.

    Parameters
    ----------
    dfs : pd.DataFrame, list/tuple of pd.DataFrame, or dict of pd.DataFrame
        The sample(s) to modify. A dict (e.g. the output of classify_mc_dict)
        is handled through its values.
    definitions : dict[str, str], optional
        {new_column: pandas.eval expression}. Defaults to the module-level
        ``derived_variables``.
    overwrite : bool
        If False, columns that already exist are left untouched.
    """
    if definitions is None:
        definitions = derived_variables

    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
    elif isinstance(dfs, dict):
        dfs = list(dfs.values())

    for df in dfs:
        for name, expr in definitions.items():
            if not overwrite and name in df.columns:
                continue
            try:
                df.eval(f'{name} = {expr}', inplace=True)
            except Exception as err:
                raise RuntimeError(
                    f"Could not compute '{name} = {expr}'. Check that the "
                    f"input branches were loaded (filter_branch / columns)."
                ) from err


######################################################################
#                  MVA selection + best candidate selection          #
######################################################################

def apply_mva_bcs_old(df, features, cut, library='lgbm', version='',model=None,bcs='vtx',importance=False):
    # load model
    if library is not None:
        if library=='ag':
            predictor = TabularPredictor.load(f"/home/belle/zhangboy/inclusive_R_D/AutogluonModels/{version}")
            pred = predictor.predict_proba(df, model)
            pred = pred.rename(columns={0: 'sig_prob', 
                                        1: 'fakeD_prob',
                                        2: 'continuum_prob',
                                        3: 'combinatorial_prob'})
            # combine the predict result
            pred['largest_prob'] = pred[['sig_prob','fakeD_prob','continuum_prob','combinatorial_prob']].max(axis=1)
            df_pred = pd.concat([df, pred], axis=1)

        elif library=='lgbm':
            if model == 'multiclass':
                predictor = lgb.Booster(model_file='/home/belle/zhangboy/inclusive_R_D/BDTs/LightGBM/lgbm_multiclass_v3.txt')
                pred_array = predictor.predict(df[features], num_iteration=15) # predictor.best_iteration
                pred = pd.DataFrame(pred_array, columns=['sig_prob','fakeD_prob','continuum_prob','combinatorial_prob'])
                # combine the predict result
                pred['largest_prob'] = pred[['sig_prob','fakeD_prob','continuum_prob','combinatorial_prob']].max(axis=1)
                df_pred = pd.concat([df, pred], axis=1)

            elif model == 'binary':
                predictor = lgb.Booster(model_file='/home/belle/zhangboy/inclusive_R_D/BDTs/LightGBM/lgbm_binary_v1.txt')
                pred_array = predictor.predict(df[features], num_iteration=20)
                pred = pd.DataFrame(pred_array, columns=['data_prob',])
                df_pred = pd.concat([df, pred], axis=1)
        
            if importance: # feature importances
                # Plotting top 10 features based on 'gain'
                ax1=lgb.plot_importance(predictor, importance_type='gain', max_num_features=20, figsize=(18,20))
                ax1.set_ylabel('Features', fontsize=18)
                ax1.tick_params(axis='y', labelsize=18)
                ax1.set_xlabel('Feature importance', fontsize=18)
                ax1.tick_params(axis='x', labelsize=18)
                ax1.set_title("LightGBM Feature Importance (Gain)", fontsize=24)
                
                # Plotting top 10 features based on 'split'
                ax2=lgb.plot_importance(predictor, importance_type='split', max_num_features=20, figsize=(18,20))
                ax2.set_ylabel('Features', fontsize=18)
                ax2.tick_params(axis='y', labelsize=18)
                ax2.set_xlabel('Feature importance', fontsize=18)
                ax2.tick_params(axis='x', labelsize=18)
                ax2.set_title("LightGBM Feature Importance (Split)", fontsize=24)
        
        # apply the MVA cut and BCS
        df_cut=df_pred.query(cut)
        
    else:
        df_cut = df.query(cut)
        
    if bcs=='vtx':
        df_bestSelected=df_cut.loc[df_cut.groupby(['__experiment__','__run__','__event__','__production__'])['B_D_ReChi2'].idxmin()]
    elif bcs=='mva':
        df_bestSelected=df_cut.loc[df_cut.groupby(['__experiment__','__run__','__event__','__production__'])['sig_prob'].idxmax()]
    else:
        df_bestSelected = df_cut

    # check if best selected
    cols = ['__experiment__','__run__','__event__','__production__']
    is_unique = not df_bestSelected.duplicated(subset=cols).any()
    print('Is best selected', is_unique)

    # rename column names
    df_bestSelected = df_bestSelected.rename(columns={"__experiment__": "experiment", '__run__': 'run','__event__': 'event', '__production__': 'production'})
    
    return df_bestSelected
    

def apply_mva_bcs(df, features, cut, library='lgbm', version='', model=None,
                  bcs='vtx', importance=False, perf_eval=False, truth_mode='mu'):
    """
    Apply the MVA prediction, the MVA cut, and the best candidate selection.

    Parameters
    ----------
    perf_eval : bool
        If True, evaluate MVA-cut and BCS performance against MC truth.
        Only meaningful for MC samples.
    truth_mode : str
        'e' or 'mu' -- lepton truth-matching branch used when perf_eval=True.
    """
    # load model
    if library is not None:
        prob_cols = ['sig_prob', 'fakeD_prob', 'continuum_prob', 'combinatorial_prob']
        
        if library=='ag':
            predictor = TabularPredictor.load(f"/home/belle/zhangboy/inclusive_R_D/AutogluonModels/{version}")
            # AutoGluon: predict_proba returns one column per class label 0..3
            pred = predictor.predict_proba(df, model)
            df_pred = df.copy()
            df_pred[prob_cols] = pred[[0, 1, 2, 3]].to_numpy()
            df_pred['largest_prob'] = df_pred[prob_cols].max(axis=1)

        elif library=='lgbm':
            if model == 'multiclass':
                predictor = lgb.Booster(model_file='/home/belle/zhangboy/inclusive_R_D/BDTs/LightGBM/lgbm_multiclass_v3.txt')
                pred_array = predictor.predict(df[features], num_iteration=15) # predictor.best_iteration
                df_pred = df.copy()
                df_pred[prob_cols] = pred_array
                df_pred['largest_prob'] = pred_array.max(axis=1)

            elif model == 'binary':
                predictor = lgb.Booster(model_file='/home/belle/zhangboy/inclusive_R_D/BDTs/LightGBM/lgbm_binary_v1.txt')
                df_pred = df.copy()
                df_pred['data_prob'] = predictor.predict(df[features], num_iteration=20)

            if importance: # feature importances
                # Plotting top features based on 'gain'
                ax1=lgb.plot_importance(predictor, importance_type='gain', max_num_features=20, figsize=(18,20))
                ax1.set_ylabel('Features', fontsize=18)
                ax1.tick_params(axis='y', labelsize=18)
                ax1.set_xlabel('Feature importance', fontsize=18)
                ax1.tick_params(axis='x', labelsize=18)
                ax1.set_title("LightGBM Feature Importance (Gain)", fontsize=24)

                # Plotting top features based on 'split'
                ax2=lgb.plot_importance(predictor, importance_type='split', max_num_features=20, figsize=(18,20))
                ax2.set_ylabel('Features', fontsize=18)
                ax2.tick_params(axis='y', labelsize=18)
                ax2.set_xlabel('Feature importance', fontsize=18)
                ax2.tick_params(axis='x', labelsize=18)
                ax2.set_title("LightGBM Feature Importance (Split)", fontsize=24)

        df_before_mva_cut = df_pred

    else:
        df_before_mva_cut = df

    # --- MVA cut performance (candidate-level, MC only) ---
    if perf_eval:
        mva_cut_metrics = mva_cut_perf_metrics(df_before_mva_cut, cut, mode=truth_mode)
        _print_mva_cut_metrics(mva_cut_metrics, cut)

    # apply the MVA cut
    df_cut = df_before_mva_cut.query(cut)

    # print # candidates before BCS (no truth needed -- works on data too)
    group_cols = ['__experiment__', '__run__', '__event__', '__production__']

    n_cands_per_event = df_cut.groupby(group_cols).size()

    multiplicity_distribution = n_cands_per_event.value_counts().sort_index()
    avg_n_candidates = n_cands_per_event.mean()
    frac_multi_candidate_events = (n_cands_per_event > 1).mean()

    print(f'Event multiplicity distribution:\n{multiplicity_distribution}')
    print(f"Average candidates/event before BCS: {avg_n_candidates:.3f}")
    print(f"Fraction of events with >1 candidate: {frac_multi_candidate_events:.3%}")

    if bcs=='vtx':
        df_bestSelected=df_cut.loc[df_cut.groupby(group_cols)['B_D_ReChi2'].idxmin()]
    elif bcs=='mva':
        df_bestSelected=df_cut.loc[df_cut.groupby(group_cols)['sig_prob'].idxmax()]
    else:
        df_bestSelected = df_cut

    # check if best selected
    is_unique = not df_bestSelected.duplicated(subset=group_cols).any()
    print('Is best selected', is_unique)

    # --- BCS performance (event-level, MC only) ---
    if perf_eval:
        bcs_metrics = bcs_correct_pick_metrics(df_cut, df_bestSelected, group_cols, mode=truth_mode)
        _print_bcs_metrics(bcs_metrics, bcs)

    # rename column names -- done last, since the BCS evaluation above
    # still relies on the original group_cols names
    # df_bestSelected = df_bestSelected.rename(columns={"__experiment__": "experiment", '__run__': 'run',
    #                                                   '__event__': 'event', '__production__': 'production'})

    return df_bestSelected


def mva_cut_perf_metrics(df_before_cut, cut, mode='mu',
                         bkg_categories=('fakeD', 'continuum', 'combinatorial', 'hadronicB_secondaryL')):
    """
    Evaluate the MVA (BDT-output) selection cut's classification
    performance against truth, at the candidate level, before BCS.

    Parameters
    ----------
    df_before_cut : DataFrame
        All candidates prior to the MVA cut (df_pred, or df if no MVA model
        was applied).
    cut : str
        Query string defining the MVA selection, e.g. 'sig_prob > 0.5'.
    mode : str
        'e' or 'mu' -- selects the lepton truth-matching branch.
    bkg_categories : tuple of str
        Keys into get_truth_categories(mode) identifying the background
        truth categories to report fake rates for -- the ones the MVA is
        actually meant to suppress.

    Returns
    -------
    dict with signal efficiency and one fake rate per entry in bkg_categories.
    """
    truth_categories = get_truth_categories(mode)

    df_before_cut = df_before_cut.copy()
    df_before_cut['passes_mva_cut'] = df_before_cut.eval(cut)

    # signal efficiency
    df_before_cut['is_true_signal'] = df_before_cut.eval(truth_categories['B2D_tau'])
    n_signal_total = df_before_cut['is_true_signal'].sum()
    n_signal_pass = (df_before_cut['is_true_signal'] & df_before_cut['passes_mva_cut']).sum()

    metrics = {
        'mva_cut_efficiency': n_signal_pass / n_signal_total if n_signal_total else float('nan'),
        'n_signal_candidates_total': int(n_signal_total),
        'n_signal_candidates_passing': int(n_signal_pass),
    }

    # per-category fake rates
    for bkg in bkg_categories:
        is_bkg = df_before_cut.eval(truth_categories[bkg])
        n_bkg_total = is_bkg.sum()
        n_bkg_pass = (is_bkg & df_before_cut['passes_mva_cut']).sum()

        metrics[f'{bkg}_fake_rate'] = n_bkg_pass / n_bkg_total if n_bkg_total else float('nan')
        metrics[f'n_{bkg}_candidates_total'] = int(n_bkg_total)
        metrics[f'n_{bkg}_candidates_passing'] = int(n_bkg_pass)

    return metrics


def bcs_correct_pick_metrics(df_cut, df_best, group_cols, mode='mu'):
    """
    Evaluate BCS performance against the canonical B0 -> D+ tau- nu truth
    category ('B2D_tau'), as defined centrally in get_truth_categories()
    (this module).

    The headline correct-pick rate is restricted to events that (a) contain
    a truth-matched candidate and (b) have more than one candidate after the
    MVA cut. Single-candidate events are excluded because BCS trivially
    picks the true candidate there, which would inflate the rate without
    reflecting the BCS variable's discriminating power. The inclusive rate
    over all events with a true candidate is kept for reference.

    Parameters
    ----------
    df_cut : DataFrame
        All candidates surviving the MVA cut, before BCS.
    df_best : DataFrame
        One candidate per event, after BCS (e.g. df_bestSelected),
        still carrying the original group_cols names.
    group_cols : list of str
        Columns identifying a unique event, e.g.
        ['__experiment__', '__run__', '__event__', '__production__'].
    mode : str
        'e' or 'mu' -- selects the lepton truth-matching branch.

    Returns
    -------
    dict of performance numbers.
    """
    truth_query = get_truth_categories(mode)['B2D_tau']

    # tag every pre-BCS candidate as truth-matched or not
    df_cut = df_cut.copy()
    df_cut['is_true_signal'] = df_cut.eval(truth_query)

    # per event: does *any* candidate match truth?
    has_true_candidate = df_cut.groupby(group_cols)['is_true_signal'].any()
    events_with_true = set(has_true_candidate[has_true_candidate].index)

    # per event: how many candidates compete in the BCS?
    n_cands_per_event = df_cut.groupby(group_cols).size()
    events_multi_cand = set(n_cands_per_event[n_cands_per_event > 1].index)

    # tag the BCS-selected candidates with the same truth definition
    df_best = df_best.copy()
    df_best['is_true_signal'] = df_best.eval(truth_query)
    df_best_idx = df_best.set_index(group_cols)

    # events where a correct answer was available
    mask_true_exists = df_best_idx.index.isin(events_with_true)
    # ... and where BCS actually had to choose
    mask_true_multi = mask_true_exists & df_best_idx.index.isin(events_multi_cand)

    n_true_exists = int(mask_true_exists.sum())
    n_true_multi = int(mask_true_multi.sum())

    # headline: BCS correct-pick rate on multi-candidate events only
    bcs_correct_pick_rate = (df_best_idx.loc[mask_true_multi, 'is_true_signal'].mean()
                             if n_true_multi else float('nan'))

    # reference: inclusive rate, includes trivially-correct single-candidate events
    bcs_correct_pick_rate_inclusive = (df_best_idx.loc[mask_true_exists, 'is_true_signal'].mean()
                                       if n_true_exists else float('nan'))

    # for reference only -- NOT a BCS-performance number, folds in preselection acceptance
    overall_selected_purity = df_best_idx['is_true_signal'].mean()

    n_events_total = len(df_best_idx)

    return {
        'bcs_correct_pick_rate': bcs_correct_pick_rate,
        'bcs_correct_pick_rate_inclusive': bcs_correct_pick_rate_inclusive,
        'overall_selected_purity': overall_selected_purity,
        'n_events_total': n_events_total,
        'n_events_with_true_candidate': n_true_exists,
        'n_events_with_true_candidate_multi': n_true_multi,
        'frac_events_with_true_candidate': n_true_exists / n_events_total if n_events_total else float('nan'),
        'frac_true_events_multi': n_true_multi / n_true_exists if n_true_exists else float('nan'),
    }


def _print_mva_cut_metrics(metrics, cut, decimals=4):
    """
    Print the output of mva_cut_perf_metrics() as an aligned table:
    one row for signal efficiency, one row per background fake rate.
    Background categories are read from the '<bkg>_fake_rate' keys, so
    the table follows whatever bkg_categories was used to build `metrics`.
    """
    rows = [('signal efficiency (B2D_tau)',
             metrics['mva_cut_efficiency'],
             metrics['n_signal_candidates_passing'],
             metrics['n_signal_candidates_total'])]

    bkg_categories = [key[:-len('_fake_rate')] for key in metrics if key.endswith('_fake_rate')]
    for bkg in bkg_categories:
        rows.append((f'{bkg} fake rate',
                     metrics[f'{bkg}_fake_rate'],
                     metrics[f'n_{bkg}_candidates_passing'],
                     metrics[f'n_{bkg}_candidates_total']))

    label_width = max(len(row[0]) for row in rows)
    header = f"{'Category':<{label_width}}  {'Rate':>8}  {'Passing':>10}  {'Total':>10}"

    print(f"\nMVA cut performance  [cut: {cut}]")
    print(header)
    print('-' * len(header))
    for label, rate, n_pass, n_total in rows:
        print(f"{label:<{label_width}}  {rate:>8.{decimals}f}  {n_pass:>10,d}  {n_total:>10,d}")


def _print_bcs_metrics(metrics, bcs, decimals=4):
    """
    Print the output of bcs_correct_pick_metrics() in aligned form,
    keeping the BCS-performance numbers visually separate from the
    reference-only sample-composition number.
    """
    n_total = metrics['n_events_total']
    n_true = metrics['n_events_with_true_candidate']
    n_true_multi = metrics['n_events_with_true_candidate_multi']

    print(f"\nBCS performance  [bcs: {bcs}]")
    print(f"  Correct-pick rate (true cand., >1 cand.):         {metrics['bcs_correct_pick_rate']:.{decimals}f}")
    print(f"  Correct-pick rate (true cand., inclusive):        {metrics['bcs_correct_pick_rate_inclusive']:.{decimals}f}")
    print(f"  Overall selected purity (reference only):         {metrics['overall_selected_purity']:.{decimals}f}")
    print(f"  Events after BCS:                                 {n_total:,d}")
    print(f"  Events with a true candidate:                     {n_true:,d} "
          f"({metrics['frac_events_with_true_candidate']:.2%} of all events)")
    print(f"    ... of which with >1 candidate:                 {n_true_multi:,d} "
          f"({metrics['frac_true_events_multi']:.2%})")


######################################################################
#                       MC truth classification                      #
######################################################################

def classify_mc_dict(df, mode, template=True) -> dict:
    samples = {}
    cats = get_truth_categories(mode)

    ######################### Apply selection ###########################

    # Fake background components:
    samples.update({
        'bkg_fakeD': df.query(cats['fakeD']).copy(),
        'bkg_fakeL': df.query(cats['TDFl']).copy(),
        'bkg_fakeTracks': df.query(cats['fakeTracks']).copy(),
    })

    # True Dl background components:
    bkg_continuum            = df.query(cats['continuum']).copy()
    bkg_combinatorial        = df.query(cats['combinatorial']).copy()
    bkg_hadronicB_secondaryL = df.query(cats['hadronicB_secondaryL']).copy()
    df_signals_all           = df.query(cats['signals']).copy()
    df_TDTl_all              = df.query(cats['TDTl']).copy()

    classified_TDTl_indices = pd.concat([bkg_continuum, bkg_combinatorial,
                                         bkg_hadronicB_secondaryL, df_signals_all]).index

    bkg_other_TDTl = df_TDTl_all.loc[~df_TDTl_all.index.isin(classified_TDTl_indices)].copy()

    samples.update({
        'bkg_continuum': bkg_continuum,
        'bkg_combinatorial': bkg_combinatorial,
        'bkg_hadronicB_secondaryL': bkg_hadronicB_secondaryL,
        'bkg_other_TDTl': bkg_other_TDTl,
    })

    # True Dl signal components:
    D_tau_nu          = df.query(cats['B2D_tau']).copy()
    D_l_nu            = df.query(cats['B2D_ell']).copy()
    Dst_tau_nu        = df.query(cats['B2Dst_tau']).copy()
    Dst_l_nu          = df.query(cats['B2Dst_ell']).copy()
    Dstst_tau_nu      = df.query(cats['B2Dstst_tau']).copy()
    Dstst_l_nu_narrow = df.query(cats['B2Dstst_ell_narrow']).copy()
    Dstst_l_nu_broad  = df.query(cats['B2Dstst_ell_broad']).copy()
    D_l_nu_gap_pi     = df.query(cats['B2D_ell_gap_pi']).copy()
    D_l_nu_gap_eta    = df.query(cats['B2D_ell_gap_eta']).copy()

    classified_signal_indices = pd.concat([D_tau_nu, Dst_tau_nu, D_l_nu,
                                           Dst_l_nu, Dstst_tau_nu,
                                           Dstst_l_nu_narrow,
                                           Dstst_l_nu_broad,
                                           D_l_nu_gap_pi, D_l_nu_gap_eta]).index

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
        # ignore_index=False keeps the original candidate index, as every
        # other category does. Renumbering made this one sample's index
        # collide with unrelated early rows, which broke any index-based
        # bookkeeping over the returned dict and produced duplicate labels
        # when the categories were concatenated.
        r'$D\ell\nu$_gap': pd.concat([D_l_nu_gap_pi, D_l_nu_gap_eta], ignore_index=False),
        'bkg_other_signal': bkg_other_signal,
    })

    # Finally, assign a 'mode' to each sample based on an external mapping (DecayMode_new)
    for name, subset_df in samples.items():
        subset_df['mode'] = DecayMode_new.get(name, -1)

    # sanity check: every candidate in df classified exactly once
    _check_classification_completeness(df, samples)

    return samples


def _check_classification_completeness(df, samples):
    """
    Verify that the truth-category classification in `samples` is a
    partition of `df`: every candidate index appears in exactly one
    category -- none missed (incomplete coverage), none double-counted
    (categories not mutually exclusive). The individual truth queries in
    get_truth_categories() are not guaranteed exhaustive or disjoint by
    construction, so this is checked explicitly rather than assumed.

    Prints a warning naming the offending categories if either condition
    fails; prints a one-line confirmation otherwise.
    """
    all_indices = np.concatenate([subset_df.index.values for subset_df in samples.values()])
    index_counts = pd.Series(all_indices).value_counts()

    n_total = len(df)
    n_duplicated = int((index_counts > 1).sum())
    missing = df.index.difference(pd.Index(all_indices))

    if n_duplicated > 0:
        dup_ids = set(index_counts[index_counts > 1].index)
        print(f"WARNING: {n_duplicated} candidate(s) assigned to more than one truth category:")
        for name, subset_df in samples.items():
            n_overlap = subset_df.index.isin(dup_ids).sum()
            if n_overlap > 0:
                print(f"  -> {n_overlap} duplicate candidate(s) found in '{name}'")

    if len(missing) > 0:
        print(f"WARNING: {len(missing)} candidate(s) out of {n_total} not assigned to any truth category (uncovered by classification).")

    if n_duplicated == 0 and len(missing) == 0:
        print(f"Classification check passed: all {n_total} candidates uniquely classified into {len(samples)} categories.")
    

######################################################################
# BBbar background reweighting + hadronic B decay classification     #
######################################################################
# The decay classification and mode-replacement factors are invariant
# during a fit.  prepare_bbbar_reweighting() computes them once; only the
# cheap category lookup in apply_bbbar_weights() belongs in an optimizer
# loop.  (Merged from the former standalone bbbar_reweighting.py.)

from collections.abc import Mapping

BBBAR_SAMPLE_NAMES = frozenset({"bkg_combinatorial", "bkg_hadronicB_secondaryL"})
COMBINATORIAL_SAMPLE_NAME = "bkg_combinatorial"
PDG_CORRECTIONS = {
    10431 * 411: 0.00105 / 0.00737857,
    10431 * 413: 0.0015 / 0.01768717,
    10431 * 421: 0.00079 / 0.00761729,
    10431 * 423: 0.0009 / 0.01706301,
}


def _compute_dmID(df: pd.DataFrame, daughter_cols: list[str]) -> pd.Series:
    """Decay-mode ID of a B decay: |product of the B-daughter PDG codes|.

    ``daughter_cols`` holds the neutral-B and charged-B daughter columns of
    one side (D or lepton).  Unfilled slots are -1, so they only affect the
    sign, which the absolute value removes.
    """
    return df[daughter_cols].astype("int64").prod(axis=1).abs()


def add_dmID(data, combinatorial: bool | None = None, verbose: bool = False):
    """Add the decay-mode ID column(s) to MC events, in place.

    The decay-mode ID is the absolute product of the PDG codes of the B
    daughters, the quantity compared against ``measured_pdg_list`` and used
    as key in ``hadronicB_replacement_map``.

    * Combinatorial background: the D and the lepton come from different B
      decays, so two columns are added, ``D_dmID`` (from the D-side daughter
      columns ``combinatorial_vars_D``) and ``ell_dmID`` (from the
      lepton-side columns ``combinatorial_vars_ell``).
    * All other MC samples: the D and the lepton come from the same B, so a
      single column ``dmID`` is added, built from the D-side columns.

    Parameters
    ----------
    data : dict[str, pd.DataFrame] or pd.DataFrame
        Either the dictionary returned by ``classify_mc_dict`` (the sample
        named ``'bkg_combinatorial'`` gets the two-column treatment), or a
        single DataFrame.
    combinatorial : bool, optional
        Only used for a single DataFrame: set True if it holds combinatorial
        background.  Ignored for a dictionary unless given explicitly, in
        which case it overrides the name-based choice for every sample.
    verbose : bool
        Print which columns were added to which sample.

    Returns
    -------
    The same object that was passed in (modified in place).
    """
    if isinstance(data, pd.DataFrame):
        items = [(None, data)]
    else:
        items = list(data.items())

    for name, df in items:
        if combinatorial is None:
            is_comb = (name == COMBINATORIAL_SAMPLE_NAME)
        else:
            is_comb = combinatorial
        targets = (
            {"D_dmID": combinatorial_vars_D, "ell_dmID": combinatorial_vars_ell}
            if is_comb
            else {"dmID": combinatorial_vars_D}
        )
        for column, daughter_cols in targets.items():
            missing = [c for c in daughter_cols if c not in df.columns]
            if missing:
                warnings.warn(
                    f"add_dmID: sample {name!r} lacks B-daughter PDG columns "
                    f"(e.g. {missing[0]!r}); '{column}' not added.",
                    stacklevel=2,
                )
                continue
            df[column] = _compute_dmID(df, daughter_cols)
            if verbose:
                print(f"add_dmID: added '{column}' to {name if name is not None else 'DataFrame'}")
    return data


def _prepare_side(
    df: pd.DataFrame,
    *,
    prefix: str,
    combinatorial_vars: list[str],
    neutral_cols: list[str],
    charged_cols: list[str],
    replacement_map: Mapping[int, int],
    replacement_use_pdg_corr: bool,
    cap_nbody: int | None,
    verbose: bool,
) -> None:
    """Add parameter-independent classification columns to ``df`` in place."""
    df[f"{prefix}dmID"] = _compute_dmID(df, combinatorial_vars)
    df[f"{prefix}mask_sl"] = df[combinatorial_vars].isin(leptons).any(axis=1)
    neutral_n = (~df[neutral_cols].isin([-1, 22])).sum(axis=1)
    charged_n = (~df[charged_cols].isin([-1, 22])).sum(axis=1)
    df[f"{prefix}n_daughters"] = np.maximum(neutral_n, charged_n).astype(int)
    df[f"{prefix}is_measured"] = df[f"{prefix}dmID"].isin(measured_pdg_list)

    n_daughters = df[f"{prefix}n_daughters"].clip(lower=2)
    if cap_nbody is None:
        labels = n_daughters.astype(str) + "-body"
    else:
        labels = np.where(
            n_daughters >= cap_nbody,
            f"{cap_nbody}+-body",
            n_daughters.astype(str) + "-body",
        )
    hadronic = np.where(
        df[f"{prefix}is_measured"],
        "BBbar_measured_hadronic",
        "BBbar_unmeasured:" + labels,
    )
    df[f"{prefix}category"] = np.where(
        df[f"{prefix}mask_sl"], "BBbar_semileptonic", hadronic
    )
    mapped_correction = df[f"{prefix}dmID"].map(PDG_CORRECTIONS).fillna(1.0)
    df[f"{prefix}pdg_corr"] = np.where(
        df[f"{prefix}is_measured"], mapped_correction, 1.0
    )

    replacement_column = f"{prefix}replacement_w"
    df[replacement_column] = 1.0
    count_base = (
        df[f"{prefix}pdg_corr"]
        if replacement_use_pdg_corr
        else pd.Series(1.0, index=df.index)
    )
    new_to_old: dict[int, list[int]] = {}
    for old_dmid, new_dmid in replacement_map.items():
        new_to_old.setdefault(abs(int(new_dmid)), []).append(abs(int(old_dmid)))
    for new_dmid, old_dmids in new_to_old.items():
        old_mask = df[f"{prefix}dmID"].isin(old_dmids)
        new_mask = df[f"{prefix}dmID"] == new_dmid
        n_old = count_base[old_mask].sum()
        n_new = count_base[new_mask].sum()
        df.loc[old_mask, replacement_column] = 0.0
        if n_new <= 0:
            if verbose:
                print(f"[{prefix}] cannot transfer {old_dmids} -> {new_dmid}: n_new=0")
            continue
        factor = (n_old + n_new) / n_new
        df.loc[new_mask, replacement_column] *= factor
        if verbose:
            print(
                f"[{prefix}] replacement {old_dmids} -> {new_dmid}: "
                f"N_old={n_old:.3f}, N_new={n_new:.3f}, R={factor:.3f}"
            )


def prepare_bbbar_reweighting(
    samples: Mapping[str, pd.DataFrame],
    *,
    weight_ell_side: bool = False,
    cap_nbody: int | None = None,
    D_replacement_map: Mapping[int, int] | None = None,
    ell_replacement_map: Mapping[int, int] | None = None,
    replacement_use_pdg_corr: bool = False,
    copy: bool = True,
    verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    """Classify BBbar events and calculate invariant correction factors once.

    By default the returned dictionary and DataFrames are copies.  Set
    ``copy=False`` to opt into in-place preparation for memory-heavy workflows.
    """
    prepared = {
        name: (df.copy() if copy else df)
        for name, df in samples.items()
    }
    for name, df in prepared.items():
        if name not in BBBAR_SAMPLE_NAMES:
            continue
        _prepare_side(
            df,
            prefix="D_",
            combinatorial_vars=combinatorial_vars_D,
            neutral_cols=neutral_cols_D,
            charged_cols=charged_cols_D,
            replacement_map=D_replacement_map or {},
            replacement_use_pdg_corr=replacement_use_pdg_corr,
            cap_nbody=cap_nbody,
            verbose=verbose,
        )
        if name == COMBINATORIAL_SAMPLE_NAME and weight_ell_side:
            _prepare_side(
                df,
                prefix="ell_",
                combinatorial_vars=combinatorial_vars_ell,
                neutral_cols=neutral_cols_ell,
                charged_cols=charged_cols_ell,
                replacement_map=ell_replacement_map or {},
                replacement_use_pdg_corr=replacement_use_pdg_corr,
                cap_nbody=cap_nbody,
                verbose=verbose,
            )
    return prepared


def apply_bbbar_weights(
    prepared_samples: Mapping[str, pd.DataFrame],
    weight_map: Mapping[str, float],
    *,
    out_weight_col: str = "BB_weight",
    weight_ell_side: bool = False,
    copy: bool = True,
    warn_missing_weight_keys: bool = True,
) -> dict[str, pd.DataFrame]:
    """Apply category weights to already-prepared samples using vectorized maps."""
    weighted = {
        name: (df.copy() if copy else df)
        for name, df in prepared_samples.items()
    }
    warned: set[str] = set()
    for name, df in weighted.items():
        df[out_weight_col] = 1.0
        if name not in BBBAR_SAMPLE_NAMES:
            continue
        prefixes = ["D_"]
        if name == COMBINATORIAL_SAMPLE_NAME and weight_ell_side:
            prefixes.append("ell_")
        for prefix in prefixes:
            category_column = f"{prefix}category"
            if category_column not in df:
                raise ValueError(
                    "BBbar samples have not been prepared; call "
                    "prepare_bbbar_reweighting() first."
                )
            missing = set(pd.unique(df[category_column])) - set(weight_map)
            new_missing = missing - warned
            if new_missing and warn_missing_weight_keys:
                warnings.warn(
                    f"Missing BBbar weights default to 1.0: {sorted(new_missing)}",
                    stacklevel=2,
                )
                warned.update(new_missing)
            manual = df[category_column].map(weight_map).fillna(1.0).astype(float)
            df[f"{prefix}manual_w"] = manual
            side_weight = (
                df[f"{prefix}pdg_corr"] * manual * df[f"{prefix}replacement_w"]
            )
            df[f"{prefix}side_weight"] = side_weight
            df[out_weight_col] *= side_weight
    return weighted


def reweight_BBbar_background(
    samples: dict[str, pd.DataFrame],
    weight_map: dict[str, float],
    out_weight_col: str = "BB_weight",
    weight_ell_side: bool = False,
    verbose: bool = False,
    cap_nbody: int | None = None,
    warn_missing_weight_keys: bool = True,
    D_replacement_map: dict[int, int] | None = None,
    ell_replacement_map: dict[int, int] | None = None,
    replacement_use_pdg_corr: bool = False,
):
    """Prepare and weight BBbar samples in place (legacy compatibility API).

    New fitting code should call ``prepare_bbbar_reweighting`` once and
    ``apply_bbbar_weights`` for each parameter point.  This wrapper
    preserves the historical structural mutation when ``weight_ell_side`` is
    false.
    """
    prepared = prepare_bbbar_reweighting(
        samples,
        weight_ell_side=weight_ell_side,
        cap_nbody=cap_nbody,
        D_replacement_map=D_replacement_map,
        ell_replacement_map=ell_replacement_map,
        replacement_use_pdg_corr=replacement_use_pdg_corr,
        copy=False,
        verbose=verbose,
    )
    weighted = apply_bbbar_weights(
        prepared,
        weight_map,
        out_weight_col=out_weight_col,
        weight_ell_side=weight_ell_side,
        copy=False,
        warn_missing_weight_keys=warn_missing_weight_keys,
    )
    if not weight_ell_side:
        combined = pd.concat(
            [weighted["bkg_combinatorial"], weighted["bkg_hadronicB_secondaryL"]]
        )
        by_category = dict(tuple(combined.groupby("D_category")))
        weighted.pop("bkg_combinatorial")
        weighted.pop("bkg_hadronicB_secondaryL")
        weighted.update(by_category)
    return weighted


def classify_measured_modes(df, base='D',corr_col_name='BF_corr_w_D'):
    # decay-mode ID column: 'dmID', or the per-side '<base>_dmID' that
    # add_dmID() writes for the combinatorial background
    dmID_col = "dmID" if "dmID" in df.columns else f"{base}_dmID"
    if dmID_col not in df.columns:
        raise KeyError(f"No 'dmID' or '{base}_dmID' column found; call add_dmID() first.")
    # number of daughters of neutral and charged B
    if base=='D':
        neutral_n = (~df[neutral_cols_D].isin([-1, 22])).sum(axis=1) # count # of B daughters, excluding photons
        charged_n = (~df[charged_cols_D].isin([-1, 22])).sum(axis=1) # cautions: this accidentally excludes modes like B->K gamma gamma
        # neutral_n = (df[neutral_cols_D].ne(-1)).sum(axis=1)
        # charged_n = (df[charged_cols_D].ne(-1)).sum(axis=1)
    elif base=='ell':
        neutral_n = (~df[neutral_cols_ell].isin([-1, 22])).sum(axis=1)
        charged_n = (~df[charged_cols_ell].isin([-1, 22])).sum(axis=1)
    
    # define important variables
    # df["B_type"] = np.where(neutral_n >= charged_n, "B0", "B+")
    df["dmIsMeasured"] = df[dmID_col].isin(measured_pdg_list)
    df["n_daughters"] = np.where(neutral_n >= charged_n, neutral_n, charged_n)
    df["pie_category"] = np.where(df["dmIsMeasured"],'measured_modes','unmeasured:'+df["n_daughters"].astype(str)+"-body" )
    dfs_by_category = dict(tuple(df.groupby("pie_category")))

    # Compute weighted yields per category
    yields = df.groupby("pie_category", dropna=False)[corr_col_name].sum().sort_values(ascending=False)
    print(yields)

    # plot pie chart
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie( yields.to_numpy(), labels=yields.index.to_list(), autopct="%1.1f%%", startangle=90, )
    ax.set_title(f"MC composition {base=}: measured vs unmeasured (split by n-body)")
    ax.axis("equal")  # make it a circle
    plt.tight_layout()
    plt.show()

    return dfs_by_category


def classify_mc_KL(df):
    df['pie_category'] = np.where(df['nMC_K_L']<=2, 'nMC_KL:'+df['nMC_K_L'].astype(str), 'nMC_KL:2+' )
    dfs_by_nKL = dict(tuple(df.groupby('pie_category')))

    # Compute weighted yields per category
    yields = df.groupby('pie_category', dropna=False)['nMC_K_L'].count().sort_values(ascending=False)
    print(yields)

    # plot pie chart
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie( yields.to_numpy(), labels=yields.index.to_list(), autopct="%1.1f%%", startangle=90, )
    ax.set_title(f"Number of $K_L^0$ in MC")
    ax.axis("equal")  # make it a circle
    plt.tight_layout()
    plt.show()

    return dfs_by_nKL


############################## decay-mode tables (input for the decay-table parser) ##########
#
# One CSV per sample / B charge / side, listing every distinct decay mode of the ancestor B of
# the D (side 'D') or of the lepton (side 'ell'), with its candidate count. These are the input
# for matching MC decays to the lines of DECAY_BELLE2.DEC (step B0 of
# docs/BDT_OPTIMIZATION_AND_BBBAR_PLAN.md).
#
# * Signed PDG codes are kept: abs() would merge different decays, e.g. B+ -> D- pi+ pi+ and
#   B+ -> D+ pi- pi+, and loses the B flavor (sum of daughter charges) needed to match the
#   right block of the decay table.
# * Each mode is stored in a canonical order (daughter_0..6: by |PDG|, particle before
#   antiparticle, empty slots last) and in the order found in the event (raw_0..6). EvtGen
#   writes the daughters in the order of the .dec line, PYTHIA does not, so a mode seen in more
#   than one order has PYTHIA contributions (a single order can still be PYTHIA by chance).
# * -1 marks an empty slot. QUARK_LIKE_PDG flags quark/cluster/string codes, which would make
#   -1 ambiguous with the anti-d quark (none were found in the run1 e comb sample).
# * The B daughter lists have 7 slots; 'truncated' marks lists with the 7th slot filled, where
#   the decay may have had more daughters.

QUARK_LIKE_PDG = (set(range(-8, 9)) - {0}) | {91, 92, -91, -92}   # quarks, cluster, string


def decay_mode_table(df, side, b_code, out_csv=None, verbose=True, weight_col=None):
    """Distinct decay modes of the ancestor B of the D (side='D') or of the lepton (side='ell').

    Parameters
    ----------
    df : DataFrame with the B-daughter columns '<side>_<b_code>_<i>_daughterPDG', i = 0..6.
    side : 'D' or 'ell'.
    b_code : 521 (charged B) or 511 (neutral B).
    out_csv : optional path; the table is written there.
    weight_col : optional column of candidate weights; its sum per row is written to 'sum_w'.
        Without it, 'sum_w' equals 'n_candidates'.

    Returns
    -------
    DataFrame, one row per (canonical mode, raw daughter order):
    daughter_0..6 (canonical order), raw_0..6 (order in the event), n_daughters (photons
    included), truncated (7th slot filled), n_candidates, sum_w, mode_total (candidates of the
    canonical mode), n_orders (distinct raw orders of the canonical mode). Sorted by mode_total,
    then by mode, then by n_candidates; the order does not depend on the order of the input.
    """
    cols = [f'{side}_{b_code}_{i}_daughterPDG' for i in range(7)]
    d_float = df[cols].to_numpy(dtype=float)
    if not np.isfinite(d_float).all():
        raise ValueError(f'{side}_{b_code}_*_daughterPDG contains NaN; empty slots must be -1')
    d_all = d_float.astype(np.int64)
    filled = (d_all != -1).any(axis=1)
    d = d_all[filled]
    w = np.ones(len(d)) if weight_col is None else df[weight_col].to_numpy(dtype=float)[filled]
    if not np.isfinite(w).all():
        raise ValueError(f'{weight_col}: {(~np.isfinite(w)).sum()} of {len(w)} candidates have a '
                         f'non-finite weight')

    if verbose:
        gap = ((d[:, :-1] == -1) & (d[:, 1:] != -1)).any(axis=1)
        quark = np.isin(d, list(QUARK_LIKE_PDG - {-1})).any(axis=1)
        print(f'{out_csv or f"{side}_{b_code}"}: {filled.sum()} candidates; -1 before a filled slot: '
              f'{gap.sum()}; quark/string codes: {quark.sum()}; 7th slot filled: {(d[:, -1] != -1).sum()}')

    # canonical order: by |PDG|, particle before antiparticle; empty slots last
    key = np.where(d == -1, np.iinfo(np.int64).max, np.abs(d) * 2 + (d < 0))
    canon = np.take_along_axis(d, np.argsort(key, axis=1, kind='stable'), axis=1)

    mode_cols = [f'daughter_{i}' for i in range(7)]
    raw_cols = [f'raw_{i}' for i in range(7)]
    out = pd.DataFrame(np.hstack([canon, d]), columns=mode_cols + raw_cols)
    out['n_daughters'] = (d != -1).sum(axis=1)
    out['truncated'] = d[:, -1] != -1
    out['_w'] = w

    by = mode_cols + raw_cols + ['n_daughters', 'truncated']
    counts = (out.groupby(by, sort=False)['_w'].agg(n_candidates='size', sum_w='sum').reset_index())
    grp = counts.groupby(mode_cols)['n_candidates']
    counts['mode_total'] = grp.transform('sum')
    counts['n_orders'] = grp.transform('size')
    # largest mode first; the rows of one mode stay together, its most frequent order first
    counts = counts.sort_values(['mode_total'] + mode_cols + ['n_candidates'] + raw_cols,
                                ascending=False, ignore_index=True)

    if out_csv is not None:
        import os
        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        counts.to_csv(out_csv, index=False)
    if verbose:
        modes = counts.drop_duplicates(mode_cols)
        print(f'    {len(modes)} distinct modes, {(modes["n_orders"] > 1).sum()} of them seen in more '
              f'than one order')
    return counts


def b_ancestry_check(df, side, label=''):
    """Candidates whose D (side='D') or lepton (side='ell') ancestor B is charged, neutral, both
    or neither. Returns the four counts as a dict and prints them."""
    c = df[f'{side}_521_0_daughterPDG'] != -1
    n = df[f'{side}_511_0_daughterPDG'] != -1
    res = dict(charged=int((c & ~n).sum()), neutral=int((n & ~c).sum()),
               both=int((c & n).sum()), neither=int((~c & ~n).sum()))
    print(f'{label} {side} side: ' + ', '.join(f'{k} {v}' for k, v in res.items()))
    return res


def export_bbbar_decay_mode_tables(samples, tag, out_dir='csv_output', selection='',
                                   weight_col=None, overwrite=False):
    """Write the decay-mode table of the BBbar background of one sample and run the sanity checks.

    One file per sample: '<out_dir>/bbbar_modes_<tag>.csv'. It holds up to six tables, told apart
    by the columns `component`, `side` and `b_abs`: comb {charged, neutral} B x {D side, lepton
    side}, and hadB {charged, neutral} (D side only: for hadronicB_secondaryL the D and the lepton
    come from the same B; the check below confirms that the two lists agree).

    Parameters
    ----------
    samples : dict from classify_mc_dict(..., template=False), with 'bkg_combinatorial' and,
        if present, 'bkg_hadronicB_secondaryL' (the wrong-charge sample has almost none).
    tag : short name of the sample, used in the file name and written to the `tag` column.
        Suggested form: '<run>_<channel>_<region>', e.g. 'run1_e_combCR', 'run2_mu_wc_combCR',
        'run2_e_SR'. Letters, digits, '_', '-', '+' and '.' only.
    out_dir : output directory.
    selection : the cut that defines the sample (free text, written to the `selection` column), so
        that the file documents itself.
    weight_col : optional column with the candidate weight (e.g. the PID weight). Its sum goes to
        `sum_w`; without it `sum_w` equals `n_candidates`.
    overwrite : by default an existing file is not replaced, so that two samples can't end up in
        one file name by mistake. Pass True when rerunning the same sample.

    Columns of the file: tag, selection, component ('comb' or 'hadB'), side ('D' or 'ell'),
    b_abs (511 or 521), then the columns of `decay_mode_table`.

    Returns {name: table} with the names comb_{charged,neutral}_{Dside,ellside} and
    hadB_{charged,neutral}.
    """
    import os, re
    if not isinstance(tag, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.+-]*', tag):
        raise ValueError(f"tag {tag!r}: a string of letters, digits, '_', '-', '+' and '.', "
                         f"starting with a letter or digit")
    if not isinstance(selection, str):
        raise ValueError('selection must be a string')
    selection = ' '.join(selection.split())             # one line
    out_csv = os.path.join(out_dir, f'bbbar_modes_{tag}.csv')
    if os.path.exists(out_csv) and not overwrite:
        raise FileExistsError(f'{out_csv} exists; choose another tag or pass overwrite=True')

    comb = samples['bkg_combinatorial']
    hadB = samples.get('bkg_hadronicB_secondaryL')
    has_hadB = hadB is not None and len(hadB) > 0
    if len(comb) == 0:
        raise ValueError('bkg_combinatorial is empty; nothing is written')
    if not has_hadB:
        print('no bkg_hadronicB_secondaryL candidates: only the comb tables are written')
    for side in ('D', 'ell'):
        b_ancestry_check(comb, side, 'comb')
    if has_hadB:
        b_ancestry_check(hadB, 'D', 'hadB')

    spec = [('comb_charged_Dside', 'comb', comb, 'D', 521), ('comb_charged_ellside', 'comb', comb, 'ell', 521),
            ('comb_neutral_Dside', 'comb', comb, 'D', 511), ('comb_neutral_ellside', 'comb', comb, 'ell', 511)]
    if has_hadB:
        spec += [('hadB_charged', 'hadB', hadB, 'D', 521), ('hadB_neutral', 'hadB', hadB, 'D', 511)]
    tables, parts = {}, []
    for name, component, df, side, b in spec:
        print(f'{tag} {name}:')
        t = decay_mode_table(df, side, b, weight_col=weight_col)
        tables[name] = t
        head = pd.DataFrame(dict(tag=tag, selection=selection, component=component, side=side, b_abs=b),
                            index=t.index)
        parts.append(pd.concat([head, t], axis=1))
    os.makedirs(out_dir, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_csv(out_csv, index=False)
    print(f'wrote {out_csv}')

    # hadronicB_secondaryL: D and lepton share one B, so the two daughter lists should agree
    for b in (511, 521):
        if not has_hadB:
            break
        dc = [f'D_{b}_{i}_daughterPDG' for i in range(7)]
        lc = [f'ell_{b}_{i}_daughterPDG' for i in range(7)]
        m = hadB[dc[0]] != -1
        if m.any():
            same = (np.sort(hadB.loc[m, dc].to_numpy(), axis=1)
                    == np.sort(hadB.loc[m, lc].to_numpy(), axis=1)).all(axis=1)
            print(f'hadB B{b}: D and lepton daughter lists identical in {same.mean():.1%} of candidates')
    return tables


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


from matplotlib import gridspec
# region ########### BDT output vs. fit variable decorrelation study ###########
#
# Tools for the analysis-note section showing that the multiclass BDT
# (training variables and classifier outputs) does not sculpt the fit
# variables B0_recMissM2 and p_D_l.
#
# Design choices shared by every function below:
#   * All inputs are unweighted MC; we study MC itself, so no data/MC weights.
#   * Samples are dicts {component_name: DataFrame}, as returned by
#     classify_mc_dict(df, mode, template=False).
#   * Every shape comparison is between two DISJOINT samples (slice vs. its
#     complement, pass vs. fail, SR vs. SB, ...), so the two histograms are
#     statistically independent and the two-sample chi2 is valid.
#   * Shapes are normalised to the number of entries inside the histogram
#     range; entries outside the bins are ignored, as in the fit.
#   * Candidate-level samples (bcs=None) can contain several candidates per
#     event. Those entries are not strictly independent, so chi2 p-values on
#     candidate-level samples are slightly optimistic. Post-BCS samples are free
#     of this caveat.

import re
from scipy.stats import chi2 as _chi2_dist

try:
    import dcor as _dcor  # fast O(n log n) distance correlation for 1D inputs
except ImportError:
    _dcor = None

# Coarse binning for the decorrelation plots (the fit binning is too fine to be
# split into several slices). 2D default = the D_M sideband-channel grid.
DECORR_BINS_1D = {
    'B0_recMissM2': np.linspace(-2.5, 10, 26),
    'p_D_l':        np.linspace(0.4, 4.8, 23),
}
DECORR_BINS_2D = [np.linspace(-2.5, 10, 21), np.linspace(0.4, 4.8, 21)]

DECORR_MAIN_COMPONENTS = [r'$D\tau\nu$', r'$D\ell\nu$', r'$D^\ast\tau\nu$', r'$D^\ast\ell\nu$',
                          r'$D^{\ast\ast}\tau\nu$', r'$D^{\ast\ast}\ell\nu$ + gap']
DECORR_APPENDIX_COMPONENTS = ['bkg_fakeD', 'bkg_continuum', 'bkg_combinatorial',
                              'bkg_hadronicB_secondaryL', 'bkg_fakeL', 'bkg_fakeTracks']

# Sub-components merged into one fit template, as in create_templates_new
DECORR_MERGED_TEMPLATES = {
    r'$D^{\ast\ast}\ell\nu$ + gap': [r'$D^{\ast\ast}\ell\nu$_narrow', r'$D^{\ast\ast}\ell\nu$_broad',
                                    r'$D\ell\nu$_gap_pi', r'$D\ell\nu$_gap_eta'],
}

# Event-ID columns after apply_mva_bcs
EVENT_ID_COLUMNS = ['__experiment__', '__run__', '__event__', '__production__']


############################## sample handling ################################

def prepare_decorrelation_samples(df, mode, d_mass_cut=D_MASS_SR,
                                  extra_cut=Dst_veto_cut, components=None):
    """Apply the D_M window (+ D* veto by default) and split into truth components.

    Returns {component: DataFrame} restricted to `components` (all non-empty
    components if None).
    """
    cuts = [f'({c})' for c in (d_mass_cut, extra_cut) if c]
    df_sel = df.query(' and '.join(cuts)) if cuts else df
    samples = classify_mc_dict(df_sel, mode, template=False)
    names = components if components is not None else list(samples.keys())
    return {name: samples[name] for name in names
            if name in samples and len(samples[name]) > 0}


def split_disjoint(df, cut):
    """Split df into (passing, failing) DataFrames for a query string."""
    if not df.index.is_unique:
        raise ValueError('split_disjoint needs a unique index '
                         '(call reset_index() on the candidate DataFrame).')
    mask = df.index.isin(df.query(cut).index)
    return df.loc[mask], df.loc[~mask]


def merge_fit_templates(samples, merged=DECORR_MERGED_TEMPLATES, keep_parts=False):
    """Merge sub-components into the fit's template grouping.

    Adds a 'subcomponent' column so the composition before/after a cut can be
    checked with merged_df.groupby('subcomponent').size(). keep_parts=True also
    keeps the individual sub-component entries in the returned dict.
    """
    out = dict(samples)
    for name, parts in merged.items():
        present = [p for p in parts if p in samples and len(samples[p]) > 0]
        if not present:
            continue
        out[name] = pd.concat([samples[p].assign(subcomponent=p) for p in present])
        if not keep_parts:
            for p in present:
                out.pop(p, None)
    return out


def split_cut_conditions(cut):
    """Split an 'a and b and c' cut string into its individual conditions.

    Only flat conjunctions are supported (as in lgb_tight/lgb_loose/lgb_comb);
    anything containing 'or', '|' or parentheses raises, because removing one
    term from such a string is ambiguous.
    """
    parts = [p.strip() for p in re.split(r'\s+and\s+|\s*&\s*', cut) if p.strip()]
    for p in parts:
        if re.search(r'\bor\b|\||\(|\)', p):
            raise ValueError(f'Condition "{p}" is not a simple comparison; '
                             'N-1 splitting only supports flat "and" chains.')
    return parts


def n_minus_1_cuts(cut):
    """{condition: cut string with that condition removed (None if nothing left)}."""
    conditions = split_cut_conditions(cut)
    out = {}
    for i, cond in enumerate(conditions):
        others = conditions[:i] + conditions[i + 1:]
        out[cond] = ' and '.join(others) if others else None
    return out


def _condition_name(cond):
    """First variable name in a condition, e.g. 'sig_prob' for 'sig_prob>0.5'."""
    return next(t for t in re.findall(r'[A-Za-z_]\w*', cond) if t not in ('and', 'or', 'not'))


def n_minus_1_base_cuts(cut):
    """{variable: cut string with that variable's condition removed}.

    Used as base_cuts in run_quantile_study to slice one output near the working
    point while the other conditions stay at their nominal values. Raises if a
    variable appears in more than one condition.
    """
    out = {}
    for cond, others in n_minus_1_cuts(cut).items():
        var = _condition_name(cond)
        if var in out:
            raise ValueError(f'{var} appears in more than one condition of the cut.')
        out[var] = others
    return out


def add_bcs_flag(df, rank_var='B_D_ReChi2', event_cols=EVENT_ID_COLUMNS,
                 flag_col='bcs_selected'):
    """Flag the candidate that BCS keeps in each event (minimum rank_var).

    Must be run on ALL candidates of the event (every truth category), before
    classify_mc_dict, because BCS competes truth-matched candidates against
    fake ones. Same ranking as apply_mva_bcs(bcs='vtx').
    """
    df = df.copy()
    valid = df[rank_var].notna()
    best_idx = df.loc[valid].groupby(event_cols)[rank_var].idxmin()
    df[flag_col] = False
    df.loc[best_idx.to_numpy(), flag_col] = True
    return df


############################## correlation measures ###########################

def pearson_with_error(x, y):
    """Pearson r and its approximate standard error (1 - r^2) / sqrt(n - 1)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    r = np.corrcoef(x, y)[0, 1]
    return r, (1 - r**2) / np.sqrt(n - 1)


def _dcor_numpy(x, y):
    """Distance correlation via double-centred distance matrices (O(n^2) memory)."""
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0) - b.mean(axis=1)[:, None] + b.mean()
    dcov2 = (A * B).mean()
    denom = np.sqrt((A * A).mean() * (B * B).mean())
    return float(np.sqrt(max(dcov2, 0) / denom)) if denom > 0 else 0.0


def distance_correlation(x, y, max_n=50000, n_perm=3, seed=0):
    """Distance correlation (0 = independent, 1 = fully dependent) with a null baseline.

    The sample dCor of independent variables is positive (~1/sqrt(n)), so the
    mean dCor after randomly permuting y is returned as `dcor_null`; a value
    compatible with dcor_null means "no measurable dependence".
    Uses the `dcor` package if installed; otherwise a numpy O(n^2) fallback,
    for which the subsample is capped at 4000 entries.

    Returns (dcor, dcor_null, n_used).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    if _dcor is None:
        max_n = min(max_n, 4000)
    if len(x) > max_n:
        pick = rng.choice(len(x), size=max_n, replace=False)
        x, y = x[pick], y[pick]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, len(x)

    def _dc(u, v):
        if _dcor is not None:
            return float(_dcor.distance_correlation(u, v, method='mergesort'))
        return _dcor_numpy(u, v)

    value = _dc(x, y)
    null = np.mean([_dc(x, rng.permutation(y)) for _ in range(n_perm)]) if n_perm > 0 else np.nan
    return value, null, len(x)


def correlation_table(samples, variables, fit_vars=FIT_VARIABLES, components=None,
                      max_n_dcor=50000, n_perm=3, seed=0):
    """Long-format table of Pearson r and distance correlation.

    One row per (component, variable, fit_var) with columns
    n, pearson, pearson_err, dcor, dcor_null, n_dcor.
    Pass variables = training_variables + MVA_OUTPUTS to cover inputs and outputs.
    """
    rows = []
    for comp in (components if components is not None else list(samples.keys())):
        if comp not in samples:
            continue
        df = samples[comp]
        for var in variables:
            for fv in fit_vars:
                sub = df[[var, fv]].dropna()
                r, r_err = pearson_with_error(sub[var], sub[fv])
                dc, dc_null, n_dc = distance_correlation(sub[var], sub[fv],
                                                         max_n=max_n_dcor, n_perm=n_perm, seed=seed)
                rows.append(dict(component=comp, variable=var, fit_var=fv, n=len(sub),
                                 pearson=r, pearson_err=r_err,
                                 dcor=dc, dcor_null=dc_null, n_dcor=n_dc))
    return pd.DataFrame(rows)


############################## shape comparison primitives ####################

def shape_chi2_two_sample(n1, n2, min_count=10, max_rel_err=0.02):
    """Two-sample shape comparison for unweighted histograms: chi2 test + effect sizes.

    chi2 = 1/(N1*N2) * sum_i (N2*n1_i - N1*n2_i)^2 / (n1_i + n2_i),  ndf = nbins - 1
    (the normalisation is free, only shapes are compared). Bins with
    n1_i + n2_i < min_count are pooled into a single extra bin so that the chi2
    approximation holds without throwing events away. Works for 1D or 2D input.

    With millions of entries the chi2 flags even negligible differences, so two
    effect sizes are returned as well (computed on the unpooled histograms):
      tvd         total variation distance 0.5 * sum_i |n1_i/N1 - n2_i/N2|: the
                  fraction of the template that would have to move between bins
                  to turn shape 2 into shape 1 (0 = identical, 1 = disjoint).
      tvd_null    expected tvd from statistical fluctuations alone for the same
                  N1, N2 and same shape; tvd ~ tvd_null means no measurable effect.
      max_rel_dev largest signed relative difference (n1_i/N1) / (n2_i/N2) - 1 over
                  bins whose statistical error on that ratio, sqrt(1/n1_i + 1/n2_i),
                  is <= max_rel_err, so noisy tail bins cannot dominate.
      max_rel_dev_err  statistical error of max_rel_dev. Being the maximum over
                  many bins, |max_rel_dev| is typically ~2-3 x max_rel_dev_err
                  even for identical shapes; judge it against that, not against 0.
    """
    n1 = np.asarray(n1, dtype=float).ravel()
    n2 = np.asarray(n2, dtype=float).ravel()
    N1, N2 = n1.sum(), n2.sum()
    out = dict(chi2=np.nan, ndf=0, p=np.nan, tvd=np.nan, tvd_null=np.nan,
               max_rel_dev=np.nan, max_rel_dev_err=np.nan)
    if N1 == 0 or N2 == 0:
        return out

    # effect sizes on the unpooled histograms
    p1, p2 = n1 / N1, n2 / N2
    out['tvd'] = float(0.5 * np.abs(p1 - p2).sum())
    p_pool = (n1 + n2) / (N1 + N2)
    out['tvd_null'] = float(0.5 * np.sum(np.sqrt(2 / np.pi * p_pool * (1 / N1 + 1 / N2))))
    with np.errstate(divide='ignore'):
        rel_err = np.sqrt(1 / n1 + 1 / n2)
    ok = (n1 > 0) & (n2 > 0) & (rel_err <= max_rel_err)
    if ok.any():
        rel = p1[ok] / p2[ok] - 1
        i = np.argmax(np.abs(rel))
        out['max_rel_dev'] = float(rel[i])
        out['max_rel_dev_err'] = float(rel_err[ok][i] * (1 + rel[i]))

    # chi2 on histograms with sparse bins pooled
    if min_count > 0:
        low = (n1 + n2) < min_count
        if low.any():
            n1 = np.append(n1[~low], n1[low].sum())
            n2 = np.append(n2[~low], n2[low].sum())
    keep = (n1 + n2) > 0
    n1, n2 = n1[keep], n2[keep]
    ndf = len(n1) - 1
    out['ndf'] = int(ndf)
    if ndf >= 1:
        chi2 = np.sum((N2 * n1 - N1 * n2)**2 / (n1 + n2)) / (N1 * N2)
        out['chi2'] = float(chi2)
        out['p'] = float(_chi2_dist.sf(chi2, ndf))
    return out


def shape_pulls(n1, n2, min_count=10):
    """Per-bin pull between normalised shapes: (n1/N1 - n2/N2) / sigma.

    sigma uses the pooled estimate under the same-shape hypothesis,
    sigma^2 = (n1 + n2) / (N1 * N2), so sum(pull^2) equals the unpooled chi2 of
    shape_chi2_two_sample. Bins with n1 + n2 < min_count are set to NaN.
    """
    n1 = np.asarray(n1, dtype=float)
    n2 = np.asarray(n2, dtype=float)
    N1, N2 = n1.sum(), n2.sum()
    tot = n1 + n2
    with np.errstate(divide='ignore', invalid='ignore'):
        pull = (n1 / N1 - n2 / N2) / np.sqrt(tot / (N1 * N2))
    pull[tot < max(min_count, 1)] = np.nan
    return pull


def shape_rel_diff(n1, n2, max_rel_err=0.05):
    """Per-bin relative shape difference (n1/N1) / (n2/N2) - 1.

    Complements shape_pulls at high statistics, where pulls saturate and only
    show where shapes differ: this shows by how much. Bins whose statistical
    error on the ratio, sqrt(1/n1 + 1/n2), exceeds max_rel_err are set to NaN.
    """
    n1 = np.asarray(n1, dtype=float)
    n2 = np.asarray(n2, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = (n1 / n1.sum()) / (n2 / n2.sum()) - 1
        rel_err = np.sqrt(1 / n1 + 1 / n2)
    rel[~(rel_err <= max_rel_err)] = np.nan
    return rel


def _shape_map(h1, h2, map_type, min_count, max_rel_err):
    """(values, colour limit default, colourbar label suffix) for a 2D shape map."""
    if map_type == 'pull':
        return shape_pulls(h1, h2, min_count), 4, 'pull'
    if map_type == 'rel':
        return shape_rel_diff(h1, h2, max_rel_err), 0.2, 'relative difference'
    raise ValueError("map_type must be 'pull' or 'rel'")


def _normalised_shape(counts, bins):
    """Unit-area density and its statistical error from raw counts."""
    counts = np.asarray(counts, dtype=float)
    N = counts.sum()
    widths = np.diff(bins)
    if N == 0:
        return np.zeros_like(counts), np.zeros_like(counts)
    return counts / N / widths, np.sqrt(counts) / N / widths


def _draw_shape(ax, bins, counts, label, color, ls='-'):
    dens, err = _normalised_shape(counts, bins)
    centers = 0.5 * (bins[1:] + bins[:-1])
    ax.stairs(dens, bins, color=color, lw=1.8, ls=ls, label=label)
    ax.errorbar(centers, dens, yerr=err, fmt='none', ecolor=color, lw=1)


def _draw_ratio(ax, bins, counts_num, counts_den, color, den_err=True, x_offset=0.0):
    """Ratio of normalised shapes num/den; den_err=False ignores the denominator error.

    x_offset (fraction of bin width) spreads overlapping markers of several curves.
    """
    d_num, e_num = _normalised_shape(counts_num, bins)
    d_den, e_den = _normalised_shape(counts_den, bins)
    centers = 0.5 * (bins[1:] + bins[:-1])
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = d_num / d_den
        rel2 = (e_num / d_num)**2 + ((e_den / d_den)**2 if den_err else 0)
        err = np.abs(ratio) * np.sqrt(rel2)
    ok = (d_num > 0) & (d_den > 0)
    centers = centers + x_offset * np.diff(bins)
    ax.errorbar(centers[ok], ratio[ok], yerr=err[ok], fmt='o', ms=3, color=color, lw=1)


def _fmt_chi2(res):
    return (f"$\\chi^2$/ndf = {res['chi2']:.1f}/{res['ndf']}, p = {res['p']:.2g}\n"
            f"TVD = {res['tvd']:.4f} (stat. {res['tvd_null']:.4f})")


############################## generic two-sample comparison ##################

def compare_fit_variable_shapes(df_a, df_b, labels=('A', 'B'), fit_vars=FIT_VARIABLES,
                                bins_1d=None, bins_2d=None, fit_bins=None,
                                min_count=10, map_type='pull', map_lim=None,
                                max_rel_err=0.05, title=None, plot=True):
    """Compare fit-variable shapes of two disjoint samples A and B (named by `labels`).

    Panels: one normalised 1D overlay per fit variable (ratio A/B below) and a
    2D map of A vs. B: map_type='pull' (significance per bin, saturates at high
    statistics) or 'rel' (relative difference A/B - 1, bins with ratio error
    above max_rel_err left blank); map_lim sets the symmetric colour limit. If fit_bins=[MM2_bins, p_D_l_bins] (the real fit
    binning) is given, an additional 2D chi2 at that granularity is computed
    (not plotted).

    Returns (fig or None, results) with
    results = {fit_var: chi2 dict, '2d': chi2 dict, '2d_fit_binning': chi2 dict,
               'n_a': ..., 'n_b': ...}.
    """
    bins_1d = bins_1d or DECORR_BINS_1D
    bins_2d = bins_2d or DECORR_BINS_2D
    results = {'n_a': len(df_a), 'n_b': len(df_b)}

    h1 = {}
    for fv in fit_vars:
        ha, _ = np.histogram(df_a[fv], bins=bins_1d[fv])
        hb, _ = np.histogram(df_b[fv], bins=bins_1d[fv])
        h1[fv] = (ha, hb)
        results[fv] = shape_chi2_two_sample(ha, hb, min_count)

    h2a, _, _ = np.histogram2d(df_a[fit_vars[0]], df_a[fit_vars[1]], bins=bins_2d)
    h2b, _, _ = np.histogram2d(df_b[fit_vars[0]], df_b[fit_vars[1]], bins=bins_2d)
    results['2d'] = shape_chi2_two_sample(h2a, h2b, min_count)

    if fit_bins is not None:
        fa, _, _ = np.histogram2d(df_a[fit_vars[0]], df_a[fit_vars[1]], bins=fit_bins)
        fb, _, _ = np.histogram2d(df_b[fit_vars[0]], df_b[fit_vars[1]], bins=fit_bins)
        results['2d_fit_binning'] = shape_chi2_two_sample(fa, fb, min_count)

    if not plot:
        return None, results

    fig = plt.figure(figsize=(6 * (len(fit_vars) + 1), 6))
    gs = gridspec.GridSpec(2, len(fit_vars) + 1, height_ratios=[3, 1], hspace=0.05, wspace=0.3)
    for j, fv in enumerate(fit_vars):
        ax = fig.add_subplot(gs[0, j])
        axr = fig.add_subplot(gs[1, j], sharex=ax)
        ha, hb = h1[fv]
        _draw_shape(ax, bins_1d[fv], ha, f'{labels[0]} (N={len(df_a)})', 'tab:blue')
        _draw_shape(ax, bins_1d[fv], hb, f'{labels[1]} (N={len(df_b)})', 'tab:red', ls='--')
        ax.set_ylabel('normalised density')
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
        ax.legend(fontsize=8, title=_fmt_chi2(results[fv]), title_fontsize=8)
        ax.tick_params(labelbottom=False)
        ax.grid(alpha=0.3)
        _draw_ratio(axr, bins_1d[fv], ha, hb, 'k')
        axr.axhline(1, color='gray', lw=1)
        axr.set_ylim(0.5, 1.5)
        axr.set_ylabel(f'{labels[0]} / {labels[1]}')
        axr.set_xlabel(FIT_VARIABLE_LABELS.get(fv, fv))
        axr.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[:, -1])
    values, lim, what = _shape_map(h2a, h2b, map_type, min_count, max_rel_err)
    lim = map_lim if map_lim is not None else lim
    im = ax2.pcolormesh(bins_2d[0], bins_2d[1], values.T, cmap='RdBu_r', vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=ax2, label=f'{what} ({labels[0]} vs. {labels[1]})')
    ax2.set_xlabel(FIT_VARIABLE_LABELS.get(fit_vars[0], fit_vars[0]))
    ax2.set_ylabel(FIT_VARIABLE_LABELS.get(fit_vars[1], fit_vars[1]))
    ax2.set_title('2D: ' + _fmt_chi2(results['2d']), fontsize=9)

    if title:
        fig.suptitle(title, fontsize=11)
    return fig, results


def _results_to_rows(results, **tags):
    """Flatten a compare_fit_variable_shapes results dict into table rows."""
    rows = []
    for key, res in results.items():
        if isinstance(res, dict):
            rows.append(dict(**tags, test_on=key, n_a=results['n_a'], n_b=results['n_b'], **res))
    return rows


############################## step 1: correlation table ######################

def plot_correlation_heatmap(table, fit_var, metric='pearson', components=None,
                             variables=None, vmax=None, annotate=True, title=None):
    """Heatmap of one metric: rows = variables, columns = components.

    metric: 'pearson' (diverging colour scale) or 'dcor' (sequential; the
    annotation also shows the permutation baseline dcor_null in brackets).
    """
    sub = table[table['fit_var'] == fit_var]
    pivot = sub.pivot(index='variable', columns='component', values=metric)
    null = sub.pivot(index='variable', columns='component', values='dcor_null')
    # keep the order of the input lists (default: order of appearance in the table)
    variables = variables if variables is not None else list(dict.fromkeys(sub['variable']))
    components = components if components is not None else list(dict.fromkeys(sub['component']))
    pivot = pivot.reindex(index=variables, columns=components)
    null = null.reindex(index=variables, columns=components)

    vals = pivot.to_numpy(dtype=float)
    if vmax is None:
        vmax = max(np.nanmax(np.abs(vals)), 0.05)
    if metric == 'pearson':
        cmap, vmin = 'RdBu_r', -vmax
    else:
        cmap, vmin = 'Reds', 0

    fig, ax = plt.subplots(figsize=(1.6 * vals.shape[1] + 4, 0.4 * vals.shape[0] + 2))
    im = ax.imshow(vals, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    fig.colorbar(im, ax=ax, label=metric)
    ax.set_xticks(range(vals.shape[1]), pivot.columns, rotation=30, ha='right')
    ax.set_yticks(range(vals.shape[0]), pivot.index)
    if annotate:
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                if np.isnan(vals[i, j]):
                    continue
                txt = f'{vals[i, j]:.3f}'
                if metric == 'dcor':
                    txt += f'\n({null.to_numpy()[i, j]:.3f})'
                dark = abs(vals[i, j]) > 0.6 * vmax
                ax.text(j, i, txt, ha='center', va='center', fontsize=7,
                        color='white' if dark else 'black')
    ax.set_title(title or f'{metric} with {FIT_VARIABLE_LABELS.get(fit_var, fit_var)}')
    fig.tight_layout()
    return fig


def plot_profile_in_slices(df, x, y, bins_x, cond_var=None, cond_edges=None,
                           min_count=20, xlabel=None, ylabel=None, title=None):
    """Profile plot: mean of y (+- error on the mean) vs. x, one curve per cond_var bin.

    Case study for B0_CMS_cos_angle_0_1: at fixed p_D and p_l, MM2 is linear in
    cos(theta_Dl) with slope -2 p_D p_l, so slicing in p_D_l exposes the
    algebraic dependence that the inclusive correlation coefficient dilutes.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    centers = 0.5 * (bins_x[1:] + bins_x[:-1])
    groups = [(df, 'inclusive')] if cond_var is None else [
        (df[(df[cond_var] >= lo) & (df[cond_var] < hi)], f'{lo:.2f} $\\leq$ {cond_var} < {hi:.2f}')
        for lo, hi in zip(cond_edges[:-1], cond_edges[1:])]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(groups)))
    for (g, label), c in zip(groups, colors):
        idx = np.digitize(g[x], bins_x) - 1
        stats = g.groupby(idx)[y].agg(['mean', 'std', 'count'])
        stats = stats[(stats.index >= 0) & (stats.index < len(centers)) & (stats['count'] >= min_count)]
        ax.errorbar(centers[stats.index], stats['mean'], yerr=stats['std'] / np.sqrt(stats['count']),
                    fmt='o-', ms=4, color=c, label=label)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or f'mean {FIT_VARIABLE_LABELS.get(y, y)}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)
    return fig


############################## step 2: quantile slicing #######################

def _require_finite(df, cols, tag=''):
    """Raise if any of cols contains NaN/inf: every candidate must have valid values."""
    bad = ~np.isfinite(df[cols].to_numpy(dtype=float))
    if bad.any():
        counts = dict(zip(cols, bad.sum(axis=0).tolist()))
        raise ValueError(f'{tag}: NaN/inf found in {counts} out of {len(df)} rows; '
                         'fix the input sample instead of dropping rows.')


def assign_quantile_slices(values, n_slices=5):
    """Equal-population slice index for each entry, plus the slice edges.

    Tied values (e.g. many fakeD_prob ~ 0 for signal) can merge quantiles; the
    number of slices is then reduced and a warning is printed.
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError('assign_quantile_slices needs a non-empty, finite input.')
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_slices + 1)))
    if len(edges) < 2:
        raise ValueError(f'All {len(values)} values are identical ({edges[0]}); cannot slice.')
    if len(edges) - 1 < n_slices:
        print(f'Warning: only {len(edges) - 1} distinct quantile slices (ties in the input).')
    idx = np.clip(np.searchsorted(edges, values, side='right') - 1, 0, len(edges) - 2)
    return idx, edges


def plot_quantile_slices_1d(df, slice_var, n_slices=5, fit_vars=FIT_VARIABLES,
                            bins_1d=None, min_count=10, title=None):
    """Normalised fit-variable shapes in quantile slices of slice_var.

    Top: shape per slice (inclusive shape as dashed black reference).
    Bottom: slice / inclusive (display only; slice error only).
    The chi2 quoted per slice compares the slice with its COMPLEMENT (disjoint).
    Returns (fig, DataFrame of per-slice chi2 results).
    """
    _require_finite(df, [slice_var] + list(fit_vars), tag=slice_var)
    bins_1d = bins_1d or DECORR_BINS_1D
    idx, edges = assign_quantile_slices(df[slice_var], n_slices)
    n_sl = len(edges) - 1
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_sl))

    fig = plt.figure(figsize=(7 * len(fit_vars), 6))
    gs = gridspec.GridSpec(2, len(fit_vars), height_ratios=[3, 1], hspace=0.05, wspace=0.25)
    rows = []
    for j, fv in enumerate(fit_vars):
        b = bins_1d[fv]
        ax = fig.add_subplot(gs[0, j])
        axr = fig.add_subplot(gs[1, j], sharex=ax)
        h_inc, _ = np.histogram(df[fv], bins=b)
        _draw_shape(ax, b, h_inc, 'inclusive', 'k', ls='--')
        for s in range(n_sl):
            in_s = idx == s
            h_s, _ = np.histogram(df.loc[in_s, fv], bins=b)
            h_c, _ = np.histogram(df.loc[~in_s, fv], bins=b)
            res = shape_chi2_two_sample(h_s, h_c, min_count)
            rows.append(dict(slice_var=slice_var, slice=s, lo=edges[s], hi=edges[s + 1],
                             n=int(in_s.sum()), test_on=fv, **res))
            label = (f'[{edges[s]:.3f}, {edges[s + 1]:.3f}]: p={res["p"]:.2g}, '
                     f'TVD={res["tvd"]:.4f} ({res["tvd_null"]:.4f})')
            _draw_shape(ax, b, h_s, label, colors[s])
            _draw_ratio(axr, b, h_s, h_inc, colors[s], den_err=False,
                        x_offset=0.6 * (s / max(n_sl - 1, 1) - 0.5))
        ax.set_ylabel('normalised density')
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
        ax.legend(fontsize=7, title=f'{slice_var} slice: p, TVD (stat.) vs. complement', title_fontsize=7)
        ax.tick_params(labelbottom=False)
        ax.grid(alpha=0.3)
        axr.axhline(1, color='gray', lw=1)
        axr.set_ylim(0.5, 1.5)
        axr.set_ylabel('slice / incl.')
        axr.set_xlabel(FIT_VARIABLE_LABELS.get(fv, fv))
        axr.grid(alpha=0.3)
    fig.suptitle(title or f'Fit-variable shapes in quantile slices of {slice_var}', fontsize=11)
    return fig, pd.DataFrame(rows)


def plot_quantile_slices_2d(df, slice_var, n_slices=5, fit_vars=FIT_VARIABLES,
                            bins_2d=None, min_count=10, map_type='pull', map_lim=None,
                            max_rel_err=0.05, title=None):
    """2D maps (slice vs. complement) of the fit-variable plane, one panel per slice.

    map_type='pull' shows where the shapes differ (saturates at high statistics);
    'rel' shows by how much (slice/complement - 1). See compare_fit_variable_shapes.

    Returns (fig, DataFrame of per-slice 2D chi2 results).
    """
    _require_finite(df, [slice_var] + list(fit_vars), tag=slice_var)
    bins_2d = bins_2d or DECORR_BINS_2D
    idx, edges = assign_quantile_slices(df[slice_var], n_slices)
    n_sl = len(edges) - 1
    fig, axs = plt.subplots(1, n_sl, figsize=(4.2 * n_sl + 1, 4.2), sharey=True,
                            constrained_layout=True)
    axs = np.atleast_1d(axs)
    rows = []
    x, y = df[fit_vars[0]].to_numpy(), df[fit_vars[1]].to_numpy()
    for s in range(n_sl):
        in_s = idx == s
        h_s, _, _ = np.histogram2d(x[in_s], y[in_s], bins=bins_2d)
        h_c, _, _ = np.histogram2d(x[~in_s], y[~in_s], bins=bins_2d)
        res = shape_chi2_two_sample(h_s, h_c, min_count)
        rows.append(dict(slice_var=slice_var, slice=s, lo=edges[s], hi=edges[s + 1],
                         n=int(in_s.sum()), test_on='2d', **res))
        values, lim, what = _shape_map(h_s, h_c, map_type, min_count, max_rel_err)
        lim = map_lim if map_lim is not None else lim
        im = axs[s].pcolormesh(bins_2d[0], bins_2d[1], values.T, cmap='RdBu_r', vmin=-lim, vmax=lim)
        axs[s].set_title(f'[{edges[s]:.3f}, {edges[s + 1]:.3f}]\n' + _fmt_chi2(res), fontsize=8)
        axs[s].set_xlabel(FIT_VARIABLE_LABELS.get(fit_vars[0], fit_vars[0]))
    axs[0].set_ylabel(FIT_VARIABLE_LABELS.get(fit_vars[1], fit_vars[1]))
    fig.colorbar(im, ax=axs, label=f'{what} (slice vs. complement)')
    fig.suptitle(title or f'2D {what} in quantile slices of {slice_var}', fontsize=11)
    return fig, pd.DataFrame(rows)


def plot_fit_variable_correlation_vs_slice(df, slice_var, n_slices=5,
                                           fit_vars=FIT_VARIABLES, title=None):
    """Pearson rho(fit_var_0, fit_var_1) in each quantile slice of slice_var.

    Tests whether the BDT output changes the correlation *between* the two fit
    variables, which a pair of 1D projections cannot show. The grey band is the
    inclusive value +- its error.
    Returns (fig, DataFrame).
    """
    _require_finite(df, [slice_var] + list(fit_vars), tag=slice_var)
    idx, edges = assign_quantile_slices(df[slice_var], n_slices)
    rows = []
    for s in range(len(edges) - 1):
        sub = df.loc[idx == s]
        r, r_err = pearson_with_error(sub[fit_vars[0]], sub[fit_vars[1]])
        rows.append(dict(slice_var=slice_var, slice=s, lo=edges[s], hi=edges[s + 1],
                         median=sub[slice_var].median(), n=len(sub), rho=r, rho_err=r_err))
    res = pd.DataFrame(rows)
    r_inc, r_inc_err = pearson_with_error(df[fit_vars[0]], df[fit_vars[1]])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhspan(r_inc - r_inc_err, r_inc + r_inc_err, color='gray', alpha=0.3,
               label=f'inclusive: {r_inc:.3f} $\\pm$ {r_inc_err:.3f}')
    ax.axhline(r_inc, color='gray', lw=1)
    ax.errorbar(res['median'], res['rho'], yerr=res['rho_err'],
                xerr=[res['median'] - res['lo'], res['hi'] - res['median']],
                fmt='o', color='k', label='quantile slices')
    ax.set_xlabel(slice_var)
    ax.set_ylabel(f'Pearson $\\rho$({fit_vars[0]}, {fit_vars[1]})')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title(title or f'Fit-variable correlation vs. {slice_var}', fontsize=11)
    return fig, res


QUANTILE_PLOTS = ('1d', '2d', '2d_rel', 'rho')


def run_quantile_study(samples, slice_vars=MVA_OUTPUTS, components=None, n_slices=5,
                       base_cuts=None, fit_vars=FIT_VARIABLES, bins_1d=None, bins_2d=None,
                       min_count=10, plots=QUANTILE_PLOTS, save_dir=None, show=True):
    """Run the quantile-slice plots for every (component, slice_var).

    plots: which outputs to make, any of '1d' (1D overlays), '2d' (2D pull
    maps), '2d_rel' (2D relative-difference maps), 'rho' (fit-variable
    correlation per slice); e.g. plots=('1d',) for the 1D overlays only.
    Only the selected plots are computed, and only their results enter the table.
    base_cuts: optional {slice_var: cut applied before slicing} for the N-1
    slicing variant (see n_minus_1_base_cuts); None slices the full range.
    Returns one summary DataFrame of all chi2 / TVD / rho results.
    """
    import os
    unknown = set(plots) - set(QUANTILE_PLOTS)
    if unknown:
        raise ValueError(f'Unknown plot type(s) {sorted(unknown)}; choose from {QUANTILE_PLOTS}.')
    tables = []
    for comp in (components if components is not None else list(samples.keys())):
        if comp not in samples:
            continue
        for sv in slice_vars:
            df = samples[comp]
            cut = (base_cuts or {}).get(sv)
            if cut:
                df = df.query(cut)
            if len(df) < n_slices * 50:
                print(f'Skipping {comp} / {sv}: only {len(df)} entries')
                continue
            tag = f'{comp} | slices of {sv}' + (f' | base: {cut}' if cut else '')
            figs = {}
            if '1d' in plots:
                figs['1d'], t = plot_quantile_slices_1d(df, sv, n_slices, fit_vars, bins_1d,
                                                        min_count, title=tag)
                tables.append(t.assign(component=comp))
            if '2d' in plots:
                figs['2d'], t = plot_quantile_slices_2d(df, sv, n_slices, fit_vars, bins_2d,
                                                        min_count, title=tag)
                tables.append(t.assign(component=comp))
            if '2d_rel' in plots:
                figs['2d_rel'], t = plot_quantile_slices_2d(df, sv, n_slices, fit_vars, bins_2d,
                                                            min_count, map_type='rel', title=tag)
                if '2d' not in plots:   # same chi2/TVD as '2d'; avoid duplicate rows
                    tables.append(t.assign(component=comp))
            if 'rho' in plots:
                figs['rho'], t = plot_fit_variable_correlation_vs_slice(df, sv, n_slices,
                                                                       fit_vars, title=tag)
                tables.append(t.assign(component=comp, test_on='rho'))
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                stem = re.sub(r'[^A-Za-z0-9]+', '_', f'{comp}_{sv}').strip('_')
                for kind, f in figs.items():
                    f.savefig(os.path.join(save_dir, f'{stem}_{kind}.pdf'), bbox_inches='tight')
            if not show:
                for f in figs.values():
                    plt.close(f)
    if not tables:
        return pd.DataFrame()
    out = pd.concat(tables, ignore_index=True)
    return out[['component'] + [c for c in out.columns if c != 'component']]


############################## step 3: working-point checks ###################

def n_minus_1_study(df, cut, fit_vars=FIT_VARIABLES, bins_1d=None, bins_2d=None,
                    fit_bins=None, min_count=10, min_entries=50, map_type='rel',
                    map_lim=None, save_dir=None, show=True, tag=''):
    """Pass vs. fail of the full cut, then of each condition with the others applied (N-1).

    Each comparison uses compare_fit_variable_shapes on disjoint samples (pass
    vs. fail; chi2 and pulls need independent samples). The summary also has
    tvd_vs_input: the TVD between the passing sample and the sample before the
    cut, i.e. how much the cut changes the template (~ (1 - f_pass) * tvd).
    Tests that cannot be run are kept in the summary with a 'status' explaining
    why, e.g. a condition that no candidate fails once the others are applied.
    save_dir: if given, each figure is saved there as <tag>_<test>.pdf.
    show=False closes the figures after saving (no figures are made at all if
    show=False and save_dir=None; the summary table is still filled).
    Returns (summary DataFrame, {test_name: fig}).
    """
    import os
    make_plots = show or bool(save_dir)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    tests = {'full cut': (None, cut)}
    for cond, others in n_minus_1_cuts(cut).items():
        tests[f'N-1: {cond}'] = (others, cond)

    rows, figs = [], {}
    for name, (base_cut, test_cut) in tests.items():
        base = df.query(base_cut) if base_cut else df
        passed, failed = split_disjoint(base, test_cut)
        if min(len(passed), len(failed)) < min_entries:
            if len(failed) == 0:
                status = 'skipped: no candidate fails it (redundant given the base cut)'
            elif len(passed) == 0:
                status = 'skipped: no candidate passes it'
            else:
                status = f'skipped: fewer than {min_entries} entries in pass or fail'
            print(f'{tag} {name}: {status} (pass={len(passed)}, fail={len(failed)})')
            rows.append(dict(sample=tag, test=name, test_on=None, n_a=len(passed),
                             n_b=len(failed), f_pass=len(passed) / max(len(base), 1),
                             status=status))
            continue
        fig, res = compare_fit_variable_shapes(
            passed, failed, labels=('pass', 'fail'),
            fit_vars=fit_vars, bins_1d=bins_1d, bins_2d=bins_2d, fit_bins=fit_bins,
            min_count=min_count, map_type=map_type, map_lim=map_lim, plot=make_plots,
            title=f'{tag} | test: {test_cut} | base: {base_cut or "input sample"}')
        if fig is not None:
            if save_dir:
                test_name = 'full_cut' if name == 'full cut' else 'Nminus1_' + _condition_name(test_cut)
                stem = re.sub(r'[^A-Za-z0-9]+', '_', f'{tag}_{test_name}').strip('_')
                fig.savefig(os.path.join(save_dir, f'{stem}.pdf'), bbox_inches='tight')
            if not show:
                plt.close(fig)
                fig = None
        figs[name] = fig
        _, res_in = compare_fit_variable_shapes(passed, base, fit_vars=fit_vars, bins_1d=bins_1d,
                                                bins_2d=bins_2d, fit_bins=fit_bins,
                                                min_count=min_count, plot=False)
        new_rows = _results_to_rows(res, sample=tag, test=name, f_pass=len(passed) / len(base),
                                    status='ok')
        for r in new_rows:
            r['tvd_vs_input'] = res_in[r['test_on']]['tvd']
        rows += new_rows
    return pd.DataFrame(rows), figs

# endregion

##############################################################################
##            Statistics helpers shared by fitting.py and plotting.py       ##
##############################################################################
# These are used by both fitting.py and plotting.py, so they live here to keep
# the import direction one-way:  utilities <- fitting,  utilities <- plotting.

import uncertainties.unumpy as unp

def get_weights(df, w):
    """
    Create a weighted histogram where `w` can be either:
      - a scalar (e.g. 0.5)
      - a DataFrame column name
      - a list of form ['col_name', scalar]
    """
    if isinstance(w, str):
        return df[w].to_numpy()
    if np.isscalar(w):
        return np.full(len(df), w)
    if isinstance(w, (list, tuple)) and len(w) == 2 and isinstance(w[0], str):
        return (df[w[0]] * w[1]).to_numpy()
    raise TypeError(
        "w must be a scalar, a column name, or a (column name, scale) pair"
    )

def binom_error(n_sig, n_tot):
    """
    for an efficiency = nSig/nTrueSig or purity = nSig / (nSig + nBckgrd), this function calculates the
    standard deviation according to http://arxiv.org/abs/physics/0701199 .
    """
    variance = np.where(n_tot > 0, (n_sig + 1) * (n_sig + 2) / ((n_tot + 2) * (n_tot + 3)) -
                           (n_sig + 1) ** 2 / ((n_tot + 2) ** 2), 0)
    return unp.sqrt(variance)


def poisson_error(n_tot):
    """
    use poisson error, except for 0 we use an 68% CL upper limit, used for plotting only
    should not be used for creating fit templates
    p_poisson(x=0; l) = e^-l = 1-CL --> l = ln (1/ (1-CL) )
    """
    return np.where(n_tot > 0, np.sqrt(n_tot), np.log(1 / (1 - 0.6827)))


def round_uarray(uarray):
    """Rounds a uarray to 3 decimal places."""
    nominal = np.round(unp.nominal_values(uarray), 2)
    std_dev = np.round(unp.std_devs(uarray), 2)
    return unp.uarray(nominal, std_dev)


def merge_low_count_groups(model, min_count=10):
    """Group adjacent bins so that every group has >= min_count expected entries.

    Sweeps left to right, closing a group as soon as its summed model reaches
    min_count; a leftover tail that never reaches it joins the last closed group.
    Returns an int array of group labels (0, 1, ...), one per bin.
    """
    m = np.asarray(model, float).ravel()
    groups = np.empty(len(m), dtype=int)
    g, acc = 0, 0.0
    for i, mi in enumerate(m):
        groups[i] = g
        acc += mi
        if acc >= min_count:
            g, acc = g + 1, 0.0
    if len(m) and groups[-1] == g and g > 0:   # open tail below min_count
        groups[groups == g] = g - 1
    return groups


def excess_variance(data, model, data_err=None, model_err=None, normalize=True,
                    n_par=0, min_count=10, model_cov=None, low_count='merge'):
    """Data/MC mismodeling of a histogram as two numbers: normalization and shape.

    Normalization: R = sum(data) / sum(model) over all bins (model > 0), with its
    statistical uncertainty R_err from data and MC (stat + PID if model_cov has it).

    Low-count bins: the chi2 needs roughly Gaussian bins, i.e. >= min_count expected
    entries. low_count='merge' (default) merges adjacent bins of the MC prediction
    (before scaling) until each group has >= min_count (merge_low_count_groups);
    low_count='drop' leaves bins below min_count out. The same groups are used for
    the absolute chi2 and the shape test.

    Shape (excess variance): after scaling the model by R (normalize=True), the
    residuals r_g = d_g - m_g of the groups are compared with their statistical
    spread sigma_g (sigma^2 = data var + MC var):
        perfect model:            mean r_g^2 = sigma_g^2           -> chi2 ~ ndf
        model off by delta_g:     mean r_g^2 = sigma_g^2 + m_g^2 delta_g^2
    The variance beyond statistics gives the typical fractional mismodeling
        delta^2 = (chi2 - ndf) / sum_g (m_g / sigma_g)^2 ,   ndf = n_groups - n_par
    (one dof is used by R when normalize=True). Its statistical error is
    sqrt(2 ndf) / sum (m/sigma)^2; when delta^2 is not significant an upper limit
    is quoted by mismodeling_label.

    Absolute chi2 (chi2_raw, ndf_raw): the same groups against the UNSCALED model,
    testing normalization and shape together (ndf_raw = n_groups - n_par).

    data, model : arrays (any shape) or unumpy uarrays (errors taken from them)
    data_err    : default: Poisson error of the EXPECTED count (Pearson)
    model_err   : MC error per bin (default 0)
    model_cov   : full MC covariance (n_bins x n_bins), e.g. MC stat on the
                  diagonal + PID covariance from toys; needed to merge bins whose
                  errors are correlated. Overrides model_err.
    normalize   : False -> no scaling: delta then contains the normalization too
    n_par       : parameters fitted to obtain the model (post-fit plots)
    Returns dict(R, R_err, delta, delta_err, delta2, delta2_err, ul95, chi2, ndf,
                 chi2_raw, ndf_raw, groups, low_count_bins).
    low_count_bins: bins below min_count that were merged into a group (or
    dropped), i.e. that do not enter the chi2 on their own.
    """
    try:
        import uncertainties.unumpy as unp
        if data_err is None and hasattr(np.asarray(data).flat[0], 'std_dev'):
            data, data_err = unp.nominal_values(data), unp.std_devs(data)
        if model_err is None and hasattr(np.asarray(model).flat[0], 'std_dev'):
            model, model_err = unp.nominal_values(model), unp.std_devs(model)
    except ImportError:
        pass
    assert low_count in ('merge', 'drop'), "low_count must be 'merge' or 'drop'"
    d = np.asarray(data, float).ravel()
    m = np.asarray(model, float).ravel()
    n = len(m)
    ed = None if data_err is None else np.asarray(data_err, float).ravel()
    if model_cov is not None:
        C = np.asarray(model_cov, float)
        assert C.shape == (n, n), 'model_cov must be n_bins x n_bins'
    else:
        em = np.zeros(n) if model_err is None else np.asarray(model_err, float).ravel()
        C = np.diag(em ** 2)

    pos = m > 0
    D, M = d[pos].sum(), m[pos].sum()
    R = D / M if M > 0 else np.nan
    mc_var_tot = C[np.ix_(pos, pos)].sum()
    # statistical error of R; with no data at all use the 1-event scale
    R_err = np.sqrt(max(D, 1.0) / M ** 2 + (R ** 2 if D > 0 else 0.0) * mc_var_tot / M ** 2) \
        if M > 0 else np.nan

    # aggregation matrix A (n_groups x n_bins): row g sums the bins of group g
    if low_count == 'merge':
        groups = merge_low_count_groups(m, min_count)
        A = np.zeros((groups.max() + 1 if n else 0, n))
        A[groups, np.arange(n)] = 1.0
        sizes = np.bincount(groups)
        low = (m < min_count) & (sizes[groups] > 1)
    else:
        sel = m >= min_count
        groups = np.where(sel, np.cumsum(sel) - 1, -1)
        A = np.eye(n)[sel]
        low = ~sel
    out = dict(R=R, R_err=R_err, groups=groups, low_count_bins=low.reshape(np.shape(model)))

    def grouped_chi2(scale, npar):
        mg = A @ (m * scale)
        dg = A @ d
        data_var = mg if ed is None else A @ ed ** 2      # Pearson: expected count
        var = data_var + np.einsum('gi,ij,gj->g', A, C * scale ** 2, A)
        ok = (mg >= min_count) & (var > 0)
        ndf = int(ok.sum()) - npar
        if ndf < 1:
            return np.nan, ndf, np.nan
        r2 = (dg[ok] - mg[ok]) ** 2
        return float(np.sum(r2 / var[ok])), ndf, float(np.sum(mg[ok] ** 2 / var[ok]))

    # absolute chi2: unscaled model, tests normalization and shape together
    out['chi2_raw'], out['ndf_raw'], _ = grouped_chi2(1.0, n_par)

    scale = 1.0
    if normalize and np.isfinite(R) and R > 0:
        scale, n_par = R, n_par + 1
    chi2, ndf, info = grouped_chi2(scale, n_par)
    if ndf < 1:
        return dict(out, delta=np.nan, delta_err=np.nan, delta2=np.nan, delta2_err=np.nan,
                    ul95=np.nan, chi2=np.nan, ndf=ndf)
    delta2 = (chi2 - ndf) / info
    delta2_err = np.sqrt(2.0 * ndf) / info
    out.update(delta2=delta2, delta2_err=delta2_err, delta=np.sqrt(max(delta2, 0.0)),
               delta_err=delta2_err / (2 * np.sqrt(delta2)) if delta2 > 0 else np.nan,
               ul95=np.sqrt(max(delta2, 0.0) + 1.645 * delta2_err), chi2=chi2, ndf=ndf)
    return out


def mismodeling_label(res):
    """Three-line legend text for a data/MC residual plot from excess_variance():
    normalization, shape, absolute chi2/ndf (unscaled model)."""
    lines = [f"norm:  data/MC = {res['R']:.3f} $\\pm$ {res['R_err']:.3f}"]
    dim = res.get('shape_dim')
    tag = ' (M$^2_\\mathrm{miss}$ projection)' if dim == '1d' else ''
    if dim == 'none' or not np.isfinite(res.get('delta2', np.nan)):
        lines.append('shape: too few events (not used)')
    elif res['delta2'] > 2 * res['delta2_err']:
        lines.append(f"shape{tag}: $\\delta$ = {100 * res['delta']:.1f} $\\pm$ {100 * res['delta_err']:.1f}%")
    else:
        lines.append(f"shape{tag}: $\\delta$ < {100 * res['ul95']:.1f}% (95% CL)")
    if np.isfinite(res.get('chi2_raw', np.nan)):
        from scipy.stats import chi2 as chi2_dist
        p = chi2_dist.sf(res['chi2_raw'], res['ndf_raw'])
        lines.append(f"$\\chi^2$/ndf = {res['chi2_raw']:.1f}/{res['ndf_raw']}  (p = {p:.2g})")
    return '\n'.join(lines)
