# -*- coding: utf-8 -*-
# region
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

lgb_tight = 'sig_prob>0.5 and fakeD_prob<0.06 and continuum_prob<0.4 and combinatorial_prob<0.6'
lgb_loose = 'sig_prob>0.2 and fakeD_prob<0.15 and continuum_prob<0.5 and combinatorial_prob<0.6'
lgb_comb = 'fakeD_prob<0.15 and continuum_prob<0.5 and combinatorial_prob>0.5'

offline_cut = '(5<B0_roeMbc_my_mask) & (-4<B0_roeDeltae_my_mask) & (B0_roeDeltae_my_mask<1) & (B0_dr<0.1) & (ell_p<4)'

Dst_veto_cut = '( (DstVeto_massDiff_0<0.135) | (0.145<DstVeto_massDiff_0) )'

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

def apply_pid_corrections(df, MC='MC16', run='run1', channel='e', corr_col_name='total_PIDweight'):
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if channel=='e':
            sysvar.add_weights_to_dataframe('ell_BFbrems', df, systematic='custom_PID', custom_tables=e_tables, custom_thresholds=e_thresholds)
            weight_cols = ['ell_BFbrems_Weight', 'K_Weight', 'pi1_Weight', 'pi2_Weight',]
        elif channel=='mu':
            sysvar.add_weights_to_dataframe('ell', df, systematic='custom_PID', custom_tables=mu_tables, custom_thresholds=mu_thresholds)
            weight_cols = ['ell_Weight', 'K_Weight', 'pi1_Weight', 'pi2_Weight',]
        
        sysvar.add_weights_to_dataframe('K', df, systematic='custom_PID', custom_tables=k_tables, custom_thresholds=k_thresholds)
        sysvar.add_weights_to_dataframe('pi1', df, systematic='custom_PID', custom_tables=pi_tables, custom_thresholds=pi_thresholds)
        sysvar.add_weights_to_dataframe('pi2', df, systematic='custom_PID', custom_tables=pi_tables, custom_thresholds=pi_thresholds)
    
    # Consolidate the fragmented internal memory layout.
    df = df.copy()

    df[corr_col_name] = df[weight_cols].product(axis=1)

    return df


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
import numpy as np
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
    df_bestSelected = df_bestSelected.rename(columns={"__experiment__": "experiment", '__run__': 'run',
                                                      '__event__': 'event', '__production__': 'production'})

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

    Parameters
    ----------
    df_cut : DataFrame
        All candidates surviving preselection, before BCS.
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

    # tag the BCS-selected candidates with the same truth definition
    df_best = df_best.copy()
    df_best['is_true_signal'] = df_best.eval(truth_query)
    df_best_idx = df_best.set_index(group_cols)

    # restrict to events where a correct answer was actually available
    mask_true_exists = df_best_idx.index.isin(events_with_true)
    n_true_exists = mask_true_exists.sum()

    # BCS correct-pick rate == purity, once conditioned on true candidate existing
    bcs_correct_pick_rate = df_best_idx.loc[mask_true_exists, 'is_true_signal'].mean()

    # for reference only -- NOT a BCS-performance number, folds in preselection acceptance
    overall_selected_purity = df_best_idx['is_true_signal'].mean()

    return {
        'bcs_correct_pick_rate': bcs_correct_pick_rate,
        'overall_selected_purity': overall_selected_purity,
        'n_events_total': len(df_best_idx),
        'n_events_with_true_candidate': int(n_true_exists),
        'frac_events_with_true_candidate': n_true_exists / len(df_best_idx),
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
    keeping the BCS-performance number visually separate from the
    reference-only sample-composition number.
    """
    n_total = metrics['n_events_total']
    n_true = metrics['n_events_with_true_candidate']

    print(f"\nBCS performance  [bcs: {bcs}]")
    print(f"  Correct-pick rate (events with a true candidate): {metrics['bcs_correct_pick_rate']:.{decimals}f}")
    print(f"  Overall selected purity (reference only):         {metrics['overall_selected_purity']:.{decimals}f}")
    print(f"  Events after BCS:                                 {n_total:,d}")
    print(f"  Events with a true candidate:                     {n_true:,d} "
          f"({metrics['frac_events_with_true_candidate']:.2%})")


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

    New fitting code should call ``bbbar_reweighting.prepare_bbbar_reweighting``
    once and ``apply_bbbar_weights`` for each parameter point.  This wrapper
    preserves the historical structural mutation when ``weight_ell_side`` is
    false.
    """
    import bbbar_reweighting as bbbar

    prepared = bbbar.prepare_bbbar_reweighting(
        samples,
        weight_ell_side=weight_ell_side,
        cap_nbody=cap_nbody,
        D_replacement_map=D_replacement_map,
        ell_replacement_map=ell_replacement_map,
        replacement_use_pdg_corr=replacement_use_pdg_corr,
        copy=False,
        verbose=verbose,
    )
    weighted = bbbar.apply_bbbar_weights(
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


# def reweight_BBbar_background(
#     samples: dict[str, pd.DataFrame],
#     weight_map: dict[str, float],
#     out_weight_col: str = "BB_weight",
#     weight_ell_side: bool = False,
#     verbose: bool = False,
#     cap_nbody: int | None = None,
#     warn_missing_weight_keys: bool = True,
# ):
#     """
#     Reweight BBbar background components in-place in `samples`:
#       - For bkg_hadronicB_secondaryL: apply D-side classification only
#       - For bkg_combinatorial: apply D-side *and* lepton-side classification

#     Adds columns (when relevant):
#       D_dmID, D_is_measured, D_n_daughters, D_category, D_manual_w, D_pdg_corr, D_side_weight
#       ell_dmID, ell_is_measured, ell_n_daughters, ell_category, ell_manual_w, ell_pdg_corr, ell_side_weight

#     PDG BF correction is applied ONLY for measured modes:
#       pdg_corr = pdg_corr_dict[dmID] if is_measured and dmID in dict, else 1
#     """

#     pdg_corr = {
#         # dmID : (PDG BF)/(Generated BF) as of 2025PDG and MC16
#         10431 * 411: 0.00105 / 0.00737857,    # B0 -> D_s0*+ D-
#         10431 * 413: 0.0015  / 0.01768717,    # B0 -> D_s0*+ D*-
#         10431 * 421: 0.00079 / 0.00761729,    # B+ -> D_s0*+ D0
#         10431 * 423: 0.0009  / 0.01706301,    # B+ -> D_s0*+ D*0
#     }

#     def add_side_columns_inplace(
#         df: pd.DataFrame,
#         *,
#         prefix: str,
#         combinatorial_vars: list[str],
#         neutral_cols: list[str],
#         charged_cols: list[str],
#     ):
#         """
#         Writes the following columns onto df (prefixed):
#           {prefix}dmID
#           {prefix}mask_sl
#           {prefix}n_daughters
#           {prefix}is_measured
#           {prefix}category
#           {prefix}manual_w
#           {prefix}pdg_corr
#           {prefix}side_weight

#         Requires globals in your environment:
#           leptons, measured_pdg_list
#         """
#         # --- dmID (your encoding)
#         df[f"{prefix}dmID"] = df[combinatorial_vars].astype("int64").prod(axis=1).abs()

#         # --- semileptonic flag (top-level)
#         df[f"{prefix}mask_sl"] = df[combinatorial_vars].isin(leptons).any(axis=1)

#         # --- n_daughters proxy (exclude missing & photons)
#         neutral_n = (~df[neutral_cols].isin([-1, 22])).sum(axis=1)
#         charged_n = (~df[charged_cols].isin([-1, 22])).sum(axis=1)
#         df[f"{prefix}n_daughters"] = np.maximum(neutral_n, charged_n).astype(int)

#         # --- measured flag
#         df[f"{prefix}is_measured"] = df[f"{prefix}dmID"].isin(measured_pdg_list)

#         # --- hadronic categories
#         n_clip = df[f"{prefix}n_daughters"].clip(lower=2)
#         if cap_nbody is not None:
#             n_lbl = np.where(n_clip >= cap_nbody, f"{cap_nbody}+-body", n_clip.astype(str) + "-body",)
#         else:
#             n_lbl = n_clip.astype(str) + "-body"
#         had_cat = np.where(df[f"{prefix}is_measured"], "BBbar_measured_hadronic", "BBbar_unmeasured:" + n_lbl,)

#         # --- final category: semileptonic overrides everything
#         df[f"{prefix}category"] = np.where(df[f"{prefix}mask_sl"], "BBbar_semileptonic", had_cat)

#         # --- manual weights from dict
#         df[f"{prefix}manual_w"] = df[f"{prefix}category"].map(weight_map).fillna(1.0).astype(float)
        
#         if warn_missing_weight_keys:
#             present = set(pd.unique(df[f"{prefix}category"]))
#             missing = sorted(present - set(weight_map.keys()))
#             if missing and verbose:
#                 print(f"[{prefix}] Missing weight_map keys (defaulting to 1): {missing}")

#         # --- PDG correction ONLY for measured modes, AND only when dmID is in the dict
#         mapped_corr = df[f"{prefix}dmID"].map(pdg_corr).fillna(1.0).astype(float)
#         df[f"{prefix}pdg_corr"] = np.where(df[f"{prefix}is_measured"], mapped_corr, 1.0)

#         # --- combined side weight
#         df[f"{prefix}side_weight"] = df[f"{prefix}pdg_corr"] * df[f"{prefix}manual_w"]

#     # ---- main loop
#     for name, df in samples.items():
#         # default: no reweighting
#         df[out_weight_col] = 1.0

#         if name not in ["bkg_combinatorial", "bkg_hadronicB_secondaryL"]:
#             continue

#         # D-side always
#         add_side_columns_inplace(df, prefix="D_",
#             combinatorial_vars=combinatorial_vars_D,
#             neutral_cols=neutral_cols_D,
#             charged_cols=charged_cols_D,
#         )

#         df[out_weight_col] = df["D_side_weight"]

#         # For combinatorial, also apply lepton-side
#         if name == "bkg_combinatorial" and weight_ell_side:
#             add_side_columns_inplace(df, prefix="ell_",
#                 combinatorial_vars=combinatorial_vars_ell,
#                 neutral_cols=neutral_cols_ell,
#                 charged_cols=charged_cols_ell,
#             )
#             df[out_weight_col] *= df["ell_side_weight"]
            

#         # --- diagnostics
#         if verbose:
#             print(f"\n[{name}] N={len(df)}  sum({out_weight_col})={df[out_weight_col].sum():.3f}")

#             # Weighted yields AFTER all applied weights (using out_weight_col)
#             D_yields = df.groupby("D_category", dropna=False)['D_pdg_corr'].sum().sort_values(ascending=False)
#             print(D_yields)
#             if weight_ell_side and name == "bkg_combinatorial":
#                 ell_yields = df.groupby("ell_category", dropna=False)['ell_pdg_corr'].sum().sort_values(ascending=False)
#                 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
#                 ax1.pie(D_yields.to_numpy(), labels=D_yields.index.to_list(),autopct="%1.1f%%", startangle=90)
#                 ax2.pie(ell_yields.to_numpy(), labels=ell_yields.index.to_list(),autopct="%1.1f%%", startangle=90)
#                 fig.suptitle(f"MC composition {name}: measured vs unmeasured (split by n-body)", fontsize=16)
#                 ax1.set_title("D ancestor B decay")
#                 ax2.set_title("Lepton ancestor B decay (pdg_corr only)")
#                 ax1.axis("equal")
#                 ax2.axis("equal")
#                 plt.tight_layout()
#                 plt.show()
                
#             else:
#                 fig, ax = plt.subplots(figsize=(7, 7))
#                 ax.pie(D_yields.to_numpy(), labels=D_yields.index.to_list(),autopct="%1.1f%%", startangle=90)
#                 ax.set_title(f"MC composition {name}: D-side categories")
#                 ax.axis("equal")
#                 plt.tight_layout()
#                 plt.show()

#     if not weight_ell_side:
#         BBbar_bkg = pd.concat([samples['bkg_combinatorial'], samples['bkg_hadronicB_secondaryL'] ])
#         BBbar_by_category = dict(tuple(BBbar_bkg.groupby("D_category")))
#         samples.pop('bkg_combinatorial')
#         samples.pop('bkg_hadronicB_secondaryL')
#         samples.update(BBbar_by_category)
        
#     return samples
    

def classify_measured_modes(df, base='D',corr_col_name='BF_corr_w_D'):
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
    df["dmIsMeasured"] = df["dmID"].isin(measured_pdg_list)
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


FIT_VARIABLES = ['B0_recMissM2', 'p_D_l']
MVA_OUTPUTS = ['sig_prob', 'fakeD_prob', 'continuum_prob', 'combinatorial_prob']

FIT_VARIABLE_LABELS = {
    'B0_recMissM2': r'$M_{\mathrm{miss}}^2$ [GeV$^2$]',
    'p_D_l':        r'$|p^*_D| + |p^*_\ell|$ [GeV]',
}

D_MASS_SR = '1.855<D_M<1.885'       # same window as create_templates_new
D_MASS_SB = 'D_M<1.85 or 1.9<D_M'   # same sidebands as create_templates_new

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

# Event-ID columns after apply_mva_bcs has renamed the __xxx__ columns
EVENT_ID_COLUMNS = ['experiment', 'run', 'event', 'production']


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
##                          Templates and workspace                         ##
##############################################################################

from uncertainties import ufloat, correlated_values, UFloat
import uncertainties.unumpy as unp
import copy
from termcolor import colored

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
        ax.errorbar(x=bin_centers,y=res_val,yerr=res_err,fmt='.',color='black',
                     markeredgecolor='white',markeredgewidth=0.5, label=label)
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
    axes[0, 0].errorbar(x=bin_centers_x,y=unp.nominal_values(projData_x),
                        yerr=unp.std_devs(projData_x),fmt='.',color='black',
                     markeredgecolor='white',markeredgewidth=0.5, label=label)
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
    # NOTE (BBbar categories): bbbar_reweighting emits ':5+-body' when called
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
##################################### Plotting #################################
import matplotlib.colors as mcolors

######## define my colormap ########
# Original tab20 colors
original_colors = plt.cm.tab20.colors
# New order for the colors
new_order_indices = [0,1,2,3,16,5,10,4,12,13,19,8,18,7,6]
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
                  apply_eventByEvent_correction=False, eventByEvent_weight_col='total_PID_weight'):

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

            if apply_eventByEvent_correction: # update the weight
                weight = apply_eventByEvent_weight(sample, weight, eventByEvent_weight_col)

            (counts, _) = np.histogram(sample[variable], bins=bins,weights=weight)
            (staterr_squared, _) = np.histogram(sample[variable], bins=bins,weights=weight**2)
            staterror = poisson_error(staterr_squared)

            counts, staterror = normalize_to_density(counts, staterror, bins)

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

                if apply_eventByEvent_correction: # update the weight
                    weight = apply_eventByEvent_weight(sample, weight, eventByEvent_weight_col)

                (counts, _) = np.histogram(sample[variable], bins=bins,weights=weight)
                (staterr_squared, _) = np.histogram(sample[variable], bins=bins,weights=weight**2)
                staterror = poisson_error(staterr_squared)
                

                # Normalize if density=True
                counts, staterror = normalize_to_density(
                    counts, staterror, bins, density_integral
                )
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

        return bottom
    
    
    def plot_mc_1d_overlaid(self,variable,bins,cut=None,mask=[],show_only=None,density=False, errorbars=False, 
                            weights={},figsize=(8,5),legend_nc=3,text_fs=14):
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
        axs.set_title(f'MC distribution', fontsize=text_fs)
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

    def plot_residuals(self, bins, data, model, ax, fig=None):
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
            # Compute chi-squared excluding those points
            chi2 = np.sum((res_val[mask] / res_err[mask]) ** 2)
            ndf = len(res_val[mask])
            label = f'reChi2 = {chi2:.3f} / {ndf} = {chi2/ndf:.3f}' if ndf else 'reChi2 not calculated'

            # Plot the residuals in ax
            # ax.errorbar(x=bin_centers, y=res_val, yerr=res_err, fmt='ok',label=label) # absolute residuals
            ax.plot(bin_centers, normalized_residuals, 'ok', label=label) # normalized residuals
            
            # Add a horizontal line at y=0 for reference
            ax.axhline(0, color='gray', linestyle='--')
            # Label the residual plot
            ax.set_ylabel('Normalized Residuals')
            
        elif len(bins)==2: # 2d residual
            residuals = abs(data - model) # abs to make the plot simpler
            res_val = unp.nominal_values(residuals)
            res_err = unp.std_devs(residuals)
            
            # Create a mask to exclude points where residual_errors are zero
            mask = res_val != 0
            # Compute chi-squared excluding those points
            chi2 = np.sum((res_val[mask] / res_err[mask]) ** 2)
            ndf = len(res_val[mask])
            
            if ndf==0:
                label = 'reChi2 not calculated'
            else:
                label = f'reChi2 = {chi2:.3f} / {ndf} = {chi2/ndf:.3f}'
                
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
                             figsize=(8,5),legend_nc=2,legend_fs=12,text_fs=14,title=None):
        
        assert ratio_or_residual in ['ratio', 'residual'], 'must choose ratio or residual'
        
        # Create a figure with two subplots: one for the histogram, one for the residual plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [5, 1]})
            
        # MC
        mc_counts = self.plot_mc_1d(bins=bins, variable=variable, ax=ax1, cut=cut,mask=mask, density=density,
                                    weights=weights, apply_eventByEvent_correction=apply_eventByEvent_correction, 
                                    eventByEvent_weight_col=eventByEvent_weight_col,)
        # Data
        if self.data is not None:
            data_counts = self.plot_data_1d(bins=bins, variable=variable, sig_mask=data_sig_mask,
                                            ax=ax1, cut=cut, scale=weights.get('data', 1))
            # plot residuals
            if ratio_or_residual=='ratio':
                self.plot_ratios(bins=bins, data=data_counts, model=mc_counts, ax=ax2)
            elif ratio_or_residual=='residual':
                # Residuals (Data - Model)
                self.plot_residuals(bins=bins, data=data_counts, model=mc_counts, ax=ax2)
                ax2.legend(bbox_to_anchor=(1,1),fancybox=True, shadow=True, fontsize=legend_fs)
            
        else: # if MC only
            data_counts = unp.uarray(np.zeros_like(mc_counts), np.zeros_like(mc_counts))
            bottom_comp = self.plot_mc_1d(bins=bins, sub_df=self.samples[bottom_plot], sub_name=bottom_plot, 
                                        variable=variable, ax=ax2, cut=cut,mask=mask, density=density, legend='simple_color',
                                        weights=weights, apply_eventByEvent_correction=apply_eventByEvent_correction, 
                                        eventByEvent_weight_col=eventByEvent_weight_col,)
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
        ax2.set_xlabel(self.var_name_dictionary.get(variable, variable), fontsize=text_fs)
        
        plt.tight_layout()
        plt.show()

        return data_counts, mc_counts
        
        
    def plot_mc_sig_control(self,variable,bins,cut=None,weights={},mask=[], density=False,
                            apply_eventByEvent_correction=False,
                            eventByEvent_weight_col='total_PID_weight',
                            bkg_name='bkg_fakeD',samples_sig=None,
                            figsize=(10,5),legend_nc=1,text_fs=12):
        if type(variable)==str:
            # Create a figure with two subplots: one for the histogram, one for the residual plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [5, 1]})
        elif type(variable)==list:
            # Create a figure with two subplots: one for sig, one for the control
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        if bkg_name=='bkg_fakeD':
            print('This part of the function needs to be rewritten!')
            
            # norm in sig region, for tail removal
#             Dellnu = self.samples[r'$D\ell\nu$'].query('1.84<D_M<1.9').copy()
#             Dstellnu = self.samples[r'$D^\ast\ell\nu$'].query('1.84<D_M<1.9').copy()
                    
#             # fakeD in the signal region
#             fakeD = self.samples[bkg_name]
#             fakeD_in_sig = fakeD.query('1.84<D_M<1.9').copy()
            
#             # fakeD in sidebands vs. everything in the sidebands (concatenate all components)
#             fakeD_left = fakeD.query('D_M<1.83').copy()
#             fakeD_right = fakeD.query('1.91<D_M').copy()
            
#             df_concatenated = pd.concat(self.samples.values(), ignore_index=True)
#             all_left = df_concatenated.query('D_M<1.83').copy()
#             all_right = df_concatenated.query('1.91<D_M').copy()
                
#             regions = {'left sideband': all_left if full_sideband else fakeD_left,
#                        'signal region': fakeD_in_sig,
#                        'right sideband': all_right if full_sideband else fakeD_right}
            
#             sb_total = 0 # total counts in sidebands used in residual calculation
#             sig_total = 0
#             if type(variable)==str:
#                 ###################################
#                 D_counts = self.plot_mc_1d(bins=bins, sub_df=Dellnu, sub_name=r'$D\ell\nu$', variable=variable, 
#                                            ax=None, cut=cut, correction=correction,mask=mask,
#                                            weights={'sub_df':weights.get('signal region',1)})
#                 Dst_counts = self.plot_mc_1d(bins=bins, sub_df=Dstellnu, sub_name=r'$D^\ast\ell\nu$', variable=variable, 
#                                              ax=None, cut=cut, correction=correction,mask=mask,
#                                              weights={'sub_df':weights.get('signal region',1)},)
#                 ###################################
                    
#                 for region, df in regions.items():
#                     if region=='signal region':
#                         sig_total = self.plot_data_1d(bins=bins, sub_df=df, variable=variable, ax=ax1, 
#                                                 cut=cut, name=region, scale=weights.get(region,1) )

#                     elif region in ['left sideband', 'right sideband']:
#                         bkg_counts = self.plot_mc_1d(bins=bins, sub_df=df, sub_name=region, variable=variable, 
#                                                     ax=None, cut=cut, correction=correction,mask=mask,
#                                                     weights={'sub_df':weights.get(region,1)})

                        
#                         if region == 'left sideband':
#                             sb_total += bkg_counts * len(fakeD_in_sig) / len(fakeD_left)
#                         elif region == 'right sideband':
#                             sb_total += bkg_counts * len(fakeD_in_sig) / len(fakeD_right)
                        
#                         if not merge_sidebands:
#                             count1 = unp.nominal_values(bkg_counts)
#                             ax1.hist(bins[:-1], bins, weights=count1,histtype='step',
#                     label=f'{region} \n{self.statistics(hist=[count1, bins],count_only=False)} ')
                    
#                 if merge_sidebands:
#                     count2 = unp.nominal_values(sb_total)
#                     ax1.hist(bins[:-1], bins, weights=count2,histtype='step',
#                 label=f'sidebands \n{self.statistics(hist=[count2, bins],count_only=False)} ')

#                 # Residuals (Data - Model) and their errors
#                 self.plot_residuals(bins=bins, data=sig_total, model=sb_total, ax=ax2)
#                 ax2.set_xlabel(f'{variable}')
                
#             elif type(variable)==list:
#                 assert merge_sidebands==True, 'merge_sidebands must be True'
#                 for region, df in regions.items():
#                     if region=='signal region':
#                         sig_total = self.plot_2d(bins=bins, sub_df=df, variables=variable, title_name=region,
#                                     weights={'sub_df': weights.get(region,1)},fig=fig, ax=ax1, cut=cut)
#                     elif region in ['left sideband', 'right sideband']:
#                         bkg_counts = self.plot_2d(bins=bins, sub_df=df, variables=variable, title_name=region,
#                                     weights={'sub_df': weights.get(region,1)},fig=None, ax=None, cut=cut)
#                         D_counts = self.plot_2d(bins=bins, sub_df=Dellnu, variables=variable, title_name=region,
#                                     weights={'sub_df': weights.get(region,1)},fig=None, ax=None, cut=cut)
#                         Dst_counts = self.plot_2d(bins=bins, sub_df=Dstellnu, variables=variable, title_name=region,
#                                     weights={'sub_df': weights.get(region,1)},fig=None, ax=None, cut=cut)
# #                         sb_total -= r_D * D_counts
# #                         sb_total -= r_Dst * Dst_counts
#                         sb_total += bkg_counts
                        
#                 self.plot_2d(bins=bins, hist=unp.nominal_values(sb_total),title_name='sidebands',fig=fig, ax=ax2, cut=cut)
                        
#                 # Residuals (Data - Model) and their errors
#                 self.plot_residuals(bins=bins, data=sig_total, model=sb_total, fig=fig, ax=ax3)
            
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
