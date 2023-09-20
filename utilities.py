# -*- coding: utf-8 -*-
# +
# define BDT training variables
spectators = ['__weight__', 'D_CMS_p', 'e_CMS_p', 'B0_CMS3_weMissM2', 'B0_CMS3_weQ2lnuSimple']

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
                 'D_pi1_pValue',               'D_pi2_pValue',
                 'D_vtxReChi2',                #'D_BFInvM', D mass will suppress the sideband
                 'D_A1FflightDistanceSig_IP',  'D_daughterInvM_0_1',
                 'D_daughterInvM_1_2',         'B0_vtxDDSig',]

B_variables = ['B0_Lab5_weMissPTheta',
               'B0_vtxReChi2',               'B0_flightDistanceSig',
               'B0_nROE_Tracks_my_mask',     'B0_nROE_NeutralHadrons_my_mask',
               'B0_roel_DistanceSig_dis',    'B0_roeDeltae_my_mask',
               'B0_roeEextra_my_mask',       'B0_roeMbc_my_mask',
               'B0_roeCharge_my_mask',       'B0_nROE_Photons_my_mask',
               'B0_nROE_K',                  'B0_TagVReChi2IP',]

training_variables = CS_variables + DTC_variables + B_variables
variables = training_variables + spectators

# +
## DecayHash

from collections import OrderedDict

# the order of keys might be important, try to keep the muon modes at the bottom for e reconstruction
# the e modes will be kept at the bottom for a muon reconstruction
mode_dict = {}
mode_dict['e'] = OrderedDict()
mode_dict['e']['sig_D_tau_nu']=[
    '511 (-> -411 (-> 321 -211 -211) -15 (-> -11 12 -16) 16)',
    '-511 (-> 411 (-> -321 211 211) 15 (-> 11 -12 16) -16)']

mode_dict['e']['sig_D_l_nu']=[
    '511 (-> -411 (-> 321 -211 -211) -11 12)',
    '-511 (-> 411 (-> -321 211 211) 11 -12)']

mode_dict['e']['sig_Dst_tau_nu']=[
    '511 (-> -413 (-> -411 (-> 321 -211 -211) 111) -15 (-> -11 12 -16) 16)',
    '511 (-> -413 (-> -411 (-> 321 -211 -211) 22) -15 (-> -11 12 -16) 16)',
    '-511 (-> 413 (-> 411 (-> -321 211 211) 111) 15 (-> 11 -12 16) -16)',
    '-511 (-> 413 (-> 411 (-> -321 211 211) 22) 15 (-> 11 -12 16) -16)']

mode_dict['e']['sig_Dst_l_nu']=[
    '511 (-> -413 (-> -411 (-> 321 -211 -211) 111) -11 12)',
    '511 (-> -413 (-> -411 (-> 321 -211 -211) 22) -11 12)',
    '-511 (-> 413 (-> 411 (-> -321 211 211) 111) 11 -12)',
    '-511 (-> 413 (-> 411 (-> -321 211 211) 22) 11 -12)']

mode_dict['e']['Dstst_tau_nu_mixed']=[
    '511 (-> -10413 -15 (-> -11 12 -16) 16)','-511 (-> 10413 15 (-> 11 -12 16) -16)',
    '511 (-> -10411 -15 (-> -11 12 -16) 16)','-511 (-> 10411 15 (-> 11 -12 16) -16)',
    '511 (-> -20413 -15 (-> -11 12 -16) 16)','-511 (-> 20413 15 (-> 11 -12 16) -16)',
    '511 (-> -415 -15 (-> -11 12 -16) 16)',  '-511 (-> 415 15 (-> 11 -12 16) -16)']

mode_dict['e']['Dstst_tau_nu_charged']=[
    '521 (-> -10423 -15 (-> -11 12 -16) 16)','-521 (-> 10423 15 (-> 11 -12 16) -16)',
    '521 (-> -10421 -15 (-> -11 12 -16) 16)','-521 (-> 10421 15 (-> 11 -12 16) -16)',
    '521 (-> -20423 -15 (-> -11 12 -16) 16)','-521 (-> 20423 15 (-> 11 -12 16) -16)',
    '521 (-> -425 -15 (-> -11 12 -16) 16)',  '-521 (-> 425 15 (-> 11 -12 16) -16)']

mode_dict['e']['res_Dstst_l_nu_mixed']=[
    '511 (-> -10413 -11 12)','-511 (-> 10413 11 -12)',
    '511 (-> -10411 -11 12)','-511 (-> 10411 11 -12)',
    '511 (-> -20413 -11 12)','-511 (-> 20413 11 -12)',
    '511 (-> -415 -11 12)',  '-511 (-> 415 11 -12)']

mode_dict['e']['res_Dstst_l_nu_charged']=[
    '521 (-> -10423 -11 12)','-521 (-> 10423 11 -12)',
    '521 (-> -10421 -11 12)','-521 (-> 10421 11 -12)',
    '521 (-> -20423 -11 12)','-521 (-> 20423 11 -12)',
    '521 (-> -425 -11 12)',  '-521 (-> 425 11 -12)']

mode_dict['e']['nonres_Dstst_l_nu_mixed']=[
    '511 (-> -411 111 -11 12)',     '-511 (-> 411 111 11 -12)',
    '511 (-> -411 111 111 -11 12)', '-511 (-> 411 111 111 11 -12)',
    '511 (-> -411 211 -211 -11 12)','-511 (-> 411 211 -211 11 -12)',
    '511 (-> -411 -211 211 -11 12)','-511 (-> 411 -211 211 11 -12)',
    '511 (-> -413 111 -11 12)',     '-511 (-> 413 111 11 -12)',
    '511 (-> -413 111 111 -11 12)', '-511 (-> 413 111 111 11 -12)',
    '511 (-> -413 211 -211 -11 12)','-511 (-> 413 211 -211 11 -12)',
    '511 (-> -413 -211 211 -11 12)','-511 (-> 413 -211 211 11 -12)']

mode_dict['e']['nonres_Dstst_l_nu_charged']=[
    '521 (-> -411 211 -11 12)',    '-521 (-> 411 -211 11 -12)',
    '521 (-> -411 211 111 -11 12)','-521 (-> 411 -211 111 11 -12)',
    '521 (-> -413 211 -11 12)',    '-521 (-> 413 -211 11 -12)',
    '521 (-> -413 211 111 -11 12)','-521 (-> 413 -211 111 11 -12)']

mode_dict['e']['gap_Dstst_l_nu_mixed']=[
    '511 (-> -411 221 -11 12)','-511 (-> 411 221 11 -12)',
    '511 (-> -413 221 -11 12)','-511 (-> 413 221 11 -12)']


#################################

mode_dict['mu'] = OrderedDict()
mode_dict['mu']['sig_D_tau_nu']=[
    '511 (-> -411 (-> 321 -211 -211) -15 (-> -13 14 -16) 16)',
    '-511 (-> 411 (-> -321 211 211) 15 (-> 13 -14 16) -16)']

mode_dict['mu']['sig_D_l_nu']=[
    '511 (-> -411 (-> 321 -211 -211) -13 14)',
    '-511 (-> 411 (-> -321 211 211) 13 -14)']

mode_dict['mu']['sig_Dst_tau_nu']=[
    '511 (-> -413 (-> -411 (-> 321 -211 -211) 111) -15 (-> -13 14 -16) 16)',
    '511 (-> -413 (-> -411 (-> 321 -211 -211) 22) -15 (-> -13 14 -16) 16)',
    '-511 (-> 413 (-> 411 (-> -321 211 211) 111) 15 (-> 13 -14 16) -16)',
    '-511 (-> 413 (-> 411 (-> -321 211 211) 22) 15 (-> 13 -14 16) -16)']

mode_dict['mu']['sig_Dst_l_nu']=[
    '511 (-> -413 (-> -411 (-> 321 -211 -211) 111) -13 14)',
    '511 (-> -413 (-> -411 (-> 321 -211 -211) 22) -13 14)',
    '-511 (-> 413 (-> 411 (-> -321 211 211) 111) 13 -14)',
    '-511 (-> 413 (-> 411 (-> -321 211 211) 22) 13 -14)']

mode_dict['mu']['Dstst_tau_nu_mixed']=[
    '511 (-> -10413 -15 (-> -13 14 -16) 16)','-511 (-> 10413 15 (-> 13 -14 16) -16)',
    '511 (-> -10411 -15 (-> -13 14 -16) 16)','-511 (-> 10411 15 (-> 13 -14 16) -16)',
    '511 (-> -20413 -15 (-> -13 14 -16) 16)','-511 (-> 20413 15 (-> 13 -14 16) -16)',
    '511 (-> -415 -15 (-> -13 14 -16) 16)',  '-511 (-> 415 15 (-> 13 -14 16) -16)']

mode_dict['mu']['Dstst_tau_nu_charged']=[
    '521 (-> -10423 -15 (-> -13 14 -16) 16)','-521 (-> 10423 15 (-> 13 -14 16) -16)',
    '521 (-> -10421 -15 (-> -13 14 -16) 16)','-521 (-> 10421 15 (-> 13 -14 16) -16)',
    '521 (-> -20423 -15 (-> -13 14 -16) 16)','-521 (-> 20423 15 (-> 13 -14 16) -16)',
    '521 (-> -425 -15 (-> -13 14 -16) 16)',  '-521 (-> 425 15 (-> 13 -14 16) -16)']

mode_dict['mu']['res_Dstst_l_nu_mixed']=[
    '511 (-> -10413 -13 14)','-511 (-> 10413 13 -14)',
    '511 (-> -10411 -13 14)','-511 (-> 10411 13 -14)',
    '511 (-> -20413 -13 14)','-511 (-> 20413 13 -14)',
    '511 (-> -415 -13 14)',  '-511 (-> 415 13 -14)',]

mode_dict['mu']['res_Dstst_l_nu_charged']=[
    '521 (-> -10423 -13 14)','-521 (-> 10423 13 -14)',
    '521 (-> -10421 -13 14)','-521 (-> 10421 13 -14)',
    '521 (-> -20423 -13 14)','-521 (-> 20423 13 -14)',
    '521 (-> -425 -13 14)',  '-521 (-> 425 13 -14)',]

mode_dict['mu']['nonres_Dstst_l_nu_mixed']=[
    '511 (-> -411 111 -13 14)',     '-511 (-> 411 111 13 -14)',
    '511 (-> -411 111 111 -13 14)', '-511 (-> 411 111 111 13 -14)',
    '511 (-> -411 211 -211 -13 14)','-511 (-> 411 211 -211 13 -14)',
    '511 (-> -411 -211 211 -13 14)','-511 (-> 411 -211 211 13 -14)',
    '511 (-> -413 111 -13 14)',     '-511 (-> 413 111 13 -14)',
    '511 (-> -413 111 111 -13 14)', '-511 (-> 413 111 111 13 -14)',
    '511 (-> -413 211 -211 -13 14)','-511 (-> 413 211 -211 13 -14)',
    '511 (-> -413 -211 211 -13 14)','-511 (-> 413 -211 211 13 -14)',]

mode_dict['mu']['nonres_Dstst_l_nu_charged']=[
    '521 (-> -411 211 -13 14)',    '-521 (-> 411 -211 13 -14)',
    '521 (-> -411 211 111 -13 14)','-521 (-> 411 -211 111 13 -14)',
    '521 (-> -413 211 -13 14)',    '-521 (-> 413 -211 13 -14)',
    '521 (-> -413 211 111 -13 14)','-521 (-> 413 -211 111 13 -14)']

mode_dict['mu']['gap_Dstst_l_nu_mixed']=[
    '511 (-> -411 221 -13 14)','-511 (-> 411 221 13 -14)',
    '511 (-> -413 221 -13 14)','-511 (-> 413 221 13 -14)']

# +
from enum import Enum

# define DecayModes from DecayHash
class DecayMode(Enum):
    bkg = 0
    sig_D_tau_nu = 1
    sig_D_l_nu = 2
    sig_Dst_tau_nu = 3
    sig_Dst_l_nu = 4
    Dstst_tau_nu_mixed = 5
    Dstst_tau_nu_charged = 6
    res_Dstst_l_nu_mixed = 7
    nonres_Dstst_l_nu_mixed = 8
    gap_Dstst_l_nu_mixed = 9
    res_Dstst_l_nu_charged = 10
    nonres_Dstst_l_nu_charged = 11

DecayMode = Enum('DecayMode', ['bkg',                 'sig_D_tau_nu',          'sig_D_l_nu',
                               'sig_Dst_tau_nu',      'sig_Dst_l_nu',          'Dstst_tau_nu_mixed',
                               'Dstst_tau_nu_charged','res_Dstst_l_nu_mixed',  'nonres_Dstst_l_nu_mixed',
                               'gap_Dstst_l_nu_mixed','res_Dstst_l_nu_charged','nonres_Dstst_l_nu_charged'],
                 start=0)
# DecayMode(0).name

# +
## dataframe samples
import pandas as pd
def get_dataframe_samples(df, mode, template=True):
    samples = {}
    lepton_PDG = {'e':11, 'mu':13}
    
    # Define Truth matching criteria
    true_D_tau = f'D_mcPDG*{mode}_mcPDG==411*{lepton_PDG[mode]} and {mode}_mcPDG*{mode}_genMotherPDG=={lepton_PDG[mode]}*15'
    
    true_D_l = f'D_mcPDG*{mode}_mcPDG==411*{lepton_PDG[mode]} and {mode}_genMotherPDG==B0_mcPDG'
    
    true_B0 = 'B0_mcPDG*D_mcPDG==-511*411'
    B_charged = 'B0_mcPDG*D_mcPDG==-521*411'
    
    lepton_misID = f'abs({mode}_mcPDG)!={lepton_PDG[mode]}'
    
###################### mcErrors need to be handled carefully
###################### apparently correct e mcPDG==11 could have large mcErrors, 128, 2048; mcSecPhysProc need to be checked later
    DecayErrors = {}
    # norm modes, signal modes, D** mixed modes
    for key, value in {'D_l':[8,16],'Dst_l':[8,64],'D_tau':[8,32],'Dst_Dstst_mixed':[8,64]}.items():
                    # tau to e has a 1% radiative mode, thus 32
        correct_decay = f'({value[0]}+{mode}_mcErrors)<=B0_mcErrors<({value[1]}+{mode}_mcErrors)'
        missing_photon = f'({value[0]+1024}+{mode}_mcErrors)<=B0_mcErrors<({value[1]+1024}+{mode}_mcErrors)'
        
#         wrongBremsDaughter = f'{value[0]+2048}<=B0_mcErrors<{value[1]+2048+128}'
#         missing_photon_wrongBrems = f'{value[0]+2048+1024}<=B0_mcErrors<{value[1]+2048+1024+128}'
        
        DecayErrors[f'{key}_errors'] = f'({correct_decay} or {missing_photon})'
        # note that the parentheses in the `or` statement is very important when using `and` in front
        if template and (key in ['D_l', 'Dst_l']):
            DecayErrors[f'{key}_errors'] = correct_decay
        
            
    # D** charged modes
    Bcharged_errors = f'({8+256}+{mode}_mcErrors)<=B0_mcErrors<({64+256}+{mode}_mcErrors)'
    missing_photon_Bcharged = f'({8+1024+256}+{mode}_mcErrors)<=B0_mcErrors<({64+1024+256}+{mode}_mcErrors)'
    
    # a charged particle (e.g. pi,e,mu) is added as a Brems daughter by the correctBrem module
    # however, this mistake doesn't change the MM2 and p_D_l much, still within the signal window
    # B0_mcErrors is offset by 128 due to the misID of the Brems daughter
#     wrongBremsDaughter_Bcharged = f'{8+2048+256}<=B0_mcErrors<{64+2048+256+128}'
#     missing_photon_wrongBrems_Bcharged = f'{8+2048+1024+256}<=B0_mcErrors<{64+2048+1024+256+128}'

    DecayErrors[f'Bcharged_errors'] = f'({Bcharged_errors} or {missing_photon_Bcharged})'
#         if template:
#             DecayErrors[f'Bcharged_errors'] = Bcharged_errors


    # Sig components
    sig_D_tau_nu=df.query(f'DecayMode=={DecayMode["sig_D_tau_nu"].value} and \
    {true_B0} and {true_D_tau} and {DecayErrors["D_tau_errors"]}').copy()
    
    sig_D_l_nu=df.query(f'DecayMode=={DecayMode[f"sig_D_l_nu"].value} and \
    {true_B0} and {true_D_l} and {DecayErrors["D_l_errors"]}').copy()

    sig_Dst_tau_nu=df.query(f'DecayMode=={DecayMode["sig_Dst_tau_nu"].value} and \
    {true_B0} and {true_D_tau} and {DecayErrors["Dst_Dstst_mixed_errors"]}').copy()

    sig_Dst_l_nu=df.query(f'DecayMode=={DecayMode[f"sig_Dst_l_nu"].value} and \
    {true_B0} and {true_D_l} and {DecayErrors["Dst_l_errors"]}').copy()

    Dstst_tau_nu_mixed=df.query(f'DecayMode=={DecayMode["Dstst_tau_nu_mixed"].value} and \
    {true_B0} and {true_D_tau} and {DecayErrors["Dst_Dstst_mixed_errors"]}').copy()

    res_Dstst_l_nu_mixed=df.query(f'DecayMode=={DecayMode[f"res_Dstst_l_nu_mixed"].value} and \
    {true_B0} and {true_D_l} and {DecayErrors["Dst_Dstst_mixed_errors"]}').copy()
    
    nonres_Dstst_l_nu_mixed=df.query(f'DecayMode=={DecayMode[f"nonres_Dstst_l_nu_mixed"].value} and \
    {true_B0} and {true_D_l} and {DecayErrors["Dst_Dstst_mixed_errors"]}').copy()
    
    gap_Dstst_l_nu_mixed=df.query(f'DecayMode=={DecayMode[f"gap_Dstst_l_nu_mixed"].value} and \
    {true_B0} and {true_D_l} and {DecayErrors["Dst_Dstst_mixed_errors"]}').copy()
    
    Dstst_tau_nu_charged=df.query(f'DecayMode=={DecayMode["Dstst_tau_nu_charged"].value} and \
    {B_charged} and {true_D_tau} and {DecayErrors["Bcharged_errors"]}').copy()
    
    res_Dstst_l_nu_charged=df.query(f'DecayMode=={DecayMode[f"res_Dstst_l_nu_charged"].value} and \
    {B_charged} and {true_D_l} and {DecayErrors["Bcharged_errors"]}').copy()
    
    nonres_Dstst_l_nu_charged=df.query(f'DecayMode=={DecayMode[f"nonres_Dstst_l_nu_charged"].value} and \
    {B_charged} and {true_D_l} and {DecayErrors["Bcharged_errors"]}').copy()

    
    samples[r'$D\tau\nu$'] = sig_D_tau_nu
    samples[r'$D^\ast\tau\nu$'] = sig_Dst_tau_nu
    samples[r'$D^{\ast\ast}\tau\nu$_mixed'] = Dstst_tau_nu_mixed
    samples[r'$D^{\ast\ast}\tau\nu$_charged'] = Dstst_tau_nu_charged
    samples[r'$D\ell\nu$'] = sig_D_l_nu
    samples[r'$D^\ast\ell\nu$'] = sig_Dst_l_nu
    samples[r'res_$D^{\ast\ast}\ell\nu$_mixed'] = res_Dstst_l_nu_mixed
    samples[r'nonres_$D^{\ast\ast}\ell\nu$_mixed'] = nonres_Dstst_l_nu_mixed
    samples[r'gap_$D^{\ast\ast}\ell\nu$_mixed'] = gap_Dstst_l_nu_mixed
    samples[r'res_$D^{\ast\ast}\ell\nu$_charged'] = res_Dstst_l_nu_charged
    samples[r'nonres_$D^{\ast\ast}\ell\nu$_charged'] = nonres_Dstst_l_nu_charged
   
    
    #Bkg components
    bkg_fakeD = df.query('(abs(D_mcPDG)!=411 or D_mcErrors>=8) and B0_mcErrors!=512 and B0_isContinuumEvent!=1').copy()
    bkg_fakeTracksClusters = df.query('B0_mcErrors==512 and B0_isContinuumEvent!=1').copy()
    samples[r'bkg_fakeTC'] = bkg_fakeTracksClusters
    samples[r'bkg_fakeD'] = bkg_fakeD

    bkg_combinatorial = df.query('B0_mcPDG==300553 and abs(D_mcPDG)==411 and B0_mcErrors!=512 and B0_isContinuumEvent!=1').copy()
    
    bkg_sigOtherBDTaudecay = df.query(f'DecayMode=={DecayMode["bkg"].value} and \
                 B0_mcPDG!=300553 and D_mcErrors<8 and \
                 B0_mcErrors!=512 and B0_isContinuumEvent!=1 and \
                 ({lepton_misID} or D_mcPDG*{mode}_mcPDG==411*{lepton_PDG[mode]})').copy()
                # lepton_misID and possible other decay
                # or correct Dl but not a signal decay
    
    samples[r'bkg_Odecay'] = bkg_sigOtherBDTaudecay
    samples[r'bkg_combinatorial'] = bkg_combinatorial
    
    bkg_continuum = df.query('B0_isContinuumEvent==1').copy()
    samples[r'bkg_continuum'] = bkg_continuum

    bkg_others = pd.concat([df,
                            sig_D_tau_nu,
                            sig_D_l_nu,
                            sig_Dst_tau_nu,
                            sig_Dst_l_nu,
                            Dstst_tau_nu_mixed,
                            Dstst_tau_nu_charged,
                            res_Dstst_l_nu_mixed,
                            nonres_Dstst_l_nu_mixed,
                            gap_Dstst_l_nu_mixed,
                            res_Dstst_l_nu_charged,
                            nonres_Dstst_l_nu_charged,
                            bkg_fakeTracksClusters,
                            bkg_fakeD,
                            bkg_sigOtherBDTaudecay,
                            bkg_combinatorial,
                            bkg_continuum]).drop_duplicates(keep=False)
    samples[r'bkg_others'] = bkg_others
    
    for name, df in samples.items():
        df['mode']=name

    df = pd.concat([df for df in samples.values()],ignore_index=True).reset_index(drop=True)
    df['p_D_l_region'] = np.where(df['p_D_l']>2.5,1,0)
    
    return df, samples

    # the bkg_others contains events with wrong bremsstralung corrected electrons
    # The added daughter to the electron are pions, electrons, muons
    # so the 130<B0_mcErrors<160, e_mcErrors==128, 2176, 2180
    
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
## Plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib import gridspec
# import mplhep as hep
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20.colors)

class mpl:
    def __init__(self, samples):
        self.samples = samples
        self.kwarg={'histtype':'step','lw':2}
    
    def statistics(self,df):
        counts=df.count()
        mean=df.mean()
        std=df.std()
        return f'''counts = %d \nmean = %5.3f \nstd = %5.3f''' %(counts,mean,std)

    def plot_all_separately(self, variable, cut=None, xlim=None):
        fig,axs =plt.subplots(4,3,figsize=(16,10), sharex=True, sharey=False)
        fig.suptitle(f'All components with {cut}',fontsize=16)
        fig.supylabel('# of candidates per bin',x=0.06,fontsize=16)
        fig.supxlabel(f'{variable}', y=0.06,fontsize=16)
        i=0
        j=0
        for sample_name, sample in self.samples.items():
            (counts, bins) = np.histogram(sample.query(cut)[variable] if cut else sample[variable], bins=50)
            if sample_name in [r'$D\tau\nu$',r'$D^\ast\tau\nu$',r'$D^{\ast\ast}\tau\nu$']:
                factor = 1
            elif sample_name in [r'$D\ell\nu$',r'$D^\ast\ell\nu$',r'$D^{\ast\ast}\ell\nu$']:
                factor = 1
            axs[i,j].hist(bins[:-1], bins, weights=factor*counts,label=sample_name,**self.kwarg)

            #plt.legend(bbox_to_anchor=(1,1),ncol=3, fancybox=True, shadow=True,labelspacing=1.5)
            axs[i,j].grid()
            axs[i,j].set_title(sample_name)
            if xlim:
                axs[i,j].set_xlim(xlim)
            j+=1
            if j==3:
                i+=1
                j=0
                
    def plot_signals_overlaid(self, variable, cut=None, mask=[]):
        fig,axs =plt.subplots(1,2,figsize=(12,5), sharex=False, sharey=False)
        fig.suptitle(f'Overlaid signals ({cut=})', y=1,fontsize=16)
        fig.supylabel('# of candidates per bin',x=0.06,fontsize=16)
        #fig.supxlabel('$|\\vec{p_D}|\ +\ |\\vec{p_l}|$  [GeV/c]')
        #fig.supxlabel('$M_{miss}^2 \ [GeV^2/c^4]$')
        fig.supxlabel(f'{variable}',fontsize=16)

        for name, sample in self.samples.items():
            if name in mask:
                continue
            selection_mask = sample.eval(cut)
            left= sample[selection_mask] if cut else sample
            right = sample[~selection_mask] if cut else sample
            (counts_left, bins_left) = np.histogram(left[variable], bins=50)
            (counts_right, bins_right) = np.histogram(right[variable], bins=50)
            if name in [r'$D\tau\nu$',r'$D^\ast\tau\nu$',
                        r'$D^{\ast\ast}\tau\nu$_mixed',
                        r'$D^{\ast\ast}\tau\nu$_charged',
                        r'$D\ell\nu$',r'$D^\ast\ell\nu$',]:
                factor=1
                
            elif name in [r'res_$D^{\ast\ast}\ell\nu$_mixed',
                          r'res_$D^{\ast\ast}\ell\nu$_charged',]:
                factor=2
                
            elif name in [r'nonres_$D^{\ast\ast}\ell\nu$_mixed',
                          r'nonres_$D^{\ast\ast}\ell\nu$_charged']:
                factor=16

            elif name in [r'gap_$D^{\ast\ast}\ell\nu$_mixed',]:
                factor=8
                
            else:
                factor=1
            
            axs[0].hist(bins_left[:-1], bins_left, weights=factor*counts_left,
                        label=name,**self.kwarg)
            
            axs[1].hist(bins_right[:-1], bins_right, weights=factor*counts_right,
                        label=name,**self.kwarg)
            
            axs[1].legend()
            
        axs[0].set_title(f'Bin1 with {cut}')
        axs[1].set_title(f'Bin2 with ~{cut}')
        axs[0].grid()
        axs[1].grid()
        plt.legend(bbox_to_anchor=(1,1),ncol=1, fancybox=True, shadow=True,labelspacing=1.5)
        
        
    def plot_tails_overlaid(self, variable, cut=None):
        fig,axs =plt.subplots(1,2,figsize=(12,5), sharex=True, sharey=False)
        fig.suptitle(f'Overlaid norms ({cut=})', y=1,fontsize=16)
        fig.supylabel('# of candidates per bin',x=0.06,fontsize=16)
        #fig.supxlabel('$|\\vec{p_D}|\ +\ |\\vec{p_l}|$  [GeV/c]')
        #fig.supxlabel('$M_{miss}^2 \ [GeV^2/c^4]$')
        fig.supxlabel(f'{variable}',fontsize=16)

        for sample_name, sample in self.samples.items():
            if sample_name not in [r'$D\ell\nu$',r'$D^\ast\ell\nu$']:
                continue
            (counts1, bins) = np.histogram(
                sample.query(cut)[variable] if cut else sample[variable], bins=50)
            (counts2, bins) = np.histogram(
                sample.drop(sample.query(cut).index)[variable] if cut else sample[variable], bins=bins)
            factor=1

            axs[0].hist(bins[:-1], bins, weights=factor*counts2,label=sample_name,**self.kwarg)
            axs[0].legend()

            axs[1].hist(bins[:-1], bins, weights=factor*counts1,label=sample_name,**self.kwarg)
            axs[1].legend()

        axs[0].set_title('Main')
        axs[1].set_title('Tail')
        axs[0].grid()
        axs[1].grid()
        
    def plot_all_overlaid(self,variable,cut=None,mask=[]):
            
        fig,axs =plt.subplots(sharex=True, sharey=False)
        for name, sample in self.samples.items():
            sample_size = len(sample.query(cut)) if cut else len(sample)
            if sample_size==0 or name in mask:
                continue
            var_col= sample.query(cut) if cut else sample
            (counts, bins) = np.histogram(var_col[variable], bins=50)
            if name in [r'$D\tau\nu$',r'$D^\ast\tau\nu$',
                        r'$D^{\ast\ast}\tau\nu$_mixed',
                        r'$D^{\ast\ast}\tau\nu$_charged',
                        r'$D\ell\nu$',r'$D^\ast\ell\nu$',]:
                factor=1
                
            elif name in [r'res_$D^{\ast\ast}\ell\nu$_mixed',
                          r'res_$D^{\ast\ast}\ell\nu$_charged',]:
                factor=2
                
            elif name in [r'nonres_$D^{\ast\ast}\ell\nu$_mixed',
                          r'nonres_$D^{\ast\ast}\ell\nu$_charged']:
                factor=16

            elif name in [r'gap_$D^{\ast\ast}\ell\nu$_mixed',]:
                factor=8
                
            else:
                factor=1
            axs.hist(bins[:-1], bins, weights=factor*counts,
                    label=f'{name} \n{self.statistics(var_col[variable])}',**self.kwarg)

        axs.set_title(f'Overlaid signals ({cut=})')
        axs.set_xlabel(f'{variable}')
        axs.set_ylabel('# of candidates per bin')
        axs.grid()
        plt.legend(bbox_to_anchor=(1,1),ncol=3, fancybox=True, shadow=True,labelspacing=1.5)
        

    def plot_hist_2d(self, variables=['B0_CMS3_weMissM2','p_D_l'], cut=None, mask=[1.6,1]):
        variable_x, variable_y = variables
        xedges = np.linspace(-2, 10, 48)
        yedges = np.linspace(0.4, 4.6, 42)
        
        
        fig = plt.figure(figsize=[16,26])
        for i, (name, sample) in enumerate(self.samples.items()):
            sample_size = len(sample.query(cut)) if cut else len(sample)
            if sample_size==0:
                continue
            ax = fig.add_subplot(6,3,i+1)
            (counts, xedges, yedges) = np.histogram2d(
                            sample.query(cut)[variable_x] if cut else sample[variable_x], 
                            sample.query(cut)[variable_y] if cut else sample[variable_y],
                            bins=[xedges, yedges])
            counts = counts.T
            
            mask_arr = np.ones_like(counts)
            if mask:
                # apply mask at mm2<1.6 and p_D_l>1 
                mm2_split = mask[0]
                pDl_split = mask[1]
                mm2_split_index, = np.asarray(np.isclose(xedges,mm2_split,atol=0.2)).nonzero()
                pDl_split_index, = np.asarray(np.isclose(yedges,pDl_split,atol=0.2)).nonzero()
                mask_arr[:,mm2_split_index[0]:] = mask[2] # select the small mm2
                mask_arr[:pDl_split_index[0],:] = mask[2] # select the large pDl
                

            X, Y = np.meshgrid(xedges, yedges)
            im=ax.pcolormesh(X, Y, counts, cmap='rainbow', norm=colors.LogNorm(), alpha=mask_arr)
            ax.grid()
            ax.set_xlim(xedges.min(),xedges.max())
            ax.set_ylim(yedges.min(),yedges.max())
            ax.set_title(name,fontsize=14)
            fig.colorbar(im,ax=ax)
            ax.set_title(name,fontsize=14)

        fig.suptitle(f'Signal MC ({cut=})', y=0.95, fontsize=18)
        fig.supylabel('$|p_D|\ +\ |p_l|\ \ \ [GeV]$', x=0.05,fontsize=18)
        fig.supxlabel('$M_{miss}^2\ \ \ [GeV^2/c^4]$', y=0.05,fontsize=18)
        

    def plot_correlation(self, df, cut=None, target='B0_CMS3_weMissM2', variables=variables):
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
        
    def plot_fitting_difference(self, yaml_file):
        fig,axs =plt.subplots(2,3,figsize=(16,10), sharex=True, sharey=False)
        fig.suptitle(f'fitted yield - true yield',fontsize=16)
        fig.supylabel('yield difference',x=0.06,fontsize=16)
        fig.supxlabel(f'index of subset samples', y=0.06,fontsize=16)
        i=0
        j=0
        with open(yaml_file, 'r+') as f:
            data = yaml.safe_load(f)
            components = data['signal_e']

        for comp_name, info in components.items():
            axs[i,j].errorbar(x=range(1,len(info['difference'])+1), y=info['difference'], yerr=info['errors'], fmt='ko')
            axs[i,j].axhline(y=0, linestyle='-', linewidth=3, color='r')
            axs[i,j].grid()
            axs[i,j].set_title(comp_name,fontsize=14)
            j+=1
            if j==3 and i==0:
                i+=1
                j=0
            if j==3 and i==1:
                break
            
            
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
        
def mpl_projection_residual_cabinetry(fit_result, templates_2d, data_2d, edges, slices=[1.6,1],direction='mm2', plot_with='pltbar'):
    if direction not in ['mm2', 'p_D_l'] or plot_with not in ['mplhep', 'pltbar']:
        raise ValueError('direction in [mm2, p_D_l] and plot_with in [mplhep, pltbar]')
    fitted_components_names = fit_result.labels.tolist()
    #### fitted_templates_2d = templates / normalization * yields
    fitted_templates_2d = [templates_2d[i] * fit_result.best_fit[i] for i in range(len(templates_2d))]
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

# +
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
