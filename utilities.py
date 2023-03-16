# -*- coding: utf-8 -*-
# +
## DecayHash

from collections import OrderedDict

# the order of keys might be important, try to keep the muon modes at the bottom for e reconstruction
# the e modes will be kept at the bottom for a muon reconstruction
mode_dict = {}
mode_dict['e'] = OrderedDict()
mode_dict['e']['sig_D_tau_nu']=['511 (-> -411 (-> 321 -211 -211) -15 (-> -11 12 -16) 16)',
                           '-511 (-> 411 (-> -321 211 211) 15 (-> 11 -12 16) -16)',
                           '511 (-> -411 (-> 321 -211 -211) -15 (-> -13 14 -16) 16)',
                           '-511 (-> 411 (-> -321 211 211) 15 (-> 13 -14 16) -16)']

mode_dict['e']['sig_D_e_nu']=['511 (-> -411 (-> 321 -211 -211) -11 12)',
                         '-511 (-> 411 (-> -321 211 211) 11 -12)']

mode_dict['e']['sig_Dst_tau_nu']=['511 (-> -413 (-> -411 (-> 321 -211 -211) 111) -15 (-> -11 12 -16) 16)',
                             '-511 (-> 413 (-> 411 (-> -321 211 211) 111) 15 (-> 11 -12 16) -16)',
                             '511 (-> -413 (-> -411 (-> 321 -211 -211) 111) -15 (-> -13 14 -16) 16)',
                             '-511 (-> 413 (-> 411 (-> -321 211 211) 111) 15 (-> 13 -14 16) -16)',
                             '511 (-> -413 (-> -411 (-> 321 -211 -211) 22) -15 (-> -11 12 -16) 16)',
                             '-511 (-> 413 (-> 411 (-> -321 211 211) 22) 15 (-> 11 -12 16) -16)',
                             '511 (-> -413 (-> -411 (-> 321 -211 -211) 22) -15 (-> -13 14 -16) 16)',
                             '-511 (-> 413 (-> 411 (-> -321 211 211) 22) 15 (-> 13 -14 16) -16)']

mode_dict['e']['sig_Dst_e_nu']=['511 (-> -413 (-> -411 (-> 321 -211 -211) 111) -11 12)',
                           '511 (-> -413 (-> -411 (-> 321 -211 -211) 22) -11 12)',
                           '-511 (-> 413 (-> 411 (-> -321 211 211) 111) 11 -12)',
                           '-511 (-> 413 (-> 411 (-> -321 211 211) 22) 11 -12)']

mode_dict['e']['all_Dstst_tau_nu']=['511 (-> -10413 -15 (-> -11 12 -16) 16)','-511 (-> 10413 15 (-> 11 -12 16) -16)',
                               '511 (-> -10411 -15 (-> -11 12 -16) 16)','-511 (-> 10411 15 (-> 11 -12 16) -16)',
                               '511 (-> -20413 -15 (-> -11 12 -16) 16)','-511 (-> 20413 15 (-> 11 -12 16) -16)',
                               '511 (-> -415 -15 (-> -11 12 -16) 16)',  '-511 (-> 415 15 (-> 11 -12 16) -16)',
                               '521 (-> -10423 -15 (-> -11 12 -16) 16)','-521 (-> 10423 15 (-> 11 -12 16) -16)',
                               '521 (-> -10421 -15 (-> -11 12 -16) 16)','-521 (-> 10421 15 (-> 11 -12 16) -16)',
                               '521 (-> -20423 -15 (-> -11 12 -16) 16)','-521 (-> 20423 15 (-> 11 -12 16) -16)',
                               '521 (-> -425 -15 (-> -11 12 -16) 16)',  '-521 (-> 425 15 (-> 11 -12 16) -16)',
                               '511 (-> -10413 -15 (-> -13 14 -16) 16)','-511 (-> 10413 15 (-> 13 -14 16) -16)',
                               '511 (-> -10411 -15 (-> -13 14 -16) 16)','-511 (-> 10411 15 (-> 13 -14 16) -16)',
                               '511 (-> -20413 -15 (-> -13 14 -16) 16)','-511 (-> 20413 15 (-> 13 -14 16) -16)',
                               '511 (-> -415 -15 (-> -13 14 -16) 16)',  '-511 (-> 415 15 (-> 13 -14 16) -16)',
                               '521 (-> -10423 -15 (-> -13 14 -16) 16)','-521 (-> 10423 15 (-> 13 -14 16) -16)',
                               '521 (-> -10421 -15 (-> -13 14 -16) 16)','-521 (-> 10421 15 (-> 13 -14 16) -16)',
                               '521 (-> -20423 -15 (-> -13 14 -16) 16)','-521 (-> 20423 15 (-> 13 -14 16) -16)',
                               '521 (-> -425 -15 (-> -13 14 -16) 16)',  '-521 (-> 425 15 (-> 13 -14 16) -16)']

mode_dict['e']['all_Dstst_e_nu']=['511 (-> -10413 -11 12)','-511 (-> 10413 11 -12)',
                             '511 (-> -10411 -11 12)','-511 (-> 10411 11 -12)',
                             '511 (-> -20413 -11 12)','-511 (-> 20413 11 -12)',
                             '511 (-> -415 -11 12)',  '-511 (-> 415 11 -12)',
                             '511 (-> -411 221 -11 12)','-511 (-> 411 221 11 -12)',
                             '511 (-> -411 111 -11 12)','-511 (-> 411 111 11 -12)',
                             '511 (-> -411 111 111 -11 12)','-511 (-> 411 111 111 11 -12)',
                             '511 (-> -411 211 -211 -11 12)','-511 (-> 411 211 -211 11 -12)',
                             '511 (-> -413 221 -11 12)','-511 (-> 413 221 11 -12)',
                             '511 (-> -413 111 -11 12)','-511 (-> 413 111 11 -12)',
                             '511 (-> -413 111 111 -11 12)','-511 (-> 413 111 111 11 -12)',
                             '511 (-> -413 211 -211 -11 12)','-511 (-> 413 211 -211 11 -12)',
                             '511 (-> -421 -211 -11 12)','-511 (-> 421 211 11 -12)',
                             '511 (-> -423 -211 -11 12)','-511 (-> 423 211 11 -12)',
                             '521 (-> -10423 -11 12)','-521 (-> 10423 11 -12)',
                             '521 (-> -10421 -11 12)','-521 (-> 10421 11 -12)',
                             '521 (-> -20423 -11 12)','-521 (-> 20423 11 -12)',
                             '521 (-> -425 -11 12)',  '-521 (-> 425 11 -12)',
                             '521 (-> -411 211 -11 12)','-521 (-> 411 211 11 -12)',
                             '521 (-> -411 211 111 -11 12)','-521 (-> 411 211 111 11 -12)',
                             '521 (-> -413 211 -11 12)','-521 (-> 413 211 11 -12)',
                             '521 (-> -413 211 111 -11 12)','-521 (-> 413 211 111 11 -12)']

mode_dict['e']['sig_D_mu_nu']=['511 (-> -411 (-> 321 -211 -211) -13 14)',
                          '-511 (-> 411 (-> -321 211 211) 13 -14)']

mode_dict['e']['sig_Dst_mu_nu']=['511 (-> -413 (-> -411 (-> 321 -211 -211) 111) -13 14)',
                           '511 (-> -413 (-> -411 (-> 321 -211 -211) 22) -13 14)',
                           '-511 (-> 413 (-> 411 (-> -321 211 211) 111) 13 -14)',
                           '-511 (-> 413 (-> 411 (-> -321 211 211) 22) 13 -14)']

mode_dict['e']['all_Dstst_mu_nu']=['511 (-> -10413 -13 14)','-511 (-> 10413 13 -14)',
                              '511 (-> -10411 -13 14)','-511 (-> 10411 13 -14)',
                              '511 (-> -20413 -13 14)','-511 (-> 20413 13 -14)',
                              '511 (-> -415 -13 14)',  '-511 (-> 415 13 -14)',
                              '511 (-> -411 221 -13 14)','-511 (-> 411 221 13 -14)',
                              '511 (-> -411 111 -13 14)','-511 (-> 411 111 13 -14)',
                              '511 (-> -411 111 111 -13 14)','-511 (-> 411 111 111 13 -14)',
                              '511 (-> -411 211 -211 -13 14)','-511 (-> 411 211 -211 13 -14)',
                              '511 (-> -413 221 -13 14)','-511 (-> 413 221 13 -14)',
                              '511 (-> -413 111 -13 14)','-511 (-> 413 111 13 -14)',
                              '511 (-> -413 111 111 -13 14)','-511 (-> 413 111 111 13 -14)',
                              '511 (-> -413 211 -211 -13 14)','-511 (-> 413 211 -211 13 -14)',
                              '511 (-> -421 -211 -13 14)','-511 (-> 421 211 13 -14)',
                              '511 (-> -423 -211 -13 14)','-511 (-> 423 211 13 -14)',
                              '521 (-> -10423 -13 14)','-521 (-> 10423 13 -14)',
                              '521 (-> -10421 -13 14)','-521 (-> 10421 13 -14)',
                              '521 (-> -20423 -13 14)','-521 (-> 20423 13 -14)',
                              '521 (-> -425 -13 14)',  '-521 (-> 425 13 -14)',
                              '521 (-> -411 211 -13 14)','-521 (-> 411 211 13 -14)',
                              '521 (-> -411 211 111 -13 14)','-521 (-> 411 211 111 13 -14',
                              '521 (-> -413 211 -13 14)','-521 (-> 413 211 13 -14)',
                              '521 (-> -413 211 111 -13 14)','-521 (-> 413 211 111 13 -14)']


mode_dict['mu'] = OrderedDict()
mode_dict['mu']['sig_D_tau_nu']=['511 (-> -411 (-> 321 -211 -211) -15 (-> -11 12 -16) 16)',
                           '-511 (-> 411 (-> -321 211 211) 15 (-> 11 -12 16) -16)',
                           '511 (-> -411 (-> 321 -211 -211) -15 (-> -13 14 -16) 16)',
                           '-511 (-> 411 (-> -321 211 211) 15 (-> 13 -14 16) -16)']

mode_dict['mu']['sig_D_mu_nu']=['511 (-> -411 (-> 321 -211 -211) -13 14)',
                          '-511 (-> 411 (-> -321 211 211) 13 -14)']

mode_dict['mu']['sig_Dst_tau_nu']=['511 (-> -413 (-> -411 (-> 321 -211 -211) 111) -15 (-> -11 12 -16) 16)',
                             '-511 (-> 413 (-> 411 (-> -321 211 211) 111) 15 (-> 11 -12 16) -16)',
                             '511 (-> -413 (-> -411 (-> 321 -211 -211) 111) -15 (-> -13 14 -16) 16)',
                             '-511 (-> 413 (-> 411 (-> -321 211 211) 111) 15 (-> 13 -14 16) -16)',
                             '511 (-> -413 (-> -411 (-> 321 -211 -211) 22) -15 (-> -11 12 -16) 16)',
                             '-511 (-> 413 (-> 411 (-> -321 211 211) 22) 15 (-> 11 -12 16) -16)',
                             '511 (-> -413 (-> -411 (-> 321 -211 -211) 22) -15 (-> -13 14 -16) 16)',
                             '-511 (-> 413 (-> 411 (-> -321 211 211) 22) 15 (-> 13 -14 16) -16)']

mode_dict['mu']['sig_Dst_mu_nu']=['511 (-> -413 (-> -411 (-> 321 -211 -211) 111) -13 14)',
                           '511 (-> -413 (-> -411 (-> 321 -211 -211) 22) -13 14)',
                           '-511 (-> 413 (-> 411 (-> -321 211 211) 111) 13 -14)',
                           '-511 (-> 413 (-> 411 (-> -321 211 211) 22) 13 -14)']

mode_dict['mu']['all_Dstst_tau_nu']=['511 (-> -10413 -15 (-> -11 12 -16) 16)','-511 (-> 10413 15 (-> 11 -12 16) -16)',
                               '511 (-> -10411 -15 (-> -11 12 -16) 16)','-511 (-> 10411 15 (-> 11 -12 16) -16)',
                               '511 (-> -20413 -15 (-> -11 12 -16) 16)','-511 (-> 20413 15 (-> 11 -12 16) -16)',
                               '511 (-> -415 -15 (-> -11 12 -16) 16)',  '-511 (-> 415 15 (-> 11 -12 16) -16)',
                               '521 (-> -10423 -15 (-> -11 12 -16) 16)','-521 (-> 10423 15 (-> 11 -12 16) -16)',
                               '521 (-> -10421 -15 (-> -11 12 -16) 16)','-521 (-> 10421 15 (-> 11 -12 16) -16)',
                               '521 (-> -20423 -15 (-> -11 12 -16) 16)','-521 (-> 20423 15 (-> 11 -12 16) -16)',
                               '521 (-> -425 -15 (-> -11 12 -16) 16)',  '-521 (-> 425 15 (-> 11 -12 16) -16)',
                               '511 (-> -10413 -15 (-> -13 14 -16) 16)','-511 (-> 10413 15 (-> 13 -14 16) -16)',
                               '511 (-> -10411 -15 (-> -13 14 -16) 16)','-511 (-> 10411 15 (-> 13 -14 16) -16)',
                               '511 (-> -20413 -15 (-> -13 14 -16) 16)','-511 (-> 20413 15 (-> 13 -14 16) -16)',
                               '511 (-> -415 -15 (-> -13 14 -16) 16)',  '-511 (-> 415 15 (-> 13 -14 16) -16)',
                               '521 (-> -10423 -15 (-> -13 14 -16) 16)','-521 (-> 10423 15 (-> 13 -14 16) -16)',
                               '521 (-> -10421 -15 (-> -13 14 -16) 16)','-521 (-> 10421 15 (-> 13 -14 16) -16)',
                               '521 (-> -20423 -15 (-> -13 14 -16) 16)','-521 (-> 20423 15 (-> 13 -14 16) -16)',
                               '521 (-> -425 -15 (-> -13 14 -16) 16)',  '-521 (-> 425 15 (-> 13 -14 16) -16)']

mode_dict['mu']['all_Dstst_mu_nu']=['511 (-> -10413 -13 14)','-511 (-> 10413 13 -14)',
                              '511 (-> -10411 -13 14)','-511 (-> 10411 13 -14)',
                              '511 (-> -20413 -13 14)','-511 (-> 20413 13 -14)',
                              '511 (-> -415 -13 14)',  '-511 (-> 415 13 -14)',
                              '511 (-> -411 221 -13 14)','-511 (-> 411 221 13 -14)',
                              '511 (-> -411 111 -13 14)','-511 (-> 411 111 13 -14)',
                              '511 (-> -411 111 111 -13 14)','-511 (-> 411 111 111 13 -14)',
                              '511 (-> -411 211 -211 -13 14)','-511 (-> 411 211 -211 13 -14)',
                              '511 (-> -413 221 -13 14)','-511 (-> 413 221 13 -14)',
                              '511 (-> -413 111 -13 14)','-511 (-> 413 111 13 -14)',
                              '511 (-> -413 111 111 -13 14)','-511 (-> 413 111 111 13 -14)',
                              '511 (-> -413 211 -211 -13 14)','-511 (-> 413 211 -211 13 -14)',
                              '511 (-> -421 -211 -13 14)','-511 (-> 421 211 13 -14)',
                              '511 (-> -423 -211 -13 14)','-511 (-> 423 211 13 -14)',
                              '521 (-> -10423 -13 14)','-521 (-> 10423 13 -14)',
                              '521 (-> -10421 -13 14)','-521 (-> 10421 13 -14)',
                              '521 (-> -20423 -13 14)','-521 (-> 20423 13 -14)',
                              '521 (-> -425 -13 14)',  '-521 (-> 425 13 -14)',
                              '521 (-> -411 211 -13 14)','-521 (-> 411 211 13 -14)',
                              '521 (-> -411 211 111 -13 14)','-521 (-> 411 211 111 13 -14',
                              '521 (-> -413 211 -13 14)','-521 (-> 413 211 13 -14)',
                              '521 (-> -413 211 111 -13 14)','-521 (-> 413 211 111 13 -14)']

mode_dict['mu']['sig_D_e_nu']=['511 (-> -411 (-> 321 -211 -211) -11 12)',
                         '-511 (-> 411 (-> -321 211 211) 11 -12)']

mode_dict['mu']['sig_Dst_e_nu']=['511 (-> -413 (-> -411 (-> 321 -211 -211) 111) -11 12)',
                           '511 (-> -413 (-> -411 (-> 321 -211 -211) 22) -11 12)',
                           '-511 (-> 413 (-> 411 (-> -321 211 211) 111) 11 -12)',
                           '-511 (-> 413 (-> 411 (-> -321 211 211) 22) 11 -12)']

mode_dict['mu']['all_Dstst_e_nu']=['511 (-> -10413 -11 12)','-511 (-> 10413 11 -12)',
                             '511 (-> -10411 -11 12)','-511 (-> 10411 11 -12)',
                             '511 (-> -20413 -11 12)','-511 (-> 20413 11 -12)',
                             '511 (-> -415 -11 12)',  '-511 (-> 415 11 -12)',
                             '511 (-> -411 221 -11 12)','-511 (-> 411 221 11 -12)',
                             '511 (-> -411 111 -11 12)','-511 (-> 411 111 11 -12)',
                             '511 (-> -411 111 111 -11 12)','-511 (-> 411 111 111 11 -12)',
                             '511 (-> -411 211 -211 -11 12)','-511 (-> 411 211 -211 11 -12)',
                             '511 (-> -413 221 -11 12)','-511 (-> 413 221 11 -12)',
                             '511 (-> -413 111 -11 12)','-511 (-> 413 111 11 -12)',
                             '511 (-> -413 111 111 -11 12)','-511 (-> 413 111 111 11 -12)',
                             '511 (-> -413 211 -211 -11 12)','-511 (-> 413 211 -211 11 -12)',
                             '511 (-> -421 -211 -11 12)','-511 (-> 421 211 11 -12)',
                             '511 (-> -423 -211 -11 12)','-511 (-> 423 211 11 -12)',
                             '521 (-> -10423 -11 12)','-521 (-> 10423 11 -12)',
                             '521 (-> -10421 -11 12)','-521 (-> 10421 11 -12)',
                             '521 (-> -20423 -11 12)','-521 (-> 20423 11 -12)',
                             '521 (-> -425 -11 12)',  '-521 (-> 425 11 -12)',
                             '521 (-> -411 211 -11 12)','-521 (-> 411 211 11 -12)',
                             '521 (-> -411 211 111 -11 12)','-521 (-> 411 211 111 11 -12)',
                             '521 (-> -413 211 -11 12)','-521 (-> 413 211 11 -12)',
                             '521 (-> -413 211 111 -11 12)','-521 (-> 413 211 111 11 -12)']

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
from enum import Enum

# define DecayModes from DecayHash
class DecayMode(Enum):
    bkg = 0
    sig_D_tau_nu = 1
    sig_D_e_nu = 2
    sig_Dst_tau_nu = 3
    sig_Dst_e_nu = 4
    all_Dstst_tau_nu = 5
    all_Dstst_e_nu = 6
    sig_D_mu_nu = 7
    sig_Dst_mu_nu = 8
    all_Dstst_mu_nu = 9

DecayMode = Enum('DecayMode', ['bkg', 'sig_D_tau_nu', 'sig_D_e_nu', 'sig_Dst_tau_nu',
                           'sig_Dst_e_nu', 'all_Dstst_tau_nu', 'all_Dstst_e_nu',
                           'sig_D_mu_nu', 'sig_Dst_mu_nu', 'all_Dstst_mu_nu'],
             start=0)

# +
## Plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
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

    def plot_separately(self, variable, cut=None, xlim=None):
        fig,axs =plt.subplots(4,3,figsize=(16,10), sharex=True, sharey=False)
        fig.suptitle(f'All signals with {cut}',fontsize=16)
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

    def plot_hist_2d(self, cut=None):
        variable_x = 'B0_CMS3_weMissM2'
        variable_y = 'p_D_l'
        xedges = np.linspace(-2, 10, 48)
        yedges = np.linspace(0.4, 4.6, 42)

        n_rows,n_cols = [3,3]
        fig,axs=plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(16,12),sharex=True, sharey='all')
        fig.suptitle(f'Signal MC ({cut})',fontsize=18)
        fig.supylabel('$|p_D|\ +\ |p_l|\ \ \ [GeV]$', x=0.05,fontsize=18)
        fig.supxlabel('$M_{miss}^2\ \ \ [GeV^2/c^4]$',fontsize=18)
        i=0
        j=0
        for name, sample in self.samples.items():
            if len(sample)==0:
                continue
            (counts, xedges, yedges) = np.histogram2d(
                            sample.query(cut)[variable_x] if cut else sample[variable_x], 
                            sample.query(cut)[variable_y] if cut else sample[variable_y],
                            bins=[xedges, yedges])
            counts = counts.T
            X, Y = np.meshgrid(xedges, yedges)
            im=axs[i,j].pcolormesh(X, Y, counts, cmap='rainbow', norm=colors.LogNorm())
            axs[i,j].grid()
            axs[i,j].set_xlim(xedges.min(),xedges.max())
            axs[i,j].set_ylim(yedges.min(),yedges.max())
            axs[i,j].set_title(name,fontsize=14)
            fig.colorbar(im,ax=axs[i,j])
            j+=1
            if j==3:
                i+=1
                j=0

    def plot_overlaid_signals(self, variable, cut=None):
        fig,axs =plt.subplots(1,2,figsize=(12,5), sharex=True, sharey=False)
        fig.suptitle(f'Overlaid signals with pre-selection', y=1,fontsize=16)
        fig.supylabel('# of candidates per bin',x=0.06,fontsize=16)
        #fig.supxlabel('$|\\vec{p_D}|\ +\ |\\vec{p_l}|$  [GeV/c]')
        #fig.supxlabel('$M_{miss}^2 \ [GeV^2/c^4]$')
        fig.supxlabel(f'{variable}',fontsize=16)

        for sample_name, sample in self.samples.items():
            (counts, bins) = np.histogram(
                sample.query(cut)[variable] if cut else sample[variable], bins=50)
            factor=1
            if sample_name in [r'$D\tau\nu$',r'$D^\ast\tau\nu$',r'$D^{\ast\ast}\tau\nu$']:
                axs[0].hist(bins[:-1], bins, weights=factor*counts,label=sample_name,**self.kwarg)
                axs[0].legend()
            elif sample_name in [r'$D\ell\nu$',r'$D^\ast\ell\nu$',r'$D^{\ast\ast}\ell\nu$']:
                axs[1].hist(bins[:-1], bins, weights=factor*counts,label=sample_name,**self.kwarg)
                axs[1].legend()

        axs[0].set_title('signals')
        axs[1].set_title('normalization')
        axs[0].grid()
        axs[1].grid()

    def plot_overlaid_all(self,variable,cut=None):
        fig,axs =plt.subplots(sharex=True, sharey=False)
        for sample_name, sample in self.samples.items():
            var_col= sample.query(cut)[variable] if cut else sample[variable]
            (counts, bins) = np.histogram(var_col, bins=50)
            factor=1
            if sample_name in [r'$D\tau\nu$',r'$D^\ast\tau\nu$',r'$D^{\ast\ast}\tau\nu$']:
                axs.hist(bins[:-1], bins, weights=factor*counts,
                         label=f'{sample_name} \n{self.statistics(var_col)}',**self.kwarg)
                
            elif sample_name in [r'$D\ell\nu$',r'$D^\ast\ell\nu$',r'$D^{\ast\ast}\ell\nu$']:
                axs.hist(bins[:-1], bins, weights=factor*counts,
                         label=f'{sample_name} \n{self.statistics(var_col)}',**self.kwarg)
            else:
                axs.hist(bins[:-1], bins, weights=factor*counts,
                         label=f'{sample_name} \n{self.statistics(var_col)}',**self.kwarg)

        axs.set_title('Overlaid signals with pre-selection')
        axs.set_xlabel(f'{variable}')
        axs.set_ylabel('# of candidates per bin')
        axs.grid()
        plt.legend(bbox_to_anchor=(1,1.1),ncol=3, fancybox=True, shadow=True,labelspacing=1.5)

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
        sig = pd.concat([self[i] for i in sigModes])
        bkg = pd.concat([self[i] for i in bkgModes])
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

        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('Signal Probability')
        ax1.set_ylabel('FOM', color=color)
        ax1.errorbar(x=test_points, y=BDT_FOM, yerr=BDT_FOM_err,marker='o',label='FOM',color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid()
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:red'
        ax2.set_ylabel('Efficiency', color=color)  # we already handled the x-label with ax1
        ax2.errorbar(x=test_points, y=BDT_sigEff, yerr=BDT_sigEff_err,marker='o',label='Signal Efficiency',color=color)
        ax2.errorbar(x=test_points, y=BDT_bkgEff, yerr=BDT_bkgEff_err,marker='o',label='Bkg Efficiency',color='green')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend()

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title(f'FOM for {variable=}')
        plt.xlim(0,1)
        plt.ylim(bottom=0)
        plt.show()


# +
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class plyex:
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
                          legend=dict(orientation='h',title='',x=1,y=1.1,xanchor='right',yanchor='bottom'))

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
        
    def hist2d(self, facet=False):
        # Define number of colors to generate
        color_sequence = ['rgb(255,255,255)'] + px.colors.sequential.Rainbow[1:]
        num_colors = 9
        # Generate colors with uniform spacing and Rainbow color scale
        my_colors = [[i/(num_colors-1), color_sequence[i]] for i in range(num_colors)]

        # Create a 2d histogram
        fig = px.density_heatmap(self.df, x="B0_CMS3_weMissM2", y="p_D_l",
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

# +
## dataframe samples
import pandas as pd
def get_dataframe_samples(df):
    samples = {}
    files = ['sigDDst', 'normDDst','bkgDststp_tau', 'bkgDstst0_tau','bkgDstst0_ell']

    Dstst_e_nu_selection = f'DecayMode=={DecayMode["all_Dstst_e_nu"].value} and \
                            D_mcPDG*e_mcPDG==411*11 and e_genMotherPDG==B0_mcPDG and \
            ((B0_mcErrors<64 and B0_mcPDG*D_mcPDG==-511*411) or (B0_mcErrors<512 and abs(B0_mcPDG)==521))'

    Dstst_tau_nu_selection = f'DecayMode=={DecayMode["all_Dstst_tau_nu"].value} and \
                            D_mcPDG*e_mcPDG==411*11 and e_mcPDG*e_genMotherPDG==11*15 and \
            ((B0_mcErrors<64 and B0_mcPDG*D_mcPDG==-511*411) or (B0_mcErrors<512 and abs(B0_mcPDG)==521))'

    signals_selection = 'B0_mcPDG*D_mcPDG==-511*411 and D_mcPDG*e_mcPDG==411*11 and e_mcPDG*e_genMotherPDG==11*15'
    norms_selection = 'B0_mcPDG*D_mcPDG==-511*411 and D_mcPDG*e_mcPDG==411*11 and e_genMotherPDG==B0_mcPDG'

    # Sig components
    sig_D_tau_nu=df.query(f'DecayMode=={DecayMode["sig_D_tau_nu"].value} and \
                                            B0_mcErrors<32 and {signals_selection}').copy()

    sig_Dst_tau_nu=df.query(f'DecayMode=={DecayMode["sig_Dst_tau_nu"].value} and \
                                            B0_mcErrors<64 and {signals_selection}').copy()
    samples[r'$D\tau\nu$'] = sig_D_tau_nu
    samples[r'$D^\ast\tau\nu$'] = sig_Dst_tau_nu

    sig_D_e_nu=df.query(f'DecayMode=={DecayMode["sig_D_e_nu"].value} and \
                                    B0_mcErrors<16 and {norms_selection}').copy()
    sig_Dst_e_nu=df.query(f'DecayMode=={DecayMode["sig_Dst_e_nu"].value} and \
                                        B0_mcErrors<64 and {norms_selection}').copy()
    samples[r'$D\ell\nu$'] = sig_D_e_nu
    samples[r'$D^\ast\ell\nu$'] = sig_Dst_e_nu

    Dstst_tau_nu=df.query(Dstst_tau_nu_selection).copy()
    samples[r'$D^{\ast\ast}\tau\nu$'] = Dstst_tau_nu

    Dstst_e_nu=df.query(Dstst_e_nu_selection).copy()
    samples[r'$D^{\ast\ast}\ell\nu$'] = Dstst_e_nu


    #sig_D_mu_nu=df.query('DecayMode=="sig_D_mu_nu" and B0_mcErrors<16').copy()
    #sig_Dst_mu_nu=df.query('DecayMode=="sig_Dst_mu_nu" and (16<=B0_mcErrors<32 or B0_mcErrors<8)').copy()
    #all_Dstst_mu_nu=df.query('DecayMode=="all_Dstst_mu_nu" and (16<=B0_mcErrors<64 or B0_mcErrors<8)').copy()

    #Bkg components
    bkg_fakeD = df.query('abs(D_mcPDG)!=411 and B0_mcErrors!=512 and B0_isContinuumEvent!=1').copy()
    bkg_fakeTracksClusters = df.query('B0_mcErrors==512 and B0_isContinuumEvent!=1').copy()
    bkg_fakeDTC = pd.concat([bkg_fakeD, bkg_fakeTracksClusters])
    samples[r'bkg_fakeDTC'] = bkg_fakeDTC

    bkg_combinatorial = df.query('B0_mcPDG==300553 and abs(D_mcPDG)==411 and B0_mcErrors!=512 and B0_isContinuumEvent!=1').copy()
    bkg_sigOtherBDTaudecay = df.query(f'(DecayMode=={DecayMode["bkg"].value} or \
                 DecayMode=={DecayMode["sig_D_mu_nu"].value} or DecayMode=={DecayMode["sig_Dst_mu_nu"].value} or \
                 DecayMode=={DecayMode["all_Dstst_mu_nu"].value}) and B0_mcPDG!=300553 and \
                 abs(D_mcPDG)==411 and B0_mcErrors!=512 and B0_isContinuumEvent!=1').copy()
    bkg_fakeB = pd.concat([bkg_combinatorial, bkg_sigOtherBDTaudecay])
    samples[r'bkg_fakeB'] = bkg_fakeB

    bkg_continuum = df.query('B0_isContinuumEvent==1').copy()
    samples[r'bkg_continuum'] = bkg_continuum

    bkg_others = pd.concat([df,
                            sig_D_e_nu,
                            sig_D_tau_nu,
                            sig_Dst_e_nu,
                            sig_Dst_tau_nu,
                            Dstst_e_nu,
                            Dstst_tau_nu,
                            bkg_fakeDTC,
                            bkg_fakeB,
                            bkg_continuum]).drop_duplicates(keep=False)
    samples[r'bkg_others'] = bkg_others
    
    for name, df in samples.items():
        df['mode']=name

    df = pd.concat([df for df in samples.values()])
    df['p_D_l_region'] = np.where(df['p_D_l']>2.5,1,0)
    
    return df, samples

    # Weird! the bkg_others contains some events with
    # correct sig decay hash chain and correct B0_mcPDG, D_mcPDG, e_genMotherPDG,
    # but with 128< B0_mcErrors < 256 (misID)
    
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
## template
# shape y, x
template_2d_shape = [41,47]

# 2d indices of non-zero elements in the template and data
template_indices_non_zero = (np.array([ 0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,
        2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  4,
        4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,
        5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,
        6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,
        7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
        8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
        8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,
        9,  9,  9,  9,  9,  9,  9,  9,  9, 10, 10, 10, 10, 10, 10, 10, 10,
       10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11,
       11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
       11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
       12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13,
       13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
       13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14,
       14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
       15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
       15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16,
       16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
       16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
       17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18,
       18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
       18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
       19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20,
       20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21,
       21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
       21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
       22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
       23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24,
       24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25,
       25, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26,
       26, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 27,
       27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
       28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
       29, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31, 31,
       31, 31, 31, 31, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32, 32, 32,
       32, 32, 32, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 34, 34, 34,
       34, 34, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35,
       36, 36, 36, 36, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 37, 37, 37,
       38, 38, 38, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 40,
       40, 40, 40, 40, 40]), np.array([43, 44, 45, 46, 39, 40, 41, 42, 43, 44, 45, 46, 36, 37, 38, 39, 40,
       41, 42, 43, 44, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 30,
       31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 29, 30, 31,
       32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 25, 26, 27, 28, 29,
       30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,  9, 22, 23, 24,
       25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
       19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
       36, 37, 38, 39, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
       30, 31, 32, 33, 34, 35, 36, 37, 38, 13, 14, 15, 16, 17, 18, 19, 20,
       21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 11,
       12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
       29, 30, 31, 32, 33, 34,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
       19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,  5,  6,
        7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
       24, 25, 26, 27, 28, 29, 30, 31,  4,  5,  6,  7,  8,  9, 10, 11, 12,
       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
       20, 21, 22, 23, 24, 25, 26, 27, 28, 29,  0,  2,  3,  4,  5,  6,  7,
        8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
       25, 26, 27, 28,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,  0,  1,  2,  3,
        4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
       21, 22, 23, 24,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,  0,  1,  2,  3,  4,  5,
        6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,  0,
        1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,
       14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
       11, 12, 13, 14, 15, 16, 17, 18,  0,  1,  2,  3,  4,  5,  6,  7,  8,
        9, 10, 11, 12, 13, 14, 15, 16, 17,  0,  1,  2,  3,  4,  5,  6,  7,
        8,  9, 10, 11, 12, 13, 14, 15, 16,  0,  1,  2,  3,  4,  5,  6,  7,
        8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  6,  7,  8,
        9, 10, 11, 12, 13, 14,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
       11, 12, 13, 15,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
       13,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,  0,  1,  2,
        3,  4,  5,  6,  7,  8,  9, 10, 11,  0,  1,  2,  3,  4,  5,  6,  7,
        8,  9, 10,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,  0,  1,  2,
        3,  4,  5,  6,  7,  8,  9,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
        0,  1,  2,  3,  4,  5,  6,  7,  8,  1,  2,  3,  4,  5,  6,  7,  8,
        0,  1,  2,  3,  4,  5,  6,  7,  8,  1,  2,  3,  4,  5,  6,  7,  0,
        1,  4,  5,  6,  7]))


indices_non_zero_small_pDl = (np.array([ 0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,
         2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  4,
         4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,
         5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,
         6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,
         7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
         8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
         8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,
         9,  9,  9,  9,  9,  9,  9,  9,  9, 10, 10, 10, 10, 10, 10, 10, 10,
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11,
        11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
        11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
        12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13,
        13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
        13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14,
        14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
        17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18,
        18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
        18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
        19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19]),
 np.array([43, 44, 45, 46, 39, 40, 41, 42, 43, 44, 45, 46, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 30,
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,  9, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        36, 37, 38, 39, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 36, 37, 38, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
        29, 30, 31, 32, 33, 34,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,  5,  6,
         7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31,  4,  5,  6,  7,  8,  9, 10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
         3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29,  0,  2,  3,  4,  5,  6,  7,
         8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,  0,  1,  2,  3,
         4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]))

indices_non_zero_large_pDl = (np.array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,
         2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,
         3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,
         4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,
         5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,
         6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,
         7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,
         8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9,
         9,  9,  9,  9,  9,  9,  9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
        10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12,
        12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13,
        13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17,
        17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19,
        19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20]),
 np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20,  0,  1,  2,  3,  4,  5,  6,  7,
         8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  0,  1,  2,  3,  4,
         5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,  0,  1,  2,
         3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,  0,  1,
         2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,  0,  1,
         2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,
         3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,  0,  1,  2,  3,  4,
         5,  6,  7,  8,  9, 10, 11, 12, 13, 15,  0,  1,  2,  3,  4,  5,  6,
         7,  8,  9, 10, 11, 12, 13,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
        10, 11, 12,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,  0,  1,
         2,  3,  4,  5,  6,  7,  8,  9, 10,  0,  1,  2,  3,  4,  5,  6,  7,
         8,  9, 10,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  0,  1,  2,  3,
         4,  5,  6,  7,  8,  9,  0,  1,  2,  3,  4,  5,  6,  7,  8,  1,  2,
         3,  4,  5,  6,  7,  8,  0,  1,  2,  3,  4,  5,  6,  7,  8,  1,  2,
         3,  4,  5,  6,  7,  0,  1,  4,  5,  6,  7]))
