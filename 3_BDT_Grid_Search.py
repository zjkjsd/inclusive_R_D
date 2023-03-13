# +
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid search for BDT hyperparameters.

Usage: python3 3_BDT_Grid_Search.py -d --bdt --train --test (--treename --target)

Example: python3 3_BDT_Grid_Search.py -d folder --bdt 2_1 --train CS_train.root --test CS_test.root \
                                             (--treename B0 --target Signal)
         
"""

import argparse
import subprocess
import shlex
import multiprocessing
import itertools
import pandas
from termcolor import colored
from basf2 import conditions
import basf2_mva
import basf2_mva_util
from sklearn.metrics import roc_auc_score

def argparser():
    """
    Parse options as command-line arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-d', "--dir",
                        action="store",
                        type=str,
                        required=True,
                        help="Relative path to the data sets")
    parser.add_argument("--bdt",
                        action="store",
                        type=str,
                        required=True,
                        choices=['1', '2_1','2_2','2_3'],
                        help="BDT number, 1 for CS, 2_1 for fakeDTC, 2_2 for fakeB, 2_3 for all BB bkg")
    parser.add_argument("--train",
                        action="store",
                        type=str,
                        required=True,
                        help="Training root file")
    parser.add_argument("--test",
                        action="store",
                        type=str,
                        required=True,
                        help="Test root file")
    parser.add_argument("--treename",
                        action="store",
                        type=str,
                        default='B0',
                        required=False,
                        help="Treename in root files")
    parser.add_argument("--target",
                        action="store",
                        type=str,
                        default='Signal',
                        required=False,
                        help="The target variable in root files")

    return parser


def grid_search(hyperparameters):
    nTrees, depth, learning_rate, nCutLevels, sub_sampling_rate, flatnessLoss = hyperparameters
    method = basf2_mva_util.Method(general_options.m_identifier)
    options = basf2_mva.FastBDTOptions()

    options.m_nTrees = nTrees
    options.m_nLevels = depth
    options.m_shrinkage = learning_rate
    options.m_nCuts = nCutLevels
    options.m_randRatio = sub_sampling_rate
    options.m_flatnessLoss = flatnessLoss
    

    m = method.train_teacher(training_data, general_options.m_treename, specific_options=options)
    p0, t0 = m.apply_expert(training_data, general_options.m_treename)
    p1, t1 = m.apply_expert(test_data, general_options.m_treename)
    training_auc_basf2 = basf2_mva_util.calculate_auc_efficiency_vs_background_retention(p0, t0)
    training_auc_sklearn = roc_auc_score(t0, p0)
    test_auc_basf2 = basf2_mva_util.calculate_auc_efficiency_vs_background_retention(p1, t1)
    test_auc_sklearn = roc_auc_score(t1, p1)
    return hyperparameters, training_auc_basf2, training_auc_sklearn, test_auc_basf2, test_auc_sklearn
    
    
if __name__ == "__main__":
    
    args = argparser().parse_args()
    # NOTE: do not use testing payloads in production! Any results obtained like this WILL NOT BE PUBLISHED
    conditions.testing_payloads = ['localdb/database.txt']
    
    # define training and weight files
    print(colored(f'Initializing the configrations', 'blue'))
    training_data = basf2_mva.vector(f'{args.dir}/{args.train}')
    test_data = basf2_mva.vector(f'{args.dir}/{args.test}')
    identifier = f'{args.dir}/MVA{args.bdt}_FastBDT.xml'
    test_applied = f'{args.dir}/MVA{args.bdt}_applied.root'

    # define training variables and spectators
    spectators = ['D_CMS_p', 'e_CMS_p', 'B0_CMS3_weMissM2', 'B0_CMS3_weQ2lnuSimple']
    if args.bdt=='1':
        variables = ["B0_R2",       "B0_thrustOm",   "B0_cosTBTO",    "B0_cosTBz",
                     "B0_KSFWV3",   "B0_KSFWV4",     "B0_KSFWV5",     "B0_KSFWV6",
                     "B0_KSFWV7",   "B0_KSFWV8",     "B0_KSFWV9",     "B0_KSFWV10",
                     "B0_KSFWV13",  "B0_KSFWV14",    "B0_KSFWV15",    "B0_KSFWV16",
                     "B0_KSFWV17",  "B0_KSFWV18",    "B0_CC1",        "B0_CC2",
                     "B0_CC3",      "B0_CC4",        "B0_CC5",        "B0_CC6",
                     "B0_CC7",      "B0_CC8",        "B0_CC9",]
                     #"B0_thrustBm", "B0_KSFWV1",     "B0_KSFWV2",     "B0_KSFWV11",
                     #"B0_KSFWV12",] correlates with mm2 or p_D_l
        
    elif args.bdt=='2_1':
        variables = ['D_K_kaonID_binary_noSVD',    'D_pi1_kaonID_binary_noSVD',
                     'D_pi2_kaonID_binary_noSVD',  'D_K_dr', 
                     'D_pi1_dr',                   'D_pi2_dr',
                     'D_K_dz',                     'D_pi1_dz', 
                     'D_pi2_dz',                   'D_K_pValue', 
                     'D_pi1_pValue',               'D_pi2_pValue',
                     'D_vtxReChi2',                #'D_BFInvM',
                     'D_A1FflightDistanceSig_IP',  'D_daughterInvM_0_1',
                     'D_daughterInvM_1_2',         'B0_vtxDDSig',]
        
    elif args.bdt=='2_2':
        variables = ['B0_Lab5_weMissPTheta',       'B0_vtxDDSig',
                     'B0_vtxReChi2',               'B0_flightDistanceSig',
                     'B0_nROE_Tracks_my_mask',     'B0_nROE_NeutralHadrons_my_mask',
                     'B0_roel_DistanceSig_dis',    'B0_roeDeltae_my_mask',
                     'B0_roeEextra_my_mask',       'B0_roeMbc_my_mask',
                     'B0_roeCharge_my_mask',       'B0_nROE_Photons_my_mask',
                     'B0_nROE_K',                  'B0_TagVReChi2IP',]
                     #'B0_roeNeextra_my_mask',     'B0_sig_DistanceSig', same as vtxDDsig
                     #'B0_sig_daughterAngleLab', correlates with p_D_l
                     #'B0_nROE_Charged_my_mask',   'B0_nROE_pi',
                     #'B0_nROE_NeutralECLClusters_my_mask', same as nPhotons
                     #'B0_nROE_ECLClusters_my_mask','B0_nROE_KLMClusters'
    
    # define training configrations
    general_options = basf2_mva.GeneralOptions()
    general_options.m_datafiles = training_data
    general_options.m_treename = args.treename      # "B0"
    general_options.m_identifier = identifier       # outputted weightfile
    general_options.m_variables = basf2_mva.vector(*variables)
    general_options.m_spectators = basf2_mva.vector(*spectators)
    general_options.m_target_variable = args.target # 'Signal'
    
    # perform grid search
    print(colored(f'Grid Searching...', 'magenta'))
    p = multiprocessing.Pool(None, maxtasksperchild=1)
    results = p.map(grid_search, itertools.product([600, 800, 400], 
                                                   [2, 3], 
                                                   [0.05, 0.07],
                                                   [4, 5],
                                                   [0.5],
                                                   [-1]))

    data_dict={'nTrees':[], 'depth':[], 'learning_rate':[], 'nCutLevels':[], 'sub_sampling_rate':[], 'flatnessLoss':[],
       'training_auc_basf2':[], 'training_auc_sklearn':[], 'test_auc_basf2':[], 'test_auc_sklearn':[]}
    for hyperparameters, auc0, auc1, auc2, auc3 in results:
        data_dict['nTrees'].append(hyperparameters[0])
        data_dict['depth'].append(hyperparameters[1])
        data_dict['learning_rate'].append(hyperparameters[2])
        data_dict['nCutLevels'].append(hyperparameters[3])
        data_dict['sub_sampling_rate'].append(hyperparameters[4])
        data_dict['flatnessLoss'].append(hyperparameters[5])

        data_dict['training_auc_basf2'].append(auc0)
        data_dict['training_auc_sklearn'].append(auc1)
        data_dict['test_auc_basf2'].append(auc2)
        data_dict['test_auc_sklearn'].append(auc3)

    df = pandas.DataFrame(data_dict)
    print(colored(f'Saving results to a csv file', 'green'))
    df.to_csv(f'{args.dir}/MVA{args.bdt}_grid_search_result.csv')
