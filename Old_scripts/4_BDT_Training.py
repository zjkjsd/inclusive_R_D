# +
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BDT training script.

Usage: python3 4_BDT_Training.py -d --bdt --train --test (--treename --target --hyper)

Example: python3 4_BDT_Training.py -d folder --bdt 2_1 --train CS_train.root --test CS_test.root \
                                             (--treename B0 --target Signal --hyper 800 2 0.05 4 0.5 -1)


basf2_mva_merge_mc -s MC14ri_signal.root -b MC14ri_qqbar.root MC14ri_taupair.root \
                    --columns __weight__ B0_R2 B0_thrustBm B0_thrustOm B0_cosTBTO B0_cosTBz \
                    B0_KSFWV1 B0_KSFWV2 B0_KSFWV3 B0_KSFWV4 B0_KSFWV5 B0_KSFWV6 B0_KSFWV7 \
                    B0_KSFWV8 B0_KSFWV9 B0_KSFWV10 B0_KSFWV11 B0_KSFWV12 B0_KSFWV13 B0_KSFWV14 \
                    B0_KSFWV15 B0_KSFWV16 B0_KSFWV17 B0_KSFWV18 B0_CC1 B0_CC2 B0_CC3 B0_CMS3_weQ2lnuSimple \
                    B0_CC4 B0_CC5 B0_CC6 B0_CC7 B0_CC8 B0_CC9 D_CMS_p e_CMS_p B0_CMS3_weMissM2 \
                    -o CS -t B0 --ftest 0.2 --random_state 0 (--fsig 0.3)
                    
basf2_mva_merge_mc -s MC14ri_signal.root -b MC14ri_DTCFake.root \
                    --columns __weight__ D_CMS_p e_CMS_p B0_CMS3_weMissM2 B0_CMS3_weQ2lnuSimple \
                    D_K_kaonID_binary_noSVD D_pi1_kaonID_binary_noSVD D_pi2_kaonID_binary_noSVD \
                    D_K_dr D_pi1_dr D_pi2_dr D_K_dz D_pi1_dz D_pi2_dz D_K_pValue D_pi1_pValue \
                    D_pi2_pValue D_vtxReChi2 D_BFInvM D_A1FflightDistanceSig_IP D_daughterInvM_0_1 \
                    D_daughterInvM_1_2 B0_vtxDDSig -o DTCFake -t B0 --ftest 0.2 --random_state 0 (--fsig 0.3)
                    
basf2_mva_merge_mc -s MC14ri_signal.root -b MC14ri_BFake.root \
                    --columns __weight__ D_CMS_p e_CMS_p B0_CMS3_weMissM2 B0_CMS3_weQ2lnuSimple \
                    B0_Lab5_weMissPTheta B0_vtxDDSig B0_vtxReChi2 B0_flightDistanceSig B0_sig_DistanceSig \
                    B0_nROE_NeutralHadrons_my_mask B0_roel_DistanceSig_dis B0_roeDeltae_my_mask B0_nROE_K \
                    B0_roeEextra_my_mask B0_roeMbc_my_mask B0_roeCharge_my_mask B0_TagVReChi2IP \
                    B0_nROE_NeutralECLClusters_my_mask B0_nROE_Tracks_my_mask B0_nROE_Photons_my_mask \
                    -o BFake -t B0 --ftest 0.2 --random_state 0 (--fsig 0.3)
                    
basf2_mva_merge_mc -s MC14ri_signal_applied.root -b MC14ri_BFake_applied.root MC14ri_qqbar_applied.root \
                    MC14ri_taupair_applied.root MC14ri_DTCFake_applied.root \
                    --columns __weight__ D_CMS_p e_CMS_p B0_CMS3_weMissM2 B0_CMS3_weQ2lnuSimple \
                    CS__slMVA1_FastBDT__ptxml DTCFake__slMVA2_1_FastBDT__ptxml BFake__slMVA2_2_FastBDT__ptxml \
                    -o AllBkg -t B0 --ftest 0.2 --random_state 0 --fsig 0.2
         
"""

import argparse
import subprocess
import shlex
from termcolor import colored
from basf2 import conditions
import basf2_mva
#import basf2_mva_util
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
                        choices=['1', '2_1','2_2','3'],
                        help="BDT number, 1 for CS, 2_1 for fakeDTC, 2_2 for fakeB, 3 for all BB bkg")
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
    parser.add_argument("--hyper",
                        action="store",
                        nargs='+',
                        default=['800', '2', '0.05', '4', '0.5', '-1'],
                        required=False,
                        help="Hyperparameters: nTrees, depth, learning_rate, nCutLevels, \
                              sub_sample_fraction, flatnessLoss")

    return parser


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
                    
    elif args.bdt=='3':
        variables = ['CS__slMVA1_FastBDT__ptxml',   'DTCFake__slMVA2_1_FastBDT__ptxml',
                     'BFake__slMVA2_2_FastBDT__ptxml']
    
    # define training configrations
    general_options = basf2_mva.GeneralOptions()
    general_options.m_datafiles = training_data
    general_options.m_treename = args.treename      # "B0"
    general_options.m_identifier = identifier       # outputted weightfile
    general_options.m_variables = basf2_mva.vector(*variables)
    general_options.m_spectators = basf2_mva.vector(*spectators)
    general_options.m_target_variable = args.target # 'Signal'

    sp = basf2_mva.FastBDTOptions()
    sp.m_nTrees = int(args.hyper[0])
    sp.m_nLevels = int(args.hyper[1])     #depth
    sp.m_shrinkage = float(args.hyper[2]) #learning rate
    sp.m_nCuts = int(args.hyper[3])       #nCutLevels
    sp.m_randRatio = float(args.hyper[4]) #sub_sample_fraction
    sp.m_flatnessLoss = float(args.hyper[5])

    # perform the training
    print(colored(f'Training...', 'magenta'))
    basf2_mva.teacher(general_options, sp)

    #apply the trained mva method onto data
    print(colored(f'Applying the trained weights on the test set', 'blue'))
    basf2_mva.expert(basf2_mva.vector(identifier), test_data, args.treename, test_applied)

    #FastBDT evaluation
    eval_file = f"evaluation_{args.test.strip('test.root')+'_'.join(args.hyper)}.pdf"
    print(colored(f'Evaluating the performance, results will be saved in {eval_file}', 'green'))
    command = f'basf2_mva_evaluate.py -id MVA{args.bdt}_FastBDT.xml -train {args.train} \
                -data {args.test} --treename {args.treename} -c -o {eval_file}'
    subprocess.run(shlex.split(command), check=True, text=True, cwd=args.dir)
