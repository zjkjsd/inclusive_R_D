# +

import sys
import basf2_mva
import basf2_mva_util
import multiprocessing
import itertools
from sklearn.metrics import roc_auc_score
import pandas as pd


if __name__ == "__main__":
    from basf2 import conditions
    # NOTE: do not use testing payloads in production! Any results obtained like this WILL NOT BE PUBLISHED
    conditions.testing_payloads = ['localdb/database.txt']
    
    
    mva_ver = sys.argv[1] #'2_1'
    training_data = basf2_mva.vector(f"output_train.root")
    test_data = basf2_mva.vector(f"output_test.root")
    identifier = f'MVA{mva_ver}_FastBDT.xml'

    if mva_ver in ['2_1', '2_1_withCut']:
        variables = ['B0_Lab5_weMissPTheta',
                     #'B0_vtxDDSig',
                     'B0_vtxReChi2',
                     'B0_flightDistanceSig',
                     'B0_sig_DistanceSig',
                     'B0_sig_daughterAngleLab',
                     'B0_roeD_DistanceSig_vtx',      
                     'B0_roel_DistanceSig_dis',    
                     #'B0_roeNeextra_my_mask',
                     'B0_roeEextra_my_mask',
                     'B0_roeMbc_my_mask',
                     'B0_roeDeltae_my_mask',
                     'B0_roeCharge_my_mask',
                     #'B0_nROE_Charged_my_mask',
                     'B0_nROE_NeutralECLClusters_my_mask',
                     #'B0_nROE_NeutralHadrons_my_mask',
                     'B0_nROE_Tracks_my_mask',
                     #'B0_nROE_pi',
                     'B0_nROE_K',
                     #'B0_nROE_ECLClusters_my_mask',
                     'B0_nROE_Photons_my_mask',
                     'B0_nROE_KLMClusters',
                     'B0_vetoeID',
                     'B0_vetomuID',
                     'B0_TagVReChi2IP',
                     'MVA1_5_output',
                    ]
    
    elif mva_ver in ['2_0', '2_0_withCut']:
        variables = ['B0_Lab5_weMissPTheta',
                     #'B0_vtxDDSig',
                     'B0_vtxReChi2',
                     'B0_flightDistanceSig',
                     'B0_sig_DistanceSig',
                     'B0_sig_daughterAngleLab',
                     'B0_roeD_DistanceSig_vtx',      
                     'B0_roel_DistanceSig_dis',    
                     #'B0_roeNeextra_my_mask',
                     'B0_roeEextra_my_mask',
                     'B0_roeMbc_my_mask',
                     'B0_roeDeltae_my_mask',
                     'B0_roeCharge_my_mask',
                     #'B0_nROE_Charged_my_mask',
                     'B0_nROE_NeutralECLClusters_my_mask',
                     'B0_nROE_NeutralHadrons_my_mask',
                     'B0_nROE_Tracks_my_mask',
                     #'B0_nROE_pi',
                     'B0_nROE_K',
                     #'B0_nROE_ECLClusters_my_mask',
                     'B0_nROE_Photons_my_mask',
                     #'B0_nROE_KLMClusters',
                     'B0_vetoeID',
                     'B0_vetomuID',
                     'B0_TagVReChi2IP',
                    ]
        
    elif mva_ver in ['2_2', '2_2_withCut']:
        variables = ['B0_Lab5_weMissPTheta',
                     #'B0_vtxDDSig',
                     'B0_vtxReChi2',
                     'B0_flightDistanceSig',
                     'B0_sig_DistanceSig',
                     'B0_sig_daughterAngleLab',
                     'B0_roeD_DistanceSig_vtx',      
                     'B0_roel_DistanceSig_dis',    
                     #'B0_roeNeextra_my_mask',
                     'B0_roeEextra_my_mask',
                     'B0_roeMbc_my_mask',
                     'B0_roeDeltae_my_mask',
                     'B0_roeCharge_my_mask',
                     #'B0_nROE_Charged_my_mask',
                     'B0_nROE_NeutralECLClusters_my_mask',
                     'B0_nROE_NeutralHadrons_my_mask',
                     'B0_nROE_Tracks_my_mask',
                     #'B0_nROE_pi',
                     'B0_nROE_K',
                     #'B0_nROE_ECLClusters_my_mask',
                     'B0_nROE_Photons_my_mask',
                     #'B0_nROE_KLMClusters',
                     'B0_vetoeID',
                     'B0_vetomuID',
                     'B0_TagVReChi2IP',
                     'D_K_kaonID',
                     'D_pi1_kaonID',
                     'D_pi2_kaonID',
                     #'D_K_pionID',
                     'D_pi1_pionID',
                     'D_pi2_pionID',
                     'D_K_dr', 
                     'D_pi1_dr', 
                     'D_pi2_dr',
                     'D_K_dz', 
                     'D_pi1_dz', 
                     'D_pi2_dz',
                     'D_K_nCDCHits',
                     'D_pi1_nCDCHits',
                     'D_pi2_nCDCHits', 
                     'D_vtxReChi2', 
                     'D_flightDistanceSig',
                     'D_BFInvM',
                    ]
    
    else:
        print('The current available versions are 2_0 or 2_1 or 2_2')
    
    spectators = ['D_CMS_p', 'e_CMS_p', 'B0_CMS2_weMissM2']

    general_options = basf2_mva.GeneralOptions()
    general_options.m_datafiles = training_data
    general_options.m_treename = "B0"
    general_options.m_identifier = identifier  # outputted weightfile
    general_options.m_variables = basf2_mva.vector(*variables)
    general_options.m_spectators = basf2_mva.vector(*spectators)
    general_options.m_target_variable = 'isSignal'



    import sys
    nTrees = sys.argv[2] #1400
    depth = sys.argv[3] #2
    learning_rate = sys.argv[4] #0.05
    nCutLevels = sys.argv[5] #6
    sub_sample_fraction = sys.argv[6] #0.5

    sp = basf2_mva.FastBDTOptions()
    sp.m_nTrees = int(nTrees)
    sp.m_nLevels = int(depth)
    sp.m_shrinkage = float(learning_rate)
    sp.m_nCuts = int(nCutLevels)
    sp.m_randRatio = float(sub_sample_fraction)

    basf2_mva.teacher(general_options, sp)

    #apply the trained mva method onto data
    basf2_mva.expert(basf2_mva.vector(identifier),  # weightfile
                     test_data,
                     'B0', f'MVA{mva_ver}_applied.root')

    #FastBDT evaluation
    import subprocess
    command = f'basf2_mva_evaluate.py -id MVA{mva_ver}_FastBDT.xml -train output_train.root -data output_test.root --treename B0 -o MVA{mva_ver}_evaluation_{nTrees}_{depth}_{learning_rate}_{nCutLevels}_{sub_sample_fraction}.zip'
    subprocess.call(command, shell=True)
# -


