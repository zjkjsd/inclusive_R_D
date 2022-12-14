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
    # the weightfile has to exist before runing the grid_search,
    # otherwise segmentation violation occurs

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


    def grid_search(hyperparameters):
        nTrees, depth, learning_rate, nCutLevels, sub_sampling_rate = hyperparameters
        method = basf2_mva_util.Method(general_options.m_identifier)
        options = basf2_mva.FastBDTOptions()

        options.m_nTrees = nTrees
        options.m_nLevels = depth
        options.m_shrinkage = learning_rate
        options.m_nCuts = nCutLevels
        options.m_randRatio = sub_sampling_rate

        m = method.train_teacher(training_data, general_options.m_treename, specific_options=options)
        p0, t0 = m.apply_expert(training_data, general_options.m_treename)
        p1, t1 = m.apply_expert(test_data, general_options.m_treename)
        training_auc_basf2 = basf2_mva_util.calculate_auc_efficiency_vs_background_retention(p0, t0)
        training_auc_sklearn = roc_auc_score(t0, p0)
        test_auc_basf2 = basf2_mva_util.calculate_auc_efficiency_vs_background_retention(p1, t1)
        test_auc_sklearn = roc_auc_score(t1, p1)
        return hyperparameters, training_auc_basf2, training_auc_sklearn, test_auc_basf2, test_auc_sklearn

    p = multiprocessing.Pool(None, maxtasksperchild=1)
    results = p.map(grid_search, itertools.product([1200, 1400, 1500], 
                                                   [2, 3], 
                                                   [0.05, 0.06, 0.07],
                                                   [5, 6, 7],
                                                   [0.5]))
    x=0
    d={}
    small_dfs = []
    for hyperparameters, auc0, auc1, auc2, auc3 in results:
        d['nTrees']= hyperparameters[0]
        d['depth']= hyperparameters[1]
        d['learning_rate']= hyperparameters[2]
        d['nCutLevels']= hyperparameters[3]
        d['sub_sampling_rate']= hyperparameters[4]

        d['training_auc_basf2']= auc0
        d['training_auc_sklearn']= auc1
        d['test_auc_basf2']= auc2
        d['test_auc_sklearn']= auc3

        small_df = pd.DataFrame(data=d,index=[x])
        small_dfs.append(small_df)
        x+=1

    large_df = pd.concat(small_dfs)
    large_df.to_csv(f'MVA{mva_ver}_grid_search_result_1.csv')



    #import sys
    #nTrees = sys.argv[1] #200
    #nLevels = sys.argv[2] #3
    #shrinkage = sys.argv[3] #0.1

    #sp = basf2_mva.FastBDTOptions()
    #sp.m_nTrees = int(nTrees)
    #sp.m_nLevels = int(nLevels)
    #sp.m_shrinkage = float(shrinkage)

    #basf2_mva.teacher(general_options, sp)

    # apply the trained mva method onto data
    #basf2_mva.expert(basf2_mva.vector(identifier),  # weightfile
    #                 test_data,
    #                 'B0', 'Test_BS_applied.root')

    # FastBDT evaluation
    #import subprocess
    #command = f'basf2_mva_evaluate.py -id MVAFastBDT.xml -train output_train.root -data output_test.root --treename B0 -o evaluation_{nTrees}_{nLevels}_{shrinkage}.zip'
    #subprocess.call(command, shell=True)
# -


