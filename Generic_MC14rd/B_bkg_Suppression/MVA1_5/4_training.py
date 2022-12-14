# +


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
    
    
    mva_ver = '1_5'
    training_data = basf2_mva.vector("output_train.root")
    test_data = basf2_mva.vector("output_test.root")
    identifier = f'MVA{mva_ver}_FastBDT.xml'

    variables = ['D_K_kaonID_binary_noSVD',
                 'D_pi1_kaonID_binary_noSVD',
                 'D_pi2_kaonID_binary_noSVD',
                 'D_K_dr', 
                 'D_pi1_dr', 
                 'D_pi2_dr',
                 'D_K_dz', 
                 'D_pi1_dz', 
                 'D_pi2_dz',
                 'D_K_pValue', 
                 'D_pi1_pValue', 
                 'D_pi2_pValue',
                 'D_vtxReChi2', 
                 'D_BFInvM',
                 'D_A1FflightDistanceSig_IP',
                 'D_daughterInvM_0_1',
                 'D_daughterInvM_1_2',]
    
    spectators = ['D_CMS_p', 'e_CMS_p', 'B0_CMS2_weMissM2']

    general_options = basf2_mva.GeneralOptions()
    general_options.m_datafiles = training_data
    general_options.m_treename = "B0"
    general_options.m_identifier = identifier  # outputted weightfile
    general_options.m_variables = basf2_mva.vector(*variables)
    general_options.m_spectators = basf2_mva.vector(*spectators)
    general_options.m_target_variable = 'isSignal'



    import sys
    nTrees = sys.argv[1] #1400
    depth = sys.argv[2] #2
    learning_rate = sys.argv[3] #0.05
    nCutLevels = sys.argv[4] #6
    sub_sample_fraction = sys.argv[5] #0.5

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


