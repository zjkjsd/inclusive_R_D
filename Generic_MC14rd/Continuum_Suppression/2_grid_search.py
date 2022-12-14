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


    mva_ver = '1'
    training_data = basf2_mva.vector("output_train.root")
    test_data = basf2_mva.vector("output_test.root")
    identifier = f'MVA{mva_ver}_FastBDT.xml' 
    # the weightfile has to exist before runing the grid_search,
    # otherwise segmentation violation occurs

    variables = ["B0_R2",
                 "B0_thrustBm",
                 "B0_thrustOm",
                 "B0_cosTBTO",
                 "B0_cosTBz",
                 "B0_KSFWV1",
                 "B0_KSFWV2",
                 "B0_KSFWV3",
                 "B0_KSFWV4",
                 "B0_KSFWV5",
                 "B0_KSFWV6",
                 "B0_KSFWV7",
                 "B0_KSFWV8",
                 "B0_KSFWV9",
                 "B0_KSFWV10",
                 "B0_KSFWV11",
                 "B0_KSFWV12",
                 "B0_KSFWV13",
                 "B0_KSFWV14",
                 "B0_KSFWV15",
                 "B0_KSFWV16",
                 "B0_KSFWV17",
                 "B0_KSFWV18",
                 "B0_CC1",
                 "B0_CC2",
                 "B0_CC3",
                 "B0_CC4",
                 "B0_CC5",
                 "B0_CC6",
                 "B0_CC7",
                 "B0_CC8",
                 "B0_CC9",]
    
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
    results = p.map(grid_search, itertools.product([1200, 1400, 1500, 1600], 
                                                   [2], 
                                                   [0.05, 0.06, 0.07],
                                                   [4, 5, 6],
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


