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


