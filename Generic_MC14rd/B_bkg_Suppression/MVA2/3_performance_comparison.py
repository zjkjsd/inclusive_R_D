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

    
    # Train a MVA method and directly upload it to the database    
    training_data = basf2_mva.vector("output_train.root")
    test_data = basf2_mva.vector("output_test.root")
    
    variables = ['B0_Lab_weMissPTheta',
    'cos_D_l',
    'B0_vtxReChi2',
    'B0_vtxDD',
    'B0_vtxDDSig',
    'D_vtxReChi2',
#    'D_BFM',
    'B0_Distance',
    'B0_DistanceError',
    'B0_DistanceVector_X',
    'B0_DistanceVector_Y',
    'B0_DistanceVector_Z',
    'B0_DistanceCovMatrixXX',
    'B0_DistanceCovMatrixXY',
    'B0_DistanceCovMatrixXZ',
    'B0_DistanceCovMatrixYX',
    'B0_DistanceCovMatrixYY',
    'B0_DistanceCovMatrixYZ',
    'B0_DistanceCovMatrixZX',
    'B0_DistanceCovMatrixZY',
    'B0_DistanceCovMatrixZZ',
    'B0_nROE_Charged_my_mask',
    'B0_nROE_NeutralECLClusters_my_mask',
    'B0_nROE_NeutralHadrons_my_mask',
    'B0_nROE_Tracks_my_mask',
    'B0_nROE_pi',
    'B0_nROE_K',
    'B0_nROE_ECLClusters_my_mask',
    'B0_nROE_Photons_my_mask',
    'B0_nROE_KLMClusters',
    'B0_roeNeextra_my_mask',
    'B0_roeEextra_my_mask',
    'B0_roeMbc_my_mask',
    'B0_roeDeltae_my_mask',
    'B0_roeCharge_my_mask',
    'B0_TagVChi2',
    'B0_TagVChi2IP',
    'B0_TagVNDF']
    
    spectators = ['D_CMS_p', 'e_CMS_p', 'B0_CMS2_weMissM2']
        
    general_options = basf2_mva.GeneralOptions()
    general_options.m_datafiles = training_data
    general_options.m_treename = "B0"
    general_options.m_variables = basf2_mva.vector(*variables)
    general_options.m_spectators = basf2_mva.vector(*spectators)
    general_options.m_target_variable = 'isSignal'
    
    
    
    
    # Hyperparameters for different methods

    trivial_options = basf2_mva.TrivialOptions()

    
    data_options = basf2_mva.FastBDTOptions()
    data_options.m_nTrees = 0

    
    import sys
    nTrees = sys.argv[1] #1400
    depth = sys.argv[2] #2
    learning_rate = sys.argv[3] #0.05
    nCutLevels = sys.argv[4] #6
    sub_sample_fraction = sys.argv[5] #0.5
    
    fastbdt_options = basf2_mva.FastBDTOptions()
    fastbdt_options.m_nTrees = int(nTrees)
    fastbdt_options.m_nCuts = int(nCutLevels)
    fastbdt_options.m_nLevels = int(depth)
    fastbdt_options.m_shrinkage = float(learning_rate)
    fastbdt_options.m_randRatio = float(sub_sample_fraction)

    
    fann_options = basf2_mva.FANNOptions()
    fann_options.m_number_of_threads = 2
    fann_options.m_max_epochs = 100
    fann_options.m_validation_fraction = 0.001
    fann_options.m_test_rate = fann_options.m_max_epochs + 1  # Never test
    fann_options.m_hidden_layers_architecture = "N+1"
    fann_options.m_random_seeds = 1

    
    tmva_bdt_options = basf2_mva.TMVAOptionsClassification()
    tmva_bdt_options.m_config = (f"!H:!V:CreateMVAPdfs:NTrees={nTrees}:BoostType=Grad:Shrinkage={learning_rate}:UseBaggedBoost:"
                                 f"BaggedSampleFraction={sub_sample_fraction}:nCuts=1024:MaxDepth={depth}:IgnoreNegWeightsInTraining")
#    tmva_bdt_options.m_prepareOptions = ("SplitMode=block:V:nTrain_Signal=9691:nTrain_Background=136972:"
#                                         "nTest_Signal=1:nTest_Background=1")

    
    tmva_nn_options = basf2_mva.TMVAOptionsClassification()
    tmva_nn_options.m_type = "MLP"
    tmva_nn_options.m_method = "MLP"
    tmva_nn_options.m_config = ("H:!V:CreateMVAPdfs:VarTransform=N:NCycles=100:HiddenLayers=N+1:TrainingMethod=BFGS")
#    tmva_nn_options.m_prepareOptions = ("SplitMode=block:V:nTrain_Signal=9691:nTrain_Background=136972:"
#                                        "nTest_Signal=1:nTest_Background=1")

    
    sklearn_bdt_options = basf2_mva.PythonOptions()
    sklearn_bdt_options.m_framework = "sklearn"
    param = {"n_estimators": nTrees, "learning_rate": learning_rate, "max_depth": depth, "random_state": 0, "subsample": sub_sample_fraction}
    sklearn_bdt_options.m_config = str(param)

    
    xgboost_options = basf2_mva.PythonOptions()
    xgboost_options.m_framework = "xgboost"
    param = {"max_depth": depth, "eta": learning_rate, "silent": 1, "objective": "binary:logistic",
             "subsample": sub_sample_fraction, "nthread": -1, "nTrees": nTrees}
    xgboost_options.m_config = str(param)

    
    
    hep_ml_uboost_options = basf2_mva.PythonOptions()
    #hep_ml_options.m_steering_file = 'mva/examples/python/hep_ml_uboost.py'
    # Set the parameters of the uBoostClassifier,
    # defaults are 50, which is reasonable, but I want to have a example runtime < 2 minutes
    import json
    hep_ml_uboost_options.m_config = json.dumps({'n_neighbors': 5, 'n_estimators': 5})
    hep_ml_uboost_options.m_framework = 'hep_ml'
    
    
    
    methods = {"DataLoading": data_options, "Trivial": trivial_options, 
               "FastBDT": fastbdt_options, "FANN": fann_options,
               "TMVA-BDT": tmva_bdt_options, "TMVA-NN": tmva_nn_options,
               "SKLearn-BDT": sklearn_bdt_options,# "XGBoost": xgboost_options,
               #"HEP_ML_UBoost": hep_ml_uboost_options
              }
    
    weightfiles=''
    for key in methods:
        general_options.m_identifier = key
        basf2_mva.teacher(general_options, methods[key])
        # Download the weightfile from the database and store it on disk in a xml file
        basf2_mva.download(key, key+'.xml')
        weightfiles+=key+'.xml '

#        method = basf2_mva_util.Method(general_options.m_identifier)
#        m = method.train_teacher(training_data, general_options.m_treename, specific_options=options)
#        p0, t0 = m.apply_expert(training_data, general_options.m_treename)
    

    conditions.prepend_testing_payloads('localdb/database.txt')

    #FastBDT evaluation
    import subprocess
    command = f'basf2_mva_evaluate.py -id {weightfiles}\
    -train output_train.root -data output_test.root --treename B0\
    -o comparison_{nTrees}_{depth}_{learning_rate}_{nCutLevels}_{sub_sample_fraction}.zip'
    subprocess.call(command, shell=True)
    
    
