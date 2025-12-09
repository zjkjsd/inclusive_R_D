# +
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM binary BDT training script.

Usage: python3 1_LightGBM_Binary_Training.py (-o multiclassova -c)

"""

import argparse
import uproot
import gc
import pandas as pd
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import utilities as util

def argparser():
    """
    Parse options as command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-o', "--objective",
                        action="store",
                        type=str,
                        default='binary',
                        required=False,
                        choices=['binary', 'cross-entropy'],
                        help="Training objective: binary or cross-entropy")
    parser.add_argument('-c', "--continue_train",
                        action="store_true",
                        help="Continue to train an existing booster")
    return parser


if __name__ == "__main__":
    
    args = argparser().parse_args()

    # define training variables
    training_variables = util.training_variables
    vars_to_load = training_variables + ['B0_CMS_roeP_my_mask', 'B0_CMS_roeE_my_mask', 'Ecms']

    # load data
    print(colored('Loading data and initializing configrations', 'blue'))
    MC_4Soffres = uproot.concatenate(['Samples/MC16rd/e_channel/4Soffres_deimos_3/*.root:B0'],
                          library="np",
                          filter_branch=lambda branch: branch.name in vars_to_load)

    data_4Soffres = uproot.concatenate(['Samples/Data/e_channel/proc16_4Soffres_deimos_3.root:B0'],
                              library="np",
                              filter_branch=lambda branch: branch.name in vars_to_load)
    df_mc_4Soffres = pd.DataFrame(MC_4Soffres)
    df_data_4Soffres = pd.DataFrame(data_4Soffres)

    # rescale the ROE p and E according to Ecms
    for df in [df_mc_4Soffres,df_data_4Soffres,]:
        df.eval('B0_roeMbc_cor = ( (10.58/2)**2 - (B0_CMS_roeP_my_mask*10.58/Ecms)**2 )**0.5', inplace=True)
        df.eval('B0_roeDeltae_cor = B0_CMS_roeE_my_mask*10.58/Ecms - 10.58/2', inplace=True)
        df['B0_roeDeltae_my_mask'] = df['B0_roeDeltae_cor']
        df['B0_roeMbc_my_mask'] = df['B0_roeMbc_cor']

    
    # define binary label
    df_data_4Soffres['data'] = 1
    df_data_4Soffres['weight'] = 4 #4 streams MC16rd
    df_mc_4Soffres['data'] = 0
    df_mc_4Soffres['weight'] = 1
    df_all = pd.concat([df_data_4Soffres,df_mc_4Soffres],ignore_index=True)
    
    # train test split
    print(colored('Splitting training test samples', 'green'))
    train, test = train_test_split(df_all, test_size=0.2, random_state=0, shuffle=True, stratify=df_all['data'])

    lgb_train = lgb.Dataset(data=train[training_variables], label=train['data'],
                            weight=train['weight'],free_raw_data=False)
    lgb_eval = lgb_train.create_valid(data=test[training_variables], label=test['data'],
                            weight=test['weight'])

    del df_all, df_data_4Soffres, df_mc_4Soffres
    gc.collect()


    params = {'boosting_type': 'gbdt',
              'objective': args.objective,
              'metric': ['binary_logloss','auc'],
              'learning_rate': 0.5,
              'num_leaves': 10,
              'max_depth': -1,
              'min_child_samples': 1000,
              'bagging_fraction': 0.7,
              'bagging_freq': 5,
              'feature_fraction': 0.7,
              'feature_pre_filter':False,
              'lambda_l1': 0.1,
              'lambda_l2': 1,
              # 'max_bin': 255,
              "seed":0,
              'force_col_wise': True,
              'verbosity':1}
    

    print(colored('Training...', 'magenta'))
    eval_result1 = {}
    gbm = lgb.train(params, 
                    lgb_train,
                    num_boost_round=40,
                    init_model=f'BDTs/LightGBM/lgbm_{args.objective}.txt' if args.continue_train else None,
                    valid_sets=[lgb_train, lgb_eval],
                    valid_names=['train', 'test'], 
                    keep_training_booster=True,
                    callbacks=[lgb.early_stopping(5),
                               lgb.record_evaluation(eval_result1)])

    print(colored('Finished first 40 rounds...', 'blue'))

    # continue training with decay learning rates
    # reset_parameter callback accepts:
    # 1. list with length = num_boost_round
    # 2. function(curr_iter)
    eval_result2 = {}
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=40,
                    init_model=gbm, # or the file name path 'BDTs/LightGBM/lgbm_model.txt'
                    valid_sets=[lgb_train, lgb_eval],
                    valid_names=['train', 'test'],
                    callbacks=[lgb.reset_parameter(learning_rate=lambda iter: params['learning_rate'] * (0.99 ** iter)),
                               lgb.reset_parameter(bagging_fraction=[0.7] * 20 + [0.6] * 20),
                               lgb.early_stopping(5),
                               lgb.record_evaluation(eval_result2)])

    print(colored('Finished the last 40 rounds with decay learning rates...', 'blue'))
    

    # save model and metric plots
    if args.continue_train:
        save_path = f'BDTs/LightGBM/lgbm_{args.objective}_continue'
    else:
        save_path = f'BDTs/LightGBM/lgbm_{args.objective}'
    print(colored(f'Saving model to {save_path}.txt and metric plots', 'blue'))
    gbm.save_model(save_path+'.txt', num_iteration=gbm.best_iteration)
    
    for valid_set in ['train', 'test']:
        for metric in params['metric']:
            eval_result1[valid_set][metric] += eval_result2[valid_set][metric]

    lgb.plot_metric(eval_result1, params['metric'][0], figsize=(12,9))
    plt.savefig(save_path+f"_{params['metric'][0]}.png")
    lgb.plot_metric(eval_result1, params['metric'][1], figsize=(12,9))
    plt.savefig(save_path+f"_{params['metric'][1]}.png")

    # can only predict with the best iteration (or the saving iteration)
    pred = gbm.predict(test[training_variables], num_iteration=gbm.best_iteration)

    # eval with loaded model
    auc_loaded_model = roc_auc_score(test['data'], pred)
    print(colored(f"The ROC AUC of trained model's prediction is: {auc_loaded_model}", 'magenta'))
