# +
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM multiclass BDT training script.

Usage: python3 7_LightGBM_Multiclass_Training.py (-o multiclassova -c)

"""

import argparse
import pandas as pd
import uproot
import lightgbm as lgb
import gc
import json
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
                        default='multiclass',
                        required=False,
                        choices=['multiclass', 'multiclassova'],
                        help="Training objective: multiclass or multiclassova")
    parser.add_argument('-c', "--continue_train",
                        action="store_true",
                        help="Continue to train an existing booster")
    return parser


if __name__ == "__main__":
    
    args = argparser().parse_args()

    # define training variables
    training_variables = util.training_variables
    relevant_vars = training_variables + ['target', 'Ecms','B0_CMS_roeP_my_mask', 'B0_dr']
    cut = '(B0_roeMbc_my_mask>5) & (-4<B0_roeDeltae_my_mask) & (B0_roeDeltae_my_mask<1) & (B0_dr<0.1)'

    # load data
    print(colored('Loading data and initializing configrations', 'blue'))
    sig_mc15ri = uproot.concatenate(['AutogluonModels/train.root:B0'],cut='target==0',library="np",
                                       filter_branch=lambda branch: branch.name in relevant_vars)
    df_sig = pd.DataFrame(sig_mc15ri)
    df_sig['weight']=6

    fakes_mc16rd = uproot.concatenate(['BDTs/train_fakes.root:B0'],library="np",
                                        filter_branch=lambda branch: branch.name in relevant_vars)
    df_fakes = pd.DataFrame(fakes_mc16rd)
    df_fakes['weight']=1

    data_4Soffres = uproot.concatenate(['Samples/Data/e_channel/proc16_4Soffres_deimos_1.root:B0'],
                          library="np",cut = cut,filter_branch=lambda branch: branch.name in relevant_vars)
    df_continuum = pd.DataFrame(data_4Soffres).drop('B0_roeMbc_my_mask', axis=1) 
    df_continuum.eval('B0_roeMbc_my_mask = ( (10.58/2)**2 - (B0_CMS_roeP_my_mask*10.58/Ecms)**2 )**0.5', inplace=True)
    df_continuum['weight']=2
    df_continuum['target']=3

    df_train = pd.concat([df_sig, df_fakes, df_continuum])
    print(colored(df_train[['target','weight']].value_counts(), 'green'))
    
    # train test split
    print(colored('Splitting training test samples', 'blue'))
    train_set = df_train.sample(frac=0.8, random_state=0)
    validation_set = df_train.drop(train_set.index)

    lgb_train = lgb.Dataset(data=train_set[training_variables], label=train_set['target'],
                            weight=train_set['weight'],free_raw_data=False)
    lgb_eval = lgb_train.create_valid(data=validation_set[training_variables], label=validation_set['target'],
                            weight=validation_set['weight'])

    params = {'boosting_type': 'gbdt',
              'objective': args.objective,
              'metric': ['multi_logloss','auc_mu'],
              'num_class':4,
              'learning_rate': 0.5,
              'num_leaves': 49,
              'min_child_samples': 1000,
              'bagging_fraction': 0.7,
              'bagging_freq': 5,
              'feature_fraction': 0.7,
              'feature_pre_filter':False,
              'lambda_l1': 1,
              'lambda_l2': 1,
              # 'max_bin': 255,
              "seed":0,
              'force_col_wise': True,
              'verbosity':1}
    

    print(colored(f'Training...', 'magenta'))
    eval_result1 = {}
    gbm = lgb.train(params, 
                    lgb_train,
                    num_boost_round=30,
                    init_model=f'BDTs/LightGBM/lgbm_{args.objective}.txt' if args.continue_train else None,
                    valid_sets=[lgb_train, lgb_eval],
                    valid_names=['train', 'test'], 
                    keep_training_booster=True,
                    callbacks=[lgb.early_stopping(3),
                               lgb.record_evaluation(eval_result1)])

    print(colored('Finished first 30 rounds...', 'blue'))

    # continue training with decay learning rates
    # reset_parameter callback accepts:
    # 1. list with length = num_boost_round
    # 2. function(curr_iter)
    eval_result2 = {}
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=20,
                    init_model=gbm, # or the file name path 'BDTs/LightGBM/lgbm_model.txt'
                    valid_sets=[lgb_train, lgb_eval],
                    valid_names=['train', 'test'],
                    callbacks=[lgb.reset_parameter(learning_rate=lambda iter: params['learning_rate'] * (0.99 ** iter)),
#                                lgb.reset_parameter(bagging_fraction=[0.7] * 10 + [0.6] * 10),
                               lgb.early_stopping(2),
                               lgb.record_evaluation(eval_result2)])

    print(colored('Finished the last 20 rounds with decay learning rates...', 'blue'))
    

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
            
    with open('BDTs/LightGBM/eval_result.json', "w") as f:
        json.dump(eval_result1, f)

    ax0 = lgb.plot_metric(eval_result1, params['metric'][0], figsize=(8,6))
    ax0.set_xlabel("Iterations", fontsize=14)
    ax0.set_ylabel(params['metric'][0], fontsize=14)
    plt.savefig(save_path+f"_{params['metric'][0]}.png")
    ax1 = lgb.plot_metric(eval_result1, params['metric'][1], figsize=(8,6))
    ax1.set_xlabel("Iterations", fontsize=14)
    ax1.set_ylabel(params['metric'][1], fontsize=14)
    plt.savefig(save_path+f"_{params['metric'][1]}.png")

    # can only predict with the best iteration (or the saving iteration)
    pred = gbm.predict(validation_set[training_variables], num_iteration=gbm.best_iteration)

    # eval with loaded model
    if args.objective=='multiclassova': # convert raw predictions into class probabilities using softmax function
        pred = np.exp(pred) / np.sum(np.exp(pred), axis=1, keepdims=True)
    auc_loaded_model = roc_auc_score(validation_set['target'], pred, 
                                     multi_class='ovo' if args.objective=='multiclass' else 'ovr')
    print(colored(f"The ROC AUC of trained model's prediction is: {auc_loaded_model}", 'magenta'))
