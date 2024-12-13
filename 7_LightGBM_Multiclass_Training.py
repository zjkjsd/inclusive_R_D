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

    # load data
    print(colored(f'Loading data and initializing configrations', 'blue'))
    train_sub = uproot.concatenate([f'AutogluonModels/train.root:B0'],library="np")
    df_train_sub = pd.DataFrame({k:v for k, v in train_sub.items() if k!='index'})
    
    # train test split
    print(colored(f'Splitting training test samples', 'green'))
    train_data = df_train_sub.sample(frac=0.8, random_state=0)
    validation_data = df_train_sub.drop(train_data.index)

    lgb_train = lgb.Dataset(data=train_data[training_variables], label=train_data['target'],
                            weight=train_data['__weight__'],free_raw_data=False)
    lgb_eval = lgb_train.create_valid(data=validation_data[training_variables], label=validation_data['target'],
                            weight=validation_data['__weight__'])

    params = {'boosting_type': 'gbdt',
              'objective': args.objective,
              'metric': ['multi_logloss','auc_mu'],
              'num_class':4,
              'learning_rate': 0.5,
              'num_leaves': 49,
              'min_child_samples': 30000,
              'bagging_fraction': 0.8,
              'bagging_freq': 5,
              'feature_fraction': 1,
              'feature_pre_filter':False,
              'lambda_l1': 0.7,
              'lambda_l2': 0.007,
              'max_bin': 255,
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
    pred = gbm.predict(validation_data[training_variables], num_iteration=gbm.best_iteration)

    # eval with loaded model
    if args.objective=='multiclassova': # convert raw predictions into class probabilities using softmax function
        pred = np.exp(pred) / np.sum(np.exp(pred), axis=1, keepdims=True)
    auc_loaded_model = roc_auc_score(validation_data['target'], pred, 
                                     multi_class='ovo' if args.objective=='multiclass' else 'ovr')
    print(colored(f"The ROC AUC of trained model's prediction is: {auc_loaded_model}", 'magenta'))
