# +
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM multiclass BDT training script.

Usage: python3 7_LightGBM_Multiclass_Training.py (-o multiclassova -c)

"""

import argparse
import pandas
import root_pandas
import lightgbm as lgb
import gc
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
    variables = util.variables

    # load data
    print(colored(f'Loading data and initializing configrations', 'blue'))
    signal = root_pandas.read_root('BDTs/MC14ri_signal.root', columns=variables)
    bkg_continuum = root_pandas.read_root(['BDTs/MC14ri_qqbar.root',
                                           'BDTs/MC14ri_taupair.root'], columns=variables)
    bkg_DTCFake = root_pandas.read_root('BDTs/MC14ri_DTCFake.root', columns=variables)
    bkg_BFake = root_pandas.read_root('BDTs/MC14ri_BFake.root', columns=variables)

    # define multiclass label
    signal['Signal'] = 0
    bkg_continuum['Signal'] = 1
    bkg_DTCFake['Signal'] = 2
    bkg_BFake['Signal'] = 3
    signal['__weight__'] = 40
    bkg_continuum['__weight__'] = 1
    bkg_BFake['__weight__'] = 3

    df = pandas.concat([signal,
                        bkg_continuum,
                        bkg_DTCFake.sample(n=3000000, random_state=0),
                        bkg_BFake], ignore_index=True)
    
    # train test split
    print(colored(f'Splitting training test samples', 'green'))
    train, test = train_test_split(df, test_size=0.2, random_state=0, shuffle=True, stratify=df['Signal'])

    lgb_train = lgb.Dataset(data=train[training_variables], label=train['Signal'],
                            weight=train['__weight__'],free_raw_data=False)
    lgb_eval = lgb_train.create_valid(data=test[training_variables], label=test['Signal'],
                            weight=test['__weight__'])

    del train,signal,bkg_continuum,bkg_DTCFake,bkg_BFake, df
    gc.collect()


    params = {'boosting_type': 'gbdt',
              'objective': args.objective,
              'metric': ['multi_logloss','auc_mu'],
              'num_class':4,
              'learning_rate': 0.4,
              'num_leaves': 70,
              'min_child_samples': 9000,
              'bagging_fraction': 0.8,
              'bagging_freq': 20,
              'feature_fraction': 1,
              'feature_pre_filter':False,
              'lambda_l1': 8,
              'lambda_l2': 4.626876468793015e-07,
              'max_bin': 255,
              "seed":0,
              'force_col_wise': True,
              'verbosity':1}
    

    print(colored(f'Training...', 'magenta'))
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
    if args.objective=='multiclassova': # convert raw predictions into class probabilities using softmax function
        pred = np.exp(pred) / np.sum(np.exp(pred), axis=1, keepdims=True)
    auc_loaded_model = roc_auc_score(test['Signal'], pred, 
                                     multi_class='ovo' if args.objective=='multiclass' else 'ovr')
    print(colored(f"The ROC AUC of trained model's prediction is: {auc_loaded_model}", 'magenta'))
