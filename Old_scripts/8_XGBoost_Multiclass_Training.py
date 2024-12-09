# +
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost multiclass BDT training script.

Usage: python3 8_XGBoost_Multiclass_Training.py (-o multi:softmax -c -m BRF)

"""

import argparse
import pandas
import root_pandas
from lightgbm import plot_metric
import xgboost as xgb
import gc
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
                        default='multi:softprob',
                        required=False,
                        choices=['multi:softmax', 'multi:softprob'],
                        help="Training objective: multi:softprob or multi:softmax")
    parser.add_argument('-c', "--continue_train",
                        action="store_true",
                        help="Continue to train an existing booster")
    parser.add_argument('-m', "--model",
                        action="store",
                        type=str,
                        default='BDT',
                        required=False,
                        choices=['BDT', 'RF','BRF'],
                        help="Training model: Boosted Decision Tree, Random Forest or \
                        Boosted Random Forest")
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

    dtrain = xgb.DMatrix(data=train[training_variables], label=train['Signal'],
                            weight=train['__weight__'])
    dtest = xgb.DMatrix(data=test[training_variables], label=test['Signal'],
                            weight=test['__weight__'])

    del signal,bkg_continuum,bkg_DTCFake,bkg_BFake,df,train
    gc.collect()

    params = {'booster': 'gbtree',
              'tree_method': 'hist',
              'objective': args.objective,
              'num_class':4,
              'eval_metric': ['mlogloss','auc'], # 'auc' only works with 'multi:softprob' objective
              'num_parallel_tree': 1, # tune this if to train a boosted random forest
              'grow_policy': 'depthwise',
              'lambda': 1.3241, 
              'alpha': 0.0025,
              'learning_rate': 0.1, # eta
              'max_depth': 6,
              'max_leaves': 0,
              'min_split_loss': 3, # gamma
              'min_child_weight': 350,
              'max_delta_step':10,
              'subsample': 0.8,
              'colsample_bytree': 0.8,
              'reg_lambda': 1,
              'reg_alpha': 0,
              'max_bin': 256,
              "seed":0,
              'verbosity':1}
    num_round = 50
    if args.model!='BDT':
        params['num_parallel_tree'] = 60
        params['learning_rate'] = 1.51573422403091
        params['colsample_bynode'] = 0.8
        num_round = 1 if args.model=='RF' else 8
        
    print(colored(f'Training...', 'magenta'))
    
    eval_result1 = {}
    bst = xgb.train(params, 
                    dtrain,
                    num_boost_round=num_round,
                    xgb_model=f'BDTs/XGBoost/{args.model}_{args.objective}.json' if args.continue_train else None,
                    evals=[(dtrain, 'train'), (dtest, 'test')],
                    evals_result=eval_result1,
                    early_stopping_rounds=5)

    print(colored(f'Finished first {num_round} rounds...', 'blue'))

    # continue training
    # change to decay learning rates during training
#     eval_result2 = {}
#     bst = xgb.train(params, 
#                     dtrain,
#                     num_boost_round=num_round,
#                     xgb_model=bst,
#                     evals=[(dtrain, 'train'), (dtest, 'test')],
#                     evals_result=eval_result2,
#                     early_stopping_rounds=5,
#                     callbacks=[xgb.callback.LearningRateScheduler(lambda iter: params['learning_rate'] * (0.99 ** iter))])

#     print(colored(f'Finished another {num_round} rounds with decay learning rates...', 'blue'))
    

    # save model and metric plots
    if args.continue_train:
        save_path = f'BDTs/XGBoost/{args.model}_{args.objective}_continue'
    else:
        save_path = f'BDTs/XGBoost/{args.model}_{args.objective}'
    print(colored(f'Saving model to {save_path}.json and metric plots', 'blue'))
    bst.save_model(save_path+'.json')
    
#     for valid_set in ['train', 'test']:
#         for metric in params['eval_metric']:
#             eval_result1[valid_set][metric] += eval_result2[valid_set][metric]

    plot_metric(eval_result1, params['eval_metric'][0], figsize=(12,9))
    plt.savefig(save_path+f"_{params['eval_metric'][0]}.png")
    plot_metric(eval_result1, params['eval_metric'][1], figsize=(12,9))
    plt.savefig(save_path+f"_{params['eval_metric'][1]}.png")

    # getting the prediction on the test sample
    pred = bst.predict(dtest)

    # eval with loaded model
    if args.objective=='multi:softmax':
        print(colored(f"{args.objective} is chosen to be the activation function, no auc calculation", 'magenta'))
    else:
        auc_loaded_model = roc_auc_score(test['Signal'], pred, 
                                         multi_class='ovo' if args.objective=='multi:softprob' else 'ovr')
        print(colored(f"The ROC AUC of trained model's prediction is: {auc_loaded_model}", 'magenta'))
