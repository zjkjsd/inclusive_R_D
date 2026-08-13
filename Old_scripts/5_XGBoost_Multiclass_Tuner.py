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
from sklearn.model_selection import train_test_split
import utilities as util
import optuna

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
    signal = pandas.read_parquet('BDTs/MC14ri_signal.parquet', engine="pyarrow", columns=variables)
    bkg_continuum = pandas.read_parquet('BDTs/MC14ri_qqbar.parquet', engine="pyarrow", columns=variables)
    bkg_DTCFake = pandas.read_parquet('BDTs/MC14ri_DTCFake.parquet', engine="pyarrow", columns=variables)
    bkg_BFake = pandas.read_parquet('BDTs/MC14ri_BFake.parquet', engine="pyarrow", columns=variables)

    # define multiclass label
    signal['Signal'] = 0
    signal['__weight__'] = 13
    bkg_continuum['Signal'] = 1
    bkg_DTCFake['Signal'] = 2
    bkg_BFake['Signal'] = 3


    df = pandas.concat([signal,
                        bkg_continuum.sample(n=1000000, random_state=0),
                        bkg_DTCFake.sample(n=1000000, random_state=0),
                        bkg_BFake], ignore_index=True)
    
    # use all data for cv
    print(colored(f'Running CV tuner', 'magenta'))
    dtrain = xgb.DMatrix(data=df[training_variables], label=df['Signal'],
                            weight=df['__weight__'])

    del signal,bkg_continuum,bkg_DTCFake,bkg_BFake,df
    gc.collect()

    # define optuna optimization target
    def objective(trial):
        param = {'objective': args.objective,
                 'num_class':4,
                 'eval_metric': ['mlogloss','auc'], # 'auc' only works with 'multi:softprob' objective
                 'tree_method': 'hist',
                 "seed":0,
                 'verbosity':0,
                 "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
                 "lambda": trial.suggest_float("lambda", 1, 100, log=True),
                 "alpha": trial.suggest_float("alpha", 1e-8, 10, log=True),
                 "learning_rate": trial.suggest_float("learning_rate", 0.2, 3), # eta
                 'subsample': trial.suggest_float('subsample', 0.4, 1),
                 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
                 "max_depth": trial.suggest_int("max_depth", 1, 9),
                 "min_split_loss": trial.suggest_int("min_split_loss", 1, 100), # gamma
                 'min_child_weight': trial.suggest_int("min_child_weight", 1, 1e+8),
                 "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
                 'num_parallel_tree': 1, # tune this if to train a boosted random forest
                }
        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
        
        # model dependent parameters
        num_round = 50
        if args.model!='BDT':
            param['num_parallel_tree'] = trial.suggest_int("num_parallel_tree", 1, 100)
            num_round = 1 if args.model=='RF' else 10

        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-mlogloss")
        history = xgb.cv(param, dtrain, num_boost_round=num_round, callbacks=[pruning_callback])

        mean_loss = history["test-mlogloss-mean"].values[-1]
        return mean_loss
    
    
    # running optuna in RDB
    study_name = f"xgb-study-{args.model}"  # Unique identifier of the study.
    storage_name = f"sqlite:///BDTs/XGBoost/{study_name}.db"
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="minimize")
    
    study.optimize(objective, n_trials=50, n_jobs=2, gc_after_trial=False)
    # n_jobs has to be set to 2 if running on kekcc, this will help allocate cpu usage correctly

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
