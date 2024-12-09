# +
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM multiclass BDT training script.

Usage: python3 7_LightGBM_Multiclass_TunerCV.py (-o multiclassova -c)

"""

import argparse
import pandas as pd
import uproot
import gc
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
import utilities as util

import lightgbm as lgb
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
    
    # Disable logging messages from LightGBM and Optuna
    #optuna.logging.set_verbosity(optuna.logging.WARNING)

    # define training variables
    training_variables = util.training_variables

    # load data
    print(colored(f'Loading data and initializing configrations', 'blue'))
    train_sub = uproot.concatenate([f'AutogluonModels/train.root:B0'],library="np")
    df_train_sub = pd.DataFrame({k:v for k, v in train_sub.items() if k!='index'})
    
    # train test split
    print(colored(f'Splitting training test samples', 'green'))
    train_data = df_train_sub.sample(frac=1, random_state=0)

    # use all data for cv
    print(colored(f'Running CV tuner', 'magenta'))
    lgb_train = lgb.Dataset(data=train_data[training_variables], label=train_data['target'],
                            weight=train_data['__weight__'],free_raw_data=True)

    def objective(trial):
        param = {'boosting_type': 'gbdt',
                 'objective': args.objective,
                 'metric': ['multi_logloss','auc_mu'],
                 'num_class':4,
                 'max_bin': 255,
                 "seed":0,
                 'force_col_wise': True,
                 'feature_fraction': 0.9,
                 'feature_pre_filter':False,
                 'verbosity':-1,
                 'bagging_fraction': 0.8,
                 'bagging_freq': trial.suggest_int("bagging_freq", 2, 20),
                 "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                 "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                 "num_leaves": trial.suggest_int("num_leaves", 2, 80),
                 "learning_rate": trial.suggest_float("learning_rate", 0.1, 2),
                 "min_child_samples": trial.suggest_int("min_child_samples", 100, 1e+8, step=5),
                }

        # Add a callback for pruning.
        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "multi_logloss", valid_name='test')
        history = lgb.cv(param, lgb_train, num_boost_round=40, callbacks=[pruning_callback])

        mean_loss = history["multi_logloss-mean"][-1]
        return mean_loss
    
    
    # running optuna in RDB
    study_name = "lgbm-study"  # Unique identifier of the study.
    storage_name = "sqlite:///BDTs/LightGBM/{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="minimize")
    
    study.optimize(objective, n_trials=50, n_jobs=2, gc_after_trial=False)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
        
#     # DC
#     study = optuna.load_study(
#         study_name="distributed-example", storage="mysql://root@localhost/example",
#         pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize")

#     # running optuna in local memory
#     study = optuna.create_study(
#         pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize")
