# +
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM multiclass BDT training script.

Usage: python3 7_LightGBM_TunerCV.py (-o multiclassova -c)

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
                        choices=['multiclass', 'multiclassova','binary'],
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
    training_variables = util.CS_variables

    # load data
    print(colored('Loading data and initializing configrations', 'blue'))
    MC_4Soffres = uproot.concatenate(['Samples/MC16rd/e_channel/4Soffres_deimos_1/*.root:B0'],
                          library="np",
                          filter_branch=lambda branch: branch.name in training_variables)

    data_4Soffres = uproot.concatenate(['Samples/Data/e_channel/proc16_4Soffres_deimos_1.root:B0'],
                              library="np",
                              filter_branch=lambda branch: branch.name in training_variables)
    df_mc_4Soffres = pd.DataFrame(MC_4Soffres)
    df_data_4Soffres = pd.DataFrame(data_4Soffres)


    # define binary label
    df_data_4Soffres['data'] = 1
    df_data_4Soffres['weight'] = 4 #4 streams MC16rd
    df_mc_4Soffres['data'] = 0
    df_mc_4Soffres['weight'] = 1
    df_all = pd.concat([df_data_4Soffres,df_mc_4Soffres],ignore_index=True)
    
    # use all data for cv
    print(colored('Running CV tuner (LightGBMTunerCV)', 'magenta'))
    lgb_train = lgb.Dataset(data=df_all[training_variables], label=df_all['data'],
                            weight=df_all['weight'],free_raw_data=True)    

    # create a optuna study
    study_name = f"lgbm_{args.objective}_study"  # Unique identifier of the study.
    storage_name = f"sqlite:///BDTs/LightGBM/lgbm_{args.objective}_study.db"
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize")

    # setup base parameters
    metric = 'auc' if args.objective=='binary' else 'auc_mu'
    base_params = {
        'boosting_type': 'gbdt',
        "objective": args.objective,   # e.g. "binary" or "multiclassova"/"multiclass"
        "metric": metric,
        "verbosity": -1,
        "seed": 0,
        "force_col_wise": True,
        'feature_pre_filter':False,
    }

    # create a lgbm tuner
    from optuna.integration import LightGBMTunerCV

    tuner = LightGBMTunerCV(
        params=base_params,
        train_set=lgb_train,
        study=study,                   # <-- attaches to your SQLite-backed Optuna study
        nfold=5,
        stratified=(args.objective == "binary"),
        shuffle=True,
        seed=0,
        num_boost_round=50,
        # pruning=True  # (default True) uses the study's pruner
    )

    # Run the tuner ---------------------------------------------------------
    tuner.run()
    
    # Results ---------------------------------------------------------------
    print("Best params from tuner:")
    print(tuner.best_params)          # dict of tuned params
    print("Best CV score:", tuner.best_score)  # mean of the chosen metric
    print("Best iteration:", tuner.best_iteration)



    # def objective(trial):
    #     param = {'boosting_type': 'gbdt',
    #              'objective': args.objective,
    #              'metric': metric, # ['multi_logloss','auc_mu']
    #              # 'num_class':4,
    #              # 'max_bin': 255,
    #              "seed":0,
    #              'force_col_wise': True,
    #              'feature_pre_filter':False,
    #              'verbosity':-1,
                 
    #              # capacity / structure
    #              'num_leaves': trial.suggest_int('num_leaves', 2, 50),
    #              'max_depth': trial.suggest_int('max_depth', 3, 5),
         
    #              # learning
    #              'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.5, log=True),
         
    #              # regularization / min data in leaf
    #              'min_child_samples': trial.suggest_int('min_child_samples', 100, 10000),
         
    #              # sampling
    #              'bagging_fraction': trial.suggest_float('subsample', 0.5, 1.0),
    #              'bagging_freq': trial.suggest_int('subsample_freq', 1, 10),
    #              'feature_fraction': trial.suggest_float('colsample_bytree', 0.5, 1.0),
         
    #              # L1/L2
    #              'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
    #              'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
    #             }

    #     from optuna.integration import LightGBMTunerCV
    #     tuner = LightGBMTunerCV(
    #                 params=base_params,
    #                 train_set=lgb_train,
    #                 study=study,                   # <-- attaches to your SQLite-backed Optuna study
    #                 nfold=5,
    #                 stratified=(args.objective == "binary"),
    #                 shuffle=True,
    #                 seed=0,
    #                 metrics=metric,                # "auc" or "auc_mu"
    #                 num_boost_round=2000,
    #                 early_stopping_rounds=100,
    #                 verbose_eval=False,            # set True if you want logs each iteration
    #                 # pruning=True  # (default True) uses the study's pruner
    #             )

    #     pruning_callback = optuna.integration.LightGBMPruningCallback(trial, metric, valid_name='cv_agg')
    #     history = lgb.cv(param, lgb_train, nfold=5, num_boost_round=50, 
    #                      metrics=metric, callbacks=[pruning_callback])

    #     mean_auc = history[f"{metric}-mean"][-1]
    #     return mean_auc
    
    
    # # running optuna in RDB
    # study_name = f"lgbm_{args.objective}_study"  # Unique identifier of the study.
    # storage_name = f"sqlite:///BDTs/LightGBM/lgbm_{args.objective}_study.db"
    # study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True,
    #                             pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize")
    
    # study.optimize(objective, n_trials=50, n_jobs=2, gc_after_trial=False)

    # print("Number of finished trials: {}".format(len(study.trials)))

    # print("Best trial:")
    # trial = study.best_trial

    # print("  Value: {}".format(trial.value))

    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))
        
        
#     # DC
#     study = optuna.load_study(
#         study_name="distributed-example", storage="mysql://root@localhost/example",
#         pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize")

#     # running optuna in local memory
#     study = optuna.create_study(
#         pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize")
