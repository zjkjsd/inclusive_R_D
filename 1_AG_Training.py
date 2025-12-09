# -*- coding: utf-8 -*-
# +
"""
Autogluon training script.

Usage: python3 1_AG_Training.py

"""
from autogluon.tabular import TabularPredictor
import uproot
import pandas as pd
import argparse
import utilities as util
from termcolor import colored

def argparser():
    """
    Parse options as command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-p', "--presets",
                        action="store",
                        type=str,
                        default='medium_quality',
                        required=False,
                        choices=['best_quality', 'high_quality','good_quality','medium_quality'],
                        help="Training presets")
    parser.add_argument('-t', "--time_limit",
                        action="store",
                        type=int,
                        default=3600,
                        required=False,
                        help="Time limit for training")
    return parser


if __name__ == "__main__":
    
    args = argparser().parse_args()

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
    df_mc_4Soffres['data'] = 0
    df_all = pd.concat([df_data_4Soffres,df_mc_4Soffres],ignore_index=True)
    
    # # train test split
    # print(colored('Splitting training test samples', 'green'))
    # train, test = train_test_split(df_all, test_size=0.2, random_state=0, shuffle=True, stratify=df_all['data'])

    # Define and fit the AutoGluon classifier
    ag = TabularPredictor(label='data', eval_metric='f1_macro',sample_weight='balance_weight')
    predictor = ag.fit(df_all, presets=args.presets, time_limit=args.time_limit,save_bag_folds=True,
                       infer_limit=0.05, infer_limit_batch_size=10000,)
#                        hyperparameters={"GBM": ['GBMLarge']},
                       excluded_model_types=['CAT',]) #'FASTAI','RF','XT','KNN','XGB'
