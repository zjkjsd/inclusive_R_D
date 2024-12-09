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

def argparser():
    """
    Parse options as command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-p', "--presets",
                        action="store",
                        type=str,
                        default='good_quality',
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

    train_sub = uproot.concatenate([f'AutogluonModels/train.root:B0'],library="np")
    df_train_sub = pd.DataFrame({k:v for k, v in train_sub.items() if k not in ['index','__weight__']})

    # Split the training set to train and validation
#     train_data = df_train_sub.sample(frac=0.8, random_state=0)
#     validation_data = df_train_sub.drop(train_data.index)

    # Define and fit the AutoGluon classifier
    ag = TabularPredictor(label='target', eval_metric='f1_macro',sample_weight='balance_weight')
    predictor = ag.fit(df_train_sub, presets=args.presets, time_limit=args.time_limit,save_bag_folds=True,
#                        hyperparameters={"GBM": ['GBMLarge']},
                       excluded_model_types=['FASTAI','RF','XT','KNN','CAT','XGB','NN_TORCH'])
