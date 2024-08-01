# +
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Usage: python3 1_binning_tuner.py (-b 50 25 -m)

"""

import argparse
import uproot
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
import utilities as util
import pandas as pd
import pyhf
import cabinetry
import json


def argparser():
    """
    Parse options as command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-b', "--binning",
                        action="store",
                        type=int,
                        nargs=2,
                        default=[50,50],
                        required=True,
                        help="Binning for the fit")
    parser.add_argument('-m', "--merge",
                        action="store_true",
                        help="Use the merged binning")
    return parser


if __name__ == "__main__":
    
    args = argparser().parse_args()
    print('Binning set as',args.binning)

    # Define the fitting range and number of bins, 'p_D_l'
    start = 0.4
    end = 5
    num_bins = args.binning[1]

    # Create the bin edges
    p_D_l_bins = np.linspace(start, end, num_bins + 1)

    # Define the fitting range and number of bins, 'B0_CMS3_weMissM2'
    start = -5
    end = 10
    num_bins = args.binning[0]

    # Create the bin edges
    MM2_bins = np.linspace(start, end, num_bins + 1)
    
    # Define relevant variables
    training_variables = util.training_variables
    columns = util.all_relevant_variables

    # Load template samples
    e_temp = uproot.concatenate([f'Samples/Generic_MC15ri/e_channel/MC15ri_local_200fb/*.root:B0'],
                              library="np",
                              #cut=input_cut,
                              filter_branch=lambda branch: branch.name in columns)

    df_e = pd.DataFrame(e_temp)

    # apply MVA and BCS
    import lightgbm as lgb
    bst_lgb = lgb.Booster(model_file=f'BDTs/LightGBM/lgbm_multiclass.txt')
    cut='signal_prob==largest_prob and signal_prob>0.8 and \
    continuum_prob<0.04 and fakeD_prob<0.05'

    df_e.eval('B_D_ReChi2 = B0_vtxReChi2 + D_vtxReChi2', inplace=True)
    df_e.eval(f'p_D_l = D_CMS_p + ell_CMS_p', inplace=True)

    pred_e = bst_lgb.predict(df_e[training_variables], num_iteration=50) #bst_lgb.best_iteration
    lgb_out_e = pd.DataFrame(pred_e, columns=['signal_prob','continuum_prob','fakeD_prob','fakeB_prob'])

    df_lgb_e = pd.concat([df_e, lgb_out_e], axis=1)
    df_lgb_e['largest_prob'] = df_lgb_e[['signal_prob','continuum_prob','fakeD_prob','fakeB_prob']].max(axis=1)
    del pred_e, lgb_out_e

    df_cut_e=df_lgb_e.query(cut)
    df_bestSelected_e=df_cut_e.loc[df_cut_e.groupby(['__experiment__','__run__','__event__','__production__']).B_D_ReChi2.idxmin()]

    # Create templates
    te=util.get_dataframe_samples_new(df_bestSelected_e, 'e', template=False)
    ## Asimov test
    indices_threshold_3,temp_asimov_e,temp_asimov_merged_e = util.create_templates(
        samples=te, bins=[MM2_bins, p_D_l_bins], 
        variables=['B0_CMS3_weMissM2','p_D_l'],
        bin_threshold=1,merge_threshold=10)
    (template_flat_e,staterr_flat_e,asimov_data_e) = temp_asimov_e
    (template_flat_e_merged,staterr_flat_e_merged,asimov_data_e_merged) = temp_asimov_merged_e

    # load and update template workspace
    workspace_path = "Notebooks/R_D_2d_workspace_SR_1ch.json"
    spec = cabinetry.workspace.load(workspace_path)
    if args.merge:
        spec = util.update_workspace(workspace=spec,temp_asimov_sets=[temp_asimov_merged_e])
    spec = util.update_workspace(workspace=spec,temp_asimov_sets=[temp_asimov_e])
    model, data = cabinetry.model_utils.model_and_data(spec)

    ## Fit
    pyhf.set_backend("jax")
    fit_results = cabinetry.fit.fit(model=model, data=data,
                                    fix_pars=[False,False,False,False,False,False])
    #, minos=model.config.parameters[:7])

    with open(f'fit_{args.binning[0]}_{args.binning[1]}_merge{args.merge}.txt', 'w') as file:
        for label, result, unc in zip(fit_results.labels, fit_results.bestfit, fit_results.uncertainty):
            file.write(f"{label}: {result:.3f} +/- {unc:.3f}" + '\n')
