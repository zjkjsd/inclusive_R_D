# +
# # #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A Script to save templates in a json file.
Usage: python3 6_Save_Templates (-t 2d_2channels_workspace_3_0.json)
"""

import argparse
import utilities as util
from termcolor import colored
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

def argparser():
    """
    Parse options as command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-t', "--template",
                        action="store",
                        type=str,
                        default='2d_2channels_workspace_3_0.json',
                        required=False,
                        help="Location of the template json file")
    return parser


if __name__ == "__main__":
    
    args = argparser().parse_args()

    # Load Ntuples
    training_variables = util.training_variables
    variables = util.variables
    files = ['sigDDst', 'normDDst','bkgDststp_tau', 'bkgDstst0_tau','bkgDstst0_ell']

    total = []
    for file_name in tqdm(files, desc=colored('Loading parquets', 'blue')):
        filename=f'./Samples/Signal_MC14ri/MC14ri_{file_name}_bengal_e_2/{file_name}_bengal_e_2_0.parquet'
        data = pd.read_parquet(filename, engine="pyarrow",
                               columns=['D_mcPDG', 'e_mcPDG','DecayMode', 
                                        'p_D_l', 'B_D_ReChi2','e_genMotherPDG','B0_mcPDG',
                                        'B0_mcErrors','B0_isContinuumEvent']+variables)
        total.append(data)
    df = pd.concat(total,ignore_index=True).reset_index()


    import lightgbm as lgb
    bst_lgb = lgb.Booster(model_file=f'./BDTs/LightGBM/lgbm_multiclass.txt')

    print(colored(f'Applying MVA...', 'green'))
    pred = bst_lgb.predict(df[training_variables], num_iteration=50) #bst_lgb.best_iteration
    lgb_out = pd.DataFrame(pred, columns=['signal_prob','continuum_prob','fakeD_prob','fakeB_prob'])

    df_lgb = pd.concat([df, lgb_out], axis=1)
    df_lgb['largest_prob'] = df_lgb[['signal_prob','continuum_prob','fakeD_prob','fakeB_prob']].max(axis=1)
    del pred, df, lgb_out

    print(colored(f'Group the samples with MVA outputs', 'green'))
    df, samples=util.get_dataframe_samples(df_lgb)

    # Inititalize the template json file
    workspace = args.template
    workspace_file = f'./Offline/{workspace}'
    cut='signal_prob==largest_prob and signal_prob>0.8'
    xedges = np.linspace(-2, 10, 48) # -7.5 for weMiss2, -2 for weMiss3, -2.5 for weMiss4
    yedges = np.linspace(0.4, 4.6, 42)
    variable_x = 'B0_CMS3_weMissM2'
    variable_y = 'p_D_l'

    i = 0
    for name, sample in samples.items():
        if name in ['bkg_continuum','bkg_fakeDTC','bkg_fakeB','bkg_others']:
            continue
        (counts, xedges, yedges) = np.histogram2d(sample.query(cut)[variable_x], 
                                                  sample.query(cut)[variable_y],
                                                  bins=[xedges, yedges])
        counts = counts.T

        print(colored(f'BeforeMVA {name=}, size = {len(sample)}', 'blue'))
        print(colored(f'After MVA {name=}, size = {np.sum(counts)}', 'magenta'))    

        with open(workspace_file, 'r+') as f:
            data = json.load(f)
            data['channels'][0]['samples'][i]['name'] = name
            data['channels'][0]['samples'][i]['data'] = counts.ravel().tolist()
            # counts.ravel()/.reshape(-1) returns a view, counts.flatten() returns a copy (slower)
            f.seek(0)        # <--- should reset file position to the beginning.
            json.dump(data, f, indent=4)
            f.truncate()     # remove remaining part

        i += 1
