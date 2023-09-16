# +
# # #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A Script to save templates in a json file.
Usage: python3 6_Save_Templates -d Samples/G -i bbbar_0.parquet -l e -t (-o templates1.json -f 0.8 -c)
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

    parser.add_argument('-d', "--dir",
                        action="store",
                        type=str,
                        default=None,
                        required=True,
                        help="relative path to the input parquet files")
    parser.add_argument('-i',"--input",
                        action="store", # if append, input = [ [-i 1 2], [-i 3 4] ]
                        nargs='+',
                        required=True,
                        help="List of input parquet files in the --dir filder")
    parser.add_argument('-l', "--lmode",
                        action="store",
                        type=str,
                        required=True,
                        choices=['e', 'mu'],
                        help="Lepton mode, e or mu")
    parser.add_argument('-o', "--output",
                        action="store",
                        type=str,
                        default='templates1.json',
                        required=False,
                        help="Name of the json file")
    parser.add_argument('-t', "--template",
                        action="store_true",
                        help="Save as templates rather than testing MC")
    parser.add_argument('-f', "--fraction",
                        action="store",
                        type=float,
                        default=1.0,
                        required=False,
                        help="Fraction of data to be stored in the template json file")
    parser.add_argument('-c', "--extra_cut",
                        action="store",
                        type=str,
                        required=False,
                        default=None,
                        help="Extra cut for the templates")
    return parser


if __name__ == "__main__":
    
    args = argparser().parse_args()

    # Load Ntuples
    training_variables = util.training_variables
    variables = util.variables
#     files = ['MC14ri_sigDDst_foldex_e_7/sigDDst_0.parquet', 
#              'MC14ri_normDDst_foldex_e_7/normDDst_0.parquet',
#              'MC14ri_Dststell2_foldex_e_7/Dststell2_0.parquet', 
#              'MC14ri_DststTau1_foldex_e_7/DststTau1_0.parquet',
#              'MC14ri_DststTau2_foldex_e_7/DststTau2_0.parquet']


    total = []
    for filename in tqdm(args.input, desc=colored('Loading input parquets', 'blue')):
        file_location = f'{args.dir}/{filename}'
        data = pd.read_parquet(file_location, engine="pyarrow",
                               columns=['__experiment__','__run__','__event__','__production__',
                                        'B0_isContinuumEvent','DecayMode', 'p_D_l', 'B_D_ReChi2',
                                        'B0_mcPDG','B0_mcErrors','D_mcErrors','D_mcPDG',
                                        f'{args.lmode}_genMotherPDG', f'{args.lmode}_mcPDG',
                                        f'{args.lmode}_mcErrors',f'{args.lmode}_CMS_mcP',
                                        f'{args.lmode}_pErr']+variables
                              )
        total.append(data)
    df = pd.concat(total,ignore_index=True).reset_index()
    df.eval(f'{args.lmode}_pSig = ({args.lmode}_CMS_mcP - e_CMS_p)/{args.lmode}_pErr', inplace=True)


    import lightgbm as lgb
    bst_lgb = lgb.Booster(model_file=f'./BDTs/LightGBM/lgbm_multiclass.txt')

    print(colored(f'Applying MVA...', 'green'))
    pred = bst_lgb.predict(df[training_variables], num_iteration=50) #bst_lgb.best_iteration
    lgb_out = pd.DataFrame(pred, columns=['signal_prob','continuum_prob','fakeD_prob','fakeB_prob'])

    df_lgb = pd.concat([df, lgb_out], axis=1)
    df_lgb['largest_prob'] = df_lgb[['signal_prob','continuum_prob','fakeD_prob','fakeB_prob']].max(axis=1)
    del pred, df, lgb_out

    print(colored(f'Group the samples with MVA outputs', 'green'))
    df, samples=util.get_dataframe_samples(df_lgb, args.lmode, args.template)

    # Inititalize the template json file
    workspace = args.output
    workspace_file = f'./Offline/{workspace}'
    cut=f'signal_prob==largest_prob and signal_prob>0.8'
    print(colored(f'Saving to a test json file', 'blue'))
    if args.template:
        print(colored(f'Saving to a template json file', 'blue'))
        cut=f'signal_prob==largest_prob and signal_prob>0.8 and {args.lmode}_pSig<100'
    xedges = np.linspace(-2, 10, 36) # 36, 48, 61 bins # -7.5 for weMiss2, -2 for weMiss3, -2.5 for weMiss4
    yedges = np.linspace(0.4, 4.6, 42)
    variable_x = 'B0_CMS3_weMissM2'
    variable_y = 'p_D_l'

    i = 0
    for name, df in samples.items():
        if name in ['bkg_continuum','bkg_fakeDTC','bkg_fakeB','bkg_others']:
            continue
        sample = df.sample(frac=args.fraction, random_state=0)
        df_cut=sample.query(cut)
        df_bestSelected=df_cut.loc[df_cut.groupby(['__experiment__','__run__','__event__','__production__']).B_D_ReChi2.idxmin()]
        if args.extra_cut:
            df_bestSelected=df_bestSelected.query(args.extra_cut)
        (counts, xedges, yedges) = np.histogram2d(df_bestSelected[variable_x], 
                                                  df_bestSelected[variable_y],
                                                  bins=[xedges, yedges])
        counts = counts.T

        print(colored(f'Initially {name=}, size = {len(sample)}', 'blue'))
        print(colored(f'After MVA {name=}, size = {len(sample.query(cut))}', 'magenta')) 
        print(colored(f'After BCS {name=}, size = {np.sum(counts)}', 'magenta'))    

        with open(workspace_file, 'r+') as f:
            data = json.load(f)
            data['channels'][0]['samples'][i]['name'] = name
            data['channels'][0]['samples'][i]['data'] = counts.ravel().tolist()
            # counts.ravel()/.reshape(-1) returns a view, counts.flatten() returns a copy (slower)
            f.seek(0)        # <--- should reset file position to the beginning.
            json.dump(data, f, indent=4)
            f.truncate()     # remove remaining part

        i += 1
