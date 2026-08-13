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
import lightgbm as lgb
import pandas as pd
import uproot
import numpy as np
from tqdm.auto import tqdm
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
#              'MC14ri_normDDst_foldex_e_8/normDDst_0.parquet',
#              'MC14ri_Dststell2_foldex_e_7/Dststell2_0.parquet', 
#              'MC14ri_DststTau1_foldex_e_7/DststTau1_0.parquet',
#              'MC14ri_DststTau2_foldex_e_7/DststTau2_0.parquet']

    columns=['__experiment__','__run__','__event__','__production__','B0_isContinuumEvent',
             'DecayMode', 'p_D_l', 'B_D_ReChi2','B0_mcPDG','B0_mcErrors','D_mcErrors','D_mcPDG',
             'D_genMotherPDG',f'{args.lmode}_genMotherPDG', f'{args.lmode}_mcPDG',
             f'{args.lmode}_mcErrors',f'{args.lmode}_pSig']+variables
    
    for filename in tqdm(args.input, desc=colored('Loading input parquets', 'blue')):
        file_location = f'{args.dir}/{filename}'
        data = pd.read_parquet(file_location, engine="pyarrow",columns=columns)

        print(colored(f'Applying MVA...', 'green'))
        bst_lgb = lgb.Booster(model_file=f'./BDTs/LightGBM/lgbm_multiclass.txt')
        pred = bst_lgb.predict(data[training_variables], num_iteration=50) #bst_lgb.best_iteration
        lgb_out = pd.DataFrame(pred, columns=['signal_prob','continuum_prob','fakeD_prob','fakeB_prob'])

        df_lgb = pd.concat([data, lgb_out], axis=1)
        df_lgb['largest_prob'] = df_lgb[['signal_prob','continuum_prob','fakeD_prob','fakeB_prob']].max(axis=1)
        del pred, data, lgb_out

        print(colored(f'Group the samples with MVA outputs', 'green'))
        data, samples=util.get_dataframe_samples(df_lgb, args.lmode, args.template)


        cut=f'signal_prob==largest_prob and signal_prob>0.8 and {args.lmode}_pSig<100'
    
#     xedges = np.linspace(-2, 10, 36) # 36, 48, 61 bins # -7.5 for weMiss2, -2 for weMiss3, -2.5 for weMiss4
#     yedges = np.linspace(0.4, 4.6, 42)
#     variable_x = 'B0_CMS3_weMissM2'
#     variable_y = 'p_D_l'

        for name, df in samples.items():
            if name in ['bkg_continuum','bkg_fakeDTC','bkg_fakeB','bkg_others']:
                continue
                
            if filename=='sigDDst_0.parquet':
                if name not in [r'$D\tau\nu$', r'$D^\ast\tau\nu$']:
                    continue
            
            if filename=='normDDst_0.parquet':
                if name not in [r'$D\ell\nu$', r'$D^\ast\ell\nu$', r'$D^{\ast\ast}\ell\nu$']:
                    continue
                    
            if filename=='Dststell2_0.parquet' and name!=r'$D^{\ast\ast}\ell\nu$':
                continue
            
            if filename in ['DststTau1_0.parquet', 'DststTau2_0.parquet'] and name!=r'$D^{\ast\ast}\tau\nu$':
                continue
                
            df_cut = df.query(cut)
            df_bestSelected=df_cut.loc[df_cut.groupby(['__experiment__','__run__','__event__','__production__']).B_D_ReChi2.idxmin()]
            df_fitting = df_bestSelected[['__weight__','B0_CMS3_weMissM2','p_D_l']]
            if args.extra_cut:
                df_bestSelected=df_bestSelected.query(args.extra_cut)
            template = df_fitting.sample(frac=args.fraction, random_state=0)
            test = pd.concat([df_fitting, template]).drop_duplicates(keep=False)

            print(colored(f'Template {name=}, size = {len(template)}', 'blue'))
            print(colored(f'Test {name=}, size = {len(test)}', 'magenta'))     

            with uproot.recreate(f'{args.dir}/{name}.root') as file:
                file['template'] = template
                file['test'] = test
