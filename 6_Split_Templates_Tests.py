# +
# # #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage: python3 6_Split_Templates_Tests -l e -n 9 (-b Dsideband)
"""

import argparse
import utilities as util
from termcolor import colored
import uproot
import glob
import numpy as np
import pandas as pd
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
                        required=False,
                        help="relative path to the input parquet files")
#     parser.add_argument('-i',"--input",
#                         action="store", # if append, input = [ [-i 1 2], [-i 3 4] ]
#                         nargs='+',
#                         required=False,
#                         help="List of input parquet files in the --dir filder")
    parser.add_argument('-l', "--lmode",
                        action="store",
                        type=str,
                        required=True,
                        choices=['e', 'mu'],
                        help="Lepton mode, e or mu")
#     parser.add_argument('-o', "--output",
#                         action="store",
#                         type=str,
#                         default='templates1.json',
#                         required=False,
#                         help="Name of the json file")
    parser.add_argument('-n', "--nSets",
                        action="store",
                        type=int,
                        default=3,
                        required=False,
                        help="Number of subsets to be splited into and saved")
    parser.add_argument('-b', "--bkgType",
                        action="store",
                        type=str,
                        required=False,
                        default=None,
                        help="Background type")
    return parser


if __name__ == "__main__":
    
    args = argparser().parse_args()
    
    # Define the number of subsets
    num_subsets = args.nSets
    
    # Define the location of output subsets
    output_directory = 'Samples/Generic_MC15ri/e_channel/BBbar_signal_region/'

    # Variables to load
    training_variables = util.training_variables
    columns = util.all_relevant_variables


    files = ['BBbar_signal_region/subset_*.root:B0',
    ]
    
    sb_files = [
#         'MC15ri_*_Dsb_3ab_korat_e_2/Dsideband_*.root:B0',  
        'subset_*.root:B0'
    ]

    if args.bkgType=='Dsideband':
        output_directory = 'Samples/Generic_MC15ri/e_channel/Dsideband/'
        files = sb_files
        
    temp = uproot.concatenate([f'Samples/Generic_MC15ri/e_channel/{f}' for f in files],
                            library="np",
                            filter_branch=lambda branch: branch.name in columns)
    df = pd.DataFrame(temp)

    #df.eval(f'cos_D_l = (D_px*ell_px + D_py*ell_py + D_pz*ell_pz)/(D_p*ell_p)', inplace=True)
    df.eval('B_D_ReChi2 = B0_vtxReChi2 + D_vtxReChi2', inplace=True)
    df.eval(f'p_D_l = D_CMS_p + ell_CMS_p', inplace=True)
    df['D_daughter_pValue_min'] = df[['D_K_pValue','D_pi1_pValue','D_pi2_pValue']].min(axis=1)
    df['D_daughter_pValue_mean'] = df[['D_K_pValue','D_pi1_pValue','D_pi2_pValue']].mean(axis=1)

    # load MVA
    import lightgbm as lgb
    # load model to predict
    bst_lgb = lgb.Booster(model_file=f'BDTs/LightGBM/lgbm_multiclass.txt')

    pred = bst_lgb.predict(df[training_variables], num_iteration=50) #bst_lgb.best_iteration
    lgb_out = pd.DataFrame(pred, columns=['signal_prob','continuum_prob','fakeD_prob','fakeB_prob'])

    df_lgb = pd.concat([df, lgb_out], axis=1)
    df_lgb['largest_prob'] = df_lgb[['signal_prob','continuum_prob','fakeD_prob','fakeB_prob']].max(axis=1)
    del pred, df, lgb_out

    print(colored(f'Group the samples with MVA outputs', 'green'))
    
    cut = 'signal_prob==largest_prob and signal_prob>0.8 and \
    D_daughter_pValue_min>D_daughter_pValue_mean/100'
    df_lgb=df_lgb.query(cut)

    # apply the mva cut
    df_bestSelected=df_lgb.loc[df_lgb.groupby(['__experiment__','__run__','__event__','__production__']).B_D_ReChi2.idxmin()]
    df, samples=util.get_dataframe_samples(df_bestSelected, args.lmode, template=True)
    
    
    # Initialize an empty list to store the subsets
    subsets = [[] for _ in range(num_subsets)]

    # Split each DataFrame within the dictionary into 9 subsets
    for key, df in samples.items():
        # Split the DataFrame into subsets
        df_subsets = np.array_split(df, num_subsets)

        # Append the subsets to the corresponding list in 'subsets'
        for i in range(num_subsets):
            subsets[i].append(df_subsets[i])

    # Merge the corresponding subsets across keys
    merged_subsets = [pd.concat(subset) for subset in subsets]

    # Calculate the number of rows in each merged subset
    subset_sizes = [len(subset) for subset in merged_subsets]
    
    print(colored(f'{subset_sizes=}', 'blue'))
    
    for i, df in enumerate(merged_subsets):
        # Define the name of the root file for the subsets
        root_file_name = f"{output_directory}subset_{i}.root"
    
        with uproot.recreate(root_file_name) as file:
            file['B0'] = df
