# +
"""
Autogluon training script.

Usage: python3 2_Apply_MVA.py

"""
from autogluon.tabular import TabularPredictor
import uproot
import pandas as pd
import argparse
import utilities as util

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

    training_variables = util.training_variables
    columns = util.all_relevant_variables + ['B0_deltaE','B0_CMS2_weMbc','B0_CMS0_weDeltae']
    
    # Load data files
    MC_4S = uproot.concatenate([f'Samples/Generic_MC15ri/e_channel/MC15ri_local_200fb_control/*.root:B0'],
                          library="np",
                          #cut = '(D_M>1.855) & (D_M<1.885)',
                          filter_branch=lambda branch: branch.name in columns)
    
    df_mc_4S = pd.DataFrame(MC_4S)
    predictor = TabularPredictor.load("AutogluonModels/ag-20241122_085044")
    pred = predictor.predict_proba(df_mc_4S)
    pred = pred.rename(columns={0: 'fakeTracks_prob', 
                            1: 'fakeD_prob',
                            2: 'fakeL_prob',
                            3: 'continuum_prob',
                            4: 'combinatorial_prob',
                            5: 'singleBbkg_prob',
                            8: 'sig_prob'})
    df_mc_4S_pred = pd.concat([df_mc_4S, pred], axis=1)
    
    with uproot.recreate(f'mc_4S_pred.root') as file:
        file['B0'] = df_mc_4S_pred
