# +
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit toys with b2luigi

Usage: python3 9_Toyfit_pipeline.py

"""
import numpy as np
import pandas as pd

import cabinetry
import pyhf
import uncertainties
from tqdm.auto import tqdm

import b2luigi
import json
import yaml

import sys
sys.path.append('/home/belle/zhangboy/inclusive_R_D/')
import utilities as util

def to_string(x):
    """
    Convert various data types to a string format.

    Args:
        x (int, float, str, or any object): The object to convert.

    Returns:
        str: The formatted string representation.
    """
    if isinstance(x, float):
        return f'{x:.3f}'
    if isinstance(x, int):
        return f'{x:d}'
    if isinstance(x, str):
        return x
    return str(x)  # Default to string conversion


def join_list_hash_function(x):
    """
    Create a hashable string representation of a list.

    Args:
        x (list or None): A list of elements to convert.

    Returns:
        str: A string where elements are joined by underscores.
    """
    if x is None:
        return 'None'
    return '_'.join([to_string(y) for y in x])


def hash_function_dict_name_key(x):
    """
    Extracts the 'name' key from a dictionary for hashing.

    Args:
        x (dict): Dictionary containing a 'name' key.

    Returns:
        str: The value associated with the 'name' key.

    Raises:
        ValueError: If the key 'name' is missing.
    """
    if 'name' not in x:
        raise ValueError("Dictionary must contain a 'name' key.")
    return x['name']



class pyhf_toy_mle_part_fitTask(b2luigi.Task):
    
    queue = 'l'
    
    pars_toFix = b2luigi.ListParameter(default=[],significant=False,hashed=True,
                                       hash_function=join_list_hash_function)
    
    toy_workspace = b2luigi.Parameter(default='',hashed=True)
    
    fit_workspace = b2luigi.Parameter(default='',hashed=True)
    
    toy_pars = b2luigi.ListParameter(default=[1]*7,hashed=True,significant=True,
                                     hash_function=join_list_hash_function)
    
    fit_inits = b2luigi.ListParameter(default=[1]*7,hashed=True,significant=False,
                                      hash_function=join_list_hash_function)
    
    n_toys = b2luigi.IntParameter(default=10, significant=False)

    part = b2luigi.IntParameter(default=0, description='Needed to be able to run N jobs (each with a part number) with n toys each')
    
    binning = b2luigi.ListParameter(default=[0,0], significant=False,hashed=True,
                                    hash_function=join_list_hash_function)

    job_name = 'pyhfAsimovToyMLE'


    def output(self):
        yield self.add_to_output('toy_part_results.json')

    def run(self):
        toy_tools = util.toy_utils(toy_workspace=self.toy_workspace,
                                   fit_workspace=self.fit_workspace,
                                   pars_toFix = self.pars_toFix, 
                                   part=self.part, binning=self.binning)
        
        result_dict = toy_tools.generate_fit_toys(toy_pars=self.toy_pars,
                                                  fit_inits=self.fit_inits, 
                                                  n_toys=self.n_toys)

        with open(self.get_output_file_name('toy_part_results.json'), 'w') as f:
            json.dump(result_dict, f, indent=4)
        return


class pyhf_toy_fitTask(b2luigi.Task):
    
    queue = 'l'
    
    pars_toFix = b2luigi.ListParameter(default=[],significant=False,hashed=True,
                                       hash_function=join_list_hash_function)
    
    toy_workspace = b2luigi.Parameter(default='',hashed=True)
    
    fit_workspace = b2luigi.Parameter(default='',hashed=True)
    
    toy_pars = b2luigi.ListParameter(default=[1]*7,hashed=True,significant=False,
                                     hash_function=join_list_hash_function)
    
    fit_inits = b2luigi.ListParameter(default=[1]*7,hashed=True,significant=False,
                                      hash_function=join_list_hash_function)
    
    n_total_toys = b2luigi.IntParameter(default=1000, significant=False)
    
    n_toys_per_job = b2luigi.IntParameter(default=5,significant=False)

    normalize_by_uncertainty = b2luigi.BoolParameter(default=True,significant=False,
                            description='Whether to calculate mean or (mean-mu)/err')
    
    job_name = 'toyGauss'


    def requires(self):

        for part in range(int(np.ceil(self.n_total_toys / self.n_toys_per_job))):
            yield pyhf_toy_mle_part_fitTask(
                toy_workspace = self.toy_workspace,
                fit_workspace = self.fit_workspace,
                pars_toFix = self.pars_toFix,
                toy_pars = self.toy_pars,
                fit_inits = self.fit_inits,
                part=part,
                n_toys=self.n_toys_per_job
            )                                   
#             yield self.clone(pyhf_toy_mle_part_fitTask,
      
            
    def output(self):
        yield self.add_to_output('pulls_toy_results.json')
        n_plot = len(self.toy_pars)
        for i in range(n_plot):
            yield self.add_to_output(f'toy_fit_pulls_{i}.pdf')

    def run(self):
        # load and merge all the toy results
        merged_dict = util.toy_utils.merge_toy_results(result_files=self.get_input_file_names('toy_part_results.json') )
        
        # calculate the pulls
        merged_dict = util.toy_utils.calculate_pulls(merged_dict=merged_dict, minos_error=True,
                                                    normalize=self.normalize_by_uncertainty)

        # fit and plot the gaussians
        nplot = 0
        merged_dict['gauss_results'] = {}
        
        for poi_index, poi in enumerate(merged_dict['toy_results']['poi']):
            if poi in self.pars_toFix:
                continue

            pulls = np.array(merged_dict['toy_results']['pulls'])[:,poi_index]
            percent_error = np.array(merged_dict['toy_results']['percent_error'])[:,poi_index]
            
            # fit pulls with a gaussian
            mu, sigma = util.fit_pull_linearity.fit_gauss(pulls)
            merged_dict['gauss_results'][poi] = {
                'mu': [float(mu.n), float(mu.s)],
                'sigma': [float(sigma.n), float(sigma.s)],
                'corr': uncertainties.correlation_matrix([mu, sigma]).tolist(),
                'simple_mean': np.mean(pulls),
                'simple_std': np.std(pulls)
            }

            # make plots
            util.toy_utils.plot_toy_gaussian(
                pulls,
                mu=mu if abs(sigma.n)<5 else uncertainties.ufloat(0, 0.1),
                sigma=sigma if abs(sigma.n)<5 else uncertainties.ufloat(1, 0.1),
                vertical_lines=([0,]), title_info= poi,
                xlabel='$(\mu-\mu_{in}) /\sigma_{\mu}$' if self.normalize_by_uncertainty else r'$\mu-\mu_{in}$',
                extra_info=f'Percent Error: {percent_error.mean():.3f}' if abs(sigma.n)<5 else 'Fit failed',
                file_name=self.get_output_file_name(f'toy_fit_pulls_{nplot}.pdf'),
            )
            nplot+=1
        with open(self.get_output_file_name('pulls_toy_results.json'), 'w') as f:
            json.dump(merged_dict, f, indent=4)
        return


class pyhf_linearity_fitTask(b2luigi.Task):
    queue = 'l'
    
    pars_toFix = b2luigi.ListParameter(default=[],significant=False,hashed=True,
                                       hash_function=join_list_hash_function)
    
    toy_workspace = b2luigi.Parameter(default='',hashed=True)
    
    fit_workspace = b2luigi.Parameter(default='',hashed=True)
    
    n_pars = b2luigi.IntParameter(default=12,significant=False)
    
    n_toys_per_point = b2luigi.IntParameter(default=10,significant=False)
    
    n_toys_per_job = b2luigi.IntParameter(default=5,significant=False)
    
    linearity_parameter_bonds = b2luigi.ListParameter(default=[0,1],hashed=True,
                                            hash_function=join_list_hash_function)
    
    n_test_points = b2luigi.IntParameter(default=50, significant=False)
    
    job_name = 'toyLinearity'


    def requires(self):
        # generate random signal strengths 
        lin_start = self.linearity_parameter_bonds[0]
        lin_end = self.linearity_parameter_bonds[1]
        
        for test_point in range(self.n_test_points):
            rng = np.random.default_rng(test_point)
            toy_pars = list(rng.uniform(lin_start,lin_end,self.n_pars).round(3))
#             toy_pars += [0.4] * n_fixed_pars
            
            for part in range(int(np.ceil(self.n_toys_per_point / self.n_toys_per_job))):
                yield pyhf_toy_mle_part_fitTask(
                    toy_workspace = self.toy_workspace,
                    fit_workspace = self.fit_workspace,
                    pars_toFix = self.pars_toFix,
                    toy_pars = toy_pars,
                    fit_inits = toy_pars,
                    part=part,
                    n_toys=self.n_toys_per_job
                )             
#                 yield self.clone(pyhf_toy_mle_part_fitTask,


    def output(self):
        yield self.add_to_output('linearity_toy_results.yaml')
        n_plot = self.n_pars
        for i in range(n_plot):
            yield self.add_to_output(f'toy_fit_linearity_{i}.pdf')

    def run(self):
        # load all the results
        merged_dict = util.toy_utils.merge_toy_results(result_files=self.get_input_file_names('toy_part_results.json') )

        # calculate the line x y coordinates and yerr
        merged_dict = util.toy_utils.calculate_linear_xy(merged_dict=merged_dict, minos_error=True)
        
        # fit and plot the line
        nplot = 0
        merged_dict['linearity_results'] = {}
        
        for poi_index, poi in enumerate(merged_dict['toy_results']['poi']):
            if poi in self.pars_toFix:
                continue

            truth = np.array(merged_dict['toy_results']['truth'])[:,poi_index]
            fitted = np.array(merged_dict['toy_results']['weighted_means'])[:,poi_index]
            error = np.array(merged_dict['toy_results']['SEM'])[:,poi_index]
        
            # fit the line with a linear function
            slope, intercept = util.fit_pull_linearity.fit_linear(truth, fitted, error)
            merged_dict['linearity_results'][poi] = {
                'fitted': fitted.tolist(),
                'error_bar': error.tolist(),
                'slope': [float(slope.n), float(slope.s)],
                'intercept': [float(intercept.n), float(intercept.s)],
            }
            
            # make plots
            util.toy_utils.plot_linearity_test(truth,fitted,error,slope,intercept,
                                               bonds=[0,self.linearity_parameter_bonds[1]],
                                               x_offset=[0],title_info= poi,
                                               file_name=self.get_output_file_name(f'toy_fit_linearity_{nplot}.pdf') )
            nplot+=1
        with open(self.get_output_file_name('linearity_toy_results.yaml'), 'w') as f:
            yaml.dump(merged_dict, f)
        return


# class pyhf_binning_fitTask(b2luigi.Task):
#     """
#     Task to summarise mle fit uncertainties on toys with different binning.
#     """
    
#     queue = 'l'
    
#     pars_toFix = b2luigi.ListParameter(default=[], significant=True,
#                                           hashed=True,
#                                           hash_function=join_list_hash_function)
    
#     binning = b2luigi.ListParameter(default=[10,81,5], significant=False,
#                                     hashed=True,
#                                     hash_function=join_list_hash_function,
#                                     description='first_nbin, last_nbin+1, increment')
    
#     n_toys_per_binning = b2luigi.IntParameter(default=10, 
#                                             significant=False,
#                                             description='Number of toys to generate for each test binning')
    
#     create_templates = b2luigi.BoolParameter(default=True, significant=False)
    
#     staterror = b2luigi.BoolParameter(default=False, significant=False)
    
#     workspace_path = b2luigi.Parameter(default='',
#                                        hashed=True,
#                                        significant=False, description='relative path to pyhf workspace')
    
#     job_name = 'toyBinning'


#     def requires(self):
#         if self.create_templates:
#             import uproot
#             import sys
#             sys.path.append('/home/belle/zhangboy/inclusive_R_D/')
#             import utilities as util
            
#             ## Loading Ntuples
#             columns = util.all_relevant_variables + ['B0_CMS4_weMissM2']

#             # Load template samples
#             e_temp = uproot.concatenate([f'../Samples/Generic_MC15ri/e_channel/MC15ri_local_200fb/*.root:B0'],
#                                       library="np",
#                                       #cut=input_cut,
#                                       filter_branch=lambda branch: branch.name in columns)

#             df_e = pd.DataFrame(e_temp)

#             # load MVA
#             import lightgbm as lgb
#             training_variables = util.training_variables
#             bst_lgb = lgb.Booster(model_file=f'../BDTs/LightGBM/lgbm_multiclass.txt')
#             cut='signal_prob==largest_prob and signal_prob>0.8 and \
#             continuum_prob<0.04 and fakeD_prob<0.05'

#             df_e.eval('B_D_ReChi2 = B0_vtxReChi2 + D_vtxReChi2', inplace=True)
#             df_e.eval(f'p_D_l = D_CMS_p + ell_CMS_p', inplace=True)

#             pred_e = bst_lgb.predict(df_e[training_variables], num_iteration=50) #bst_lgb.best_iteration
#             lgb_out_e = pd.DataFrame(pred_e, columns=['signal_prob','continuum_prob','fakeD_prob','fakeB_prob'])

#             df_lgb_e = pd.concat([df_e, lgb_out_e], axis=1)
#             df_lgb_e['largest_prob'] = df_lgb_e[['signal_prob','continuum_prob','fakeD_prob','fakeB_prob']].max(axis=1)
#             del pred_e, lgb_out_e

#             df_cut_e=df_lgb_e.query(cut)
#             df_bestSelected_e=df_cut_e.loc[df_cut_e.groupby(['__experiment__','__run__','__event__','__production__']).B_D_ReChi2.idxmin()]
#         else:
#             pass
        
#         # define binning
#         nBins = np.arange(*self.binning) # 10,15,20...100
        
#         for nMM2_bins in nBins:
#             for nPDl_bins in nBins:
#                 workspace_path = f'temp/binning_{nMM2_bins}_{nPDl_bins}.json'
                
#                 if self.create_templates:
#                     # Define the bin edges
#                     MM2_bins = np.linspace(-5, 10, nMM2_bins + 1)
#                     p_D_l_bins = np.linspace(0.4, 5, nPDl_bins + 1)

#                     # create templates
#                     te=util.get_dataframe_samples_new(df_bestSelected_e, 'e', template=False)
#                     indices_threshold_3,temp_asimov_e,temp_asimov_merged_e = util.create_templates(
#                         samples=te, bins=[MM2_bins, p_D_l_bins], 
#                         variables=['B0_CMS3_weMissM2','p_D_l'],
#                         bin_threshold=1,merge_threshold=10,
#                         sample_to_exclude=[#'bkg_FakeD','bkg_TDFl','bkg_continuum','bkg_combinatorial','bkg_singleBbkg',
#                                            'bkg_fakeTracks','bkg_other_TDTl','bkg_other_signal'])

#                     # load,update,save example workspace
#                     spec = cabinetry.workspace.load(self.workspace_path)
#                     spec = util.update_workspace(workspace=spec,
#                                  temp_asimov_sets=[temp_asimov_e],
#                                  staterror=self.staterror)
#                     cabinetry.workspace.save(spec, workspace_path)
#                 else:
#                     pass
                
#                 # submit fit task
#                 n_parameters = 12 - len(self.pars_toFix)
#                 toy_pars = [1]*n_parameters

#                 yield self.clone(pyhf_toy_mle_part_fitTask,
#                                  pars_toFix = self.pars_toFix,
#                                  workspace_path = workspace_path,
#                                  test_fakeD_sideband = False,
#                                  init_toy_pars = toy_pars,
#                                  n_toys=self.n_toys_per_binning,
#                                  binning=[int(nMM2_bins), int(nPDl_bins)])

#     def output(self):
#         yield self.add_to_output('binning_toy_results.csv')

#     def run(self):

#         # average results for each test binning
#         processed_results = {'nMM2_bins':[],
#                              'nPDl_bins':[]}

#         for input_file in tqdm(self.get_input_file_names('toy_part_results.json')):
#             with open(input_file, 'r') as f:
#                 in_dict = json.load(f)
#                 successful_fits = in_dict['attempted_fits'] - in_dict['failed_fits']
                
#                 # calculate the mean and error of hesse
#                 hesse = in_dict['results']['uncertainty']
#                 avg = np.mean(hesse,axis=0).tolist()
#                 if successful_fits==0:
#                     err = [0]*len(avg)
#                 else:
#                     err = (np.std(hesse,axis=0)/np.sqrt(successful_fits)).tolist()
                
#                 # Append the avg values to the corresponding key in the dictionary
#                 pars_hesse_avg = [par+'_hesse_avg' for par in in_dict['poi']]
#                 pars_hesse_err = [par+'_hesse_err' for par in in_dict['poi']]
#                 for i, par in enumerate(pars_hesse_avg):
#                     if par in processed_results:
#                         processed_results[par].append(avg[i])
#                     else:
#                         processed_results[par] = [avg[i]]
                        
#                 for i, par in enumerate(pars_hesse_err):
#                     if par in processed_results:
#                         processed_results[par].append(err[i])
#                     else:
#                         processed_results[par] = [err[i]]
                
#                 processed_results['nMM2_bins'].append(in_dict['binning'][0])
#                 processed_results['nPDl_bins'].append(in_dict['binning'][1])
        
#         # save the result to a csv file
#         df = pd.DataFrame.from_dict(processed_results)
#         df.to_csv(self.get_output_file_name('binning_toy_results.csv'),index=False)

#         return

class pyhf_toys_wrapper(b2luigi.WrapperTask):
    """
    Wrapper for submitting all the pyhf asimov toys tasks.
    """

    pars_toFix = b2luigi.ListParameter(default=[],significant=False,hashed=True,
                                       hash_function=join_list_hash_function)
    
    toy_workspace = b2luigi.Parameter(default='',hashed=True)
    
    fit_workspace = b2luigi.Parameter(default='',hashed=True)
    
    def requires(self):
    
#         yield self.clone(pyhf_binning_fitTask,
#                          binning = [10,76,5],
#                          pars_toFix = self.pars_toFix,
#                          n_toys_per_point = 25,
#                          workspace_path = self.workspace_path,
#                          create_templates = False,
#                          staterror = True)
    
        yield pyhf_toy_fitTask(
            toy_workspace = self.toy_workspace,
            fit_workspace = self.fit_workspace,
            pars_toFix = self.pars_toFix,
            toy_pars = [1]*12, # set it to [1]*11 if no gap mode
            fit_inits = [1]*12, # set it to [1]*11 if no gap mode
            n_total_toys = 2000,
            n_toys_per_job = 5,
            normalize_by_uncertainty = True)
        
#         yield pyhf_linearity_fitTask(
#             toy_workspace = self.toy_workspace,
#             fit_workspace = self.fit_workspace,
#             pars_toFix = self.pars_toFix,
#             n_pars = 12, # set it to 11 if no gap mode
#             linearity_parameter_bonds = [0.8,1.2],
#             n_toys_per_point = 10,
#             n_toys_per_job = 5,
#             n_test_points = 30
#         )


if __name__ == '__main__':
    
    b2luigi.process(
        pyhf_toys_wrapper(toy_workspace='2d_ws_SR_e_testBinning_allUncer_sigMC.json',
                          fit_workspace='2d_ws_SBFakeD_e_testBinning_allUncer_sigMC.json',
                          pars_toFix = ['bkg_TDFl_norm',
                                           'bkg_fakeD_norm',
                                           'bkg_continuum_norm',
                                           'bkg_combinatorial_norm',
                                           'bkg_singleBbkg_norm',
                                           r'$D^\ast\tau\nu$_norm',
                                           r'$D^{\ast\ast}\tau\nu$_norm',
                                           #'bkg_fakeTracks_norm',
                                          ]),
        workers=int(1e3),
        batch=True,
        show_output=False,
        dry_run=False,
        test=False,
    )
