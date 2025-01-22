# +
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit toys with b2luigi

Usage: python3 9_Toyfit_pipeline.py

"""
import plotting
import plotting_style

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cabinetry
import pyhf
import uncertainties
from tqdm.auto import tqdm

import b2luigi
import json
import yaml
# import logging
# import pickle


def to_string(x):
    if isinstance(x, float):
        return f'{x:.3f}'
    if isinstance(x, int):
        return f'{x:d}'
    if isinstance(x, str):
        return x
    # default and hope the instance has an inbuilt __str__
    return str(x)


def join_list_hash_function(x):
    if x is None:
        return 'None'
    return '_'.join([to_string(y) for y in x])


def hash_function_dict_name_key(x):
    if not 'name' in x.keys():
        raise ValueError
    return x['name']


class pyhf_toy_mle_part_fitTask(b2luigi.Task):
    """
    Task to perform an mle fit on toys of the asimov sample on a prepared pyhf spec.
    """
    queue = 'l'
    
    samples_toFix = b2luigi.ListParameter(default=[],significant=True,
                                          hashed=True,
                                          hash_function=join_list_hash_function)
    
    temp_workspace = b2luigi.Parameter(default='',
                                       hashed=True,
                                       description='relative path to the template workspace')
    test_workspace = b2luigi.Parameter(default='',
                                       hashed=True,
                                       description='relative path to the test workspace')
    
    init_toy_pars = b2luigi.ListParameter(default=[1]*7, 
                                          hashed=True,
                                          hash_function=join_list_hash_function)
    
    n_toys = b2luigi.IntParameter(default=10, significant=False,description='Number of toys to generate')

    part = b2luigi.IntParameter(
        default=0, description='Needed to be able to run N jobs (each with a part number) with n toys each')
    
    binning = b2luigi.ListParameter(default=[0,0], significant=False,
                                    hashed=True, hash_function=join_list_hash_function)

    # safety factor to account for failed toys
    # runs until self.n_toys successful fits or safety_factor*self.n_toys attempts.
    toy_safety_factor = 2

    job_name = 'pyhfAsimovToyMLE'
    htcondor_settings = {
        'request_cpus': 1,
        'request_memory': '4000 MB',
        '+RequestRuntime': 3600 * 3,
    }


    def output(self):
        yield self.add_to_output('toy_part_results.json')

    def run(self):
        
        # load templates
        spec_temp = cabinetry.workspace.load(self.temp_workspace)
        model, _ = cabinetry.model_utils.model_and_data(spec_temp)
        
        # Get the list of all sample names from the model
        all_sample_names = model.config.samples
        # Create a boolean list for fixing parameters
        fix_mask = [sample in self.samples_toFix for sample in all_sample_names]
        
        # Retrieve all parameter names in the correct order
        all_parameter_names = model.config.par_order
        # Create a list of parameter names for samples that are not fixed
        minos_parameters = [param for param, fix in zip(all_parameter_names, fix_mask) if not fix]
        
        
        # Generate toys
        if self.test_workspace == '':
            # initialize toys from temp workspace
            toy_pars = model.config.suggested_init()
            toy_pars[:len(self.init_toy_pars)] = self.init_toy_pars
            pdf_toy = model.make_pdf(pyhf.tensorlib.astensor(toy_pars))
        else:
            # load test model from test workspace
            spec_test = cabinetry.workspace.load(self.test_workspace)
            model_test, _ = cabinetry.model_utils.model_and_data(spec_test)
            
            # set initialisation for generating toys     
            toy_pars = model_test.config.suggested_init()
            toy_pars[:len(self.init_toy_pars)] = self.init_toy_pars
            # generate the toys:
            pdf_toy = model_test.make_pdf(pyhf.tensorlib.astensor(toy_pars))
        
        toys = pdf_toy.sample((self.n_toys,))
        
#         par_vals = par_vals.tolist()
#         par_bounds = par_bounds.tolist()
        
        # prepare containers for fit results
        fit_results = {
            'best_twice_nll': [],
            'pval': [],
            'best_fit': [],
            'uncertainty': [],
            'minos_uncertainty_up': [],
            'minos_uncertainty_down': []
        }
                
        failed_fits = 0
        attempted_fits = 0
        successful_fits = 0
        
        # fit toys
        with tqdm(total=self.n_toys, desc='Fitting toys') as pbar:
            while attempted_fits < self.n_toys:
                data = toys[attempted_fits]
                
                try:
                    try:
                        pyhf.set_backend('jax', 'scipy')
                        init_pars = pyhf.infer.mle.fit(data=data, pdf=model,
                                                       init_pars=toy_pars,
                                                       fixed_params=fix_par).tolist()

                    except:
                        init_pars = toy_pars


                    pyhf.set_backend('jax', 'minuit')
                    res = cabinetry.fit.fit(
                        model,
                        data=data,
                        init_pars=init_pars,
                        fix_pars = fix_mask,
                        # par_bounds=par_bounds,
                        goodness_of_fit=True,
                        minos=minos_parameters,
                    )

                    # save fit results
                    fit_results['best_twice_nll'].append(res.best_twice_nll)
                    fit_results['pval'].append(res.goodness_of_fit)
                    bes = res.bestfit[:len(self.init_toy_pars)].tolist()
                    err = res.uncertainty[:len(self.init_toy_pars)].tolist()
                    fit_results['best_fit'].append(bes)
                    fit_results['uncertainty'].append(err)

#                     main_data, aux_data = model.fullpdf_tv.split(pyhf.tensorlib.astensor(data))
#                     fit_results['main_data'].append(main_data.tolist())
#                     fit_results['aux_data'].append(aux_data.tolist())
                    
                    fit_results['minos_uncertainty_up'].append(
                        [abs(res.minos_uncertainty[x][1]) for x in minos_parameters])
                    fit_results['minos_uncertainty_down'].append(
                        [abs(res.minos_uncertainty[x][0]) for x in minos_parameters])

                    successful_fits += 1
                    pbar.update(1)

                except:
                    failed_fits += 1
                attempted_fits += 1
            pbar.close()

        for key in fit_results.keys():
            # convert to json safe lists (these are much quicker to load then the yaml files later)
            fit_results[key] = np.array(fit_results[key]).tolist()

        out_dict = {
            'poi': all_parameter_names[:len(self.init_toy_pars)],
            'toy_pars': self.init_toy_pars,
            'n_toys': self.n_toys,
            'part': self.part,
            'results': fit_results,
            'failed_fits': failed_fits,
            'atempted_fits': attempted_fits,
            'binning': self.binning,
        }

        with open(self.get_output_file_name('toy_part_results.json'), 'w') as f:
            json.dump(out_dict, f, indent=4)
        return


class pyhf_toy_fitTask(b2luigi.Task):
    """
    Task to summarise mle fits on toys of the asimov sample on a prepared pyhf spec.
    """
    queue = 'l'
    
    samples_toFix = b2luigi.ListParameter(default=[], significant=True,
                                          hashed=True,
                                          hash_function=join_list_hash_function)
    
    temp_workspace = b2luigi.Parameter(default='',
                                       hashed=True,
                                       description='relative path to the template workspace')
    test_workspace = b2luigi.Parameter(default='',
                                       hashed=True,
                                       description='relative path to the test workspace')
    
    init_toy_pars = b2luigi.ListParameter(default=[1]*7, 
                                          hashed=True,
                                          hash_function=join_list_hash_function)
    
    n_total_toys = b2luigi.IntParameter(default=1000, significant=False,description='Number of toys to generate')

    normalise_by_uncertainty = b2luigi.BoolParameter(default=True,
                                                     significant=False,
                                                     description='Whether to calculate mean or (mean-mu)/err')
    
    job_name = 'toyGauss'
    htcondor_settings = {
        'request_cpus': 1,
        'request_memory': '4000 MB',
        '+RequestRuntime': 3600,
    }

    def requires(self):
        n_max_toys_per_job = 10
        for part in range(int(np.ceil(self.n_total_toys / n_max_toys_per_job))):
            yield self.clone(pyhf_toy_mle_part_fitTask,
                             temp_workspace = self.temp_workspace,
                             test_workspace = self.test_workspace,
                             samples_toFix = self.samples_toFix,
                             init_toy_pars = self.init_toy_pars,
                             part=part,
                             n_toys=n_max_toys_per_job)
      
            
    def output(self):
        yield self.add_to_output('toy_results.json')
        n_plot = len(self.init_toy_pars) - len(self.samples_toFix)
        for i in range(n_plot):
            yield self.add_to_output(f'toy_fit_pulls_{i}.pdf')

    def run(self):
        import fit_utils

        # load all the results
        merged_toy_results_dict = {}
        failed_fits = 0
        for input_file in tqdm(self.get_input_file_names('toy_part_results.json')):
            with open(input_file, 'r') as f:
                in_dict = json.load(f)
                failed_fits += in_dict['failed_fits']
                merged_toy_results_dict['poi'] = in_dict['poi']
                merged_toy_results_dict['toy_pars'] = in_dict['toy_pars']
                
                for k, v in in_dict['results'].items():
                    if not k in merged_toy_results_dict.keys():
                        merged_toy_results_dict[k] = v
                    else:
                        merged_toy_results_dict[k].extend(v)

        out_dict = {
            'n_total_toys': self.n_total_toys,
            'toy_results': merged_toy_results_dict,
            'failed_fits': failed_fits,
            'gauss_results': {},
        }

        # fit and plot the gaussians
        plotting_style.set_matplotlibrc_params()
        nplot = 0
        par_to_ignore = [par+'_norm' for par in self.samples_toFix]
        
        for poi_index, poi in enumerate(merged_toy_results_dict['poi']):
            if poi in par_to_ignore:
                continue
                
            fitted = np.array(merged_toy_results_dict['best_fit'])[:,poi_index]
            truth = merged_toy_results_dict['toy_pars'][poi_index]
            diff = fitted - truth
            
            # calculate pulls
            if self.normalise_by_uncertainty:
                hesse_error = np.array(merged_toy_results_dict['uncertainty'])[:,poi_index]
                percent_error = hesse_error / fitted # will show in the plot
                if 'minos_uncertainty_up' in merged_toy_results_dict.keys():
                    # minos errors
                    minos_up = np.array(merged_toy_results_dict['minos_uncertainty_up'])[:,poi_index]
                    minos_down = np.array(merged_toy_results_dict['minos_uncertainty_down'])[:,poi_index]
                    pulls = np.where(diff > 0, diff / minos_up, diff / minos_down)
                else:
                    # hesse error
                    pulls = diff / hesse_error
            else:
                pulls = diff

            mu, sigma = fit_utils.minuit_gauss(pulls)

            out_dict['gauss_results'][poi] = {
                'mu': [float(mu.n), float(mu.s)],
                'sigma': [float(sigma.n), float(sigma.s)],
                'corr': uncertainties.correlation_matrix([mu, sigma]).tolist(),
                'simple_mean': np.mean(pulls),
                'simple_std': np.std(pulls)
            }

            plotting.plot_toy_gaussian(
                pulls,
                mu=mu if abs(sigma.n)<5 else uncertainties.ufloat(0, 0.1),
                sigma=sigma if abs(sigma.n)<5 else uncertainties.ufloat(1, 0.1),
                vertical_lines=([0,]),
                xlabel='$(\mu-\mu_{in}) /\sigma_{\mu}$' if self.normalise_by_uncertainty else r'$\mu-\mu_{in}$',
                center_bins_on_zero=self.normalise_by_uncertainty,
                gauss_color=(plotting_style.BrightColorScheme.pink
                             if poi_index<3 else plotting_style.BrightColorScheme.orange),
                extra_info=f'Percent Error: {percent_error.mean():.3f}' if abs(sigma.n)<5 else 'Fit failed',
                title_info= poi,
                file_name=self.get_output_file_name(f'toy_fit_pulls_{nplot}.pdf'),
            )
            nplot+=1

        with open(self.get_output_file_name('toy_results.json'), 'w') as f:
            json.dump(out_dict, f, indent=4)
        return


class pyhf_linearity_fitTask(b2luigi.Task):
    """
    Task to summarise mle fits on toys of the asimov sample on a prepared pyhf spec.
    """
    
    queue = 'l'
    
    samples_toFix = b2luigi.ListParameter(default=[], significant=True,
                                          hashed=True,
                                          hash_function=join_list_hash_function)
    
    temp_workspace = b2luigi.Parameter(default='',
                                       hashed=True,
                                       description='relative path to the template workspace')
    test_workspace = b2luigi.Parameter(default='',
                                       hashed=True,
                                       description='relative path to the test workspace')
    
    n_toys_per_point = b2luigi.IntParameter(default=10, 
                                            significant=False,
                                            description='Number of toys to generate for each test point')
    
    linearity_parameter_bonds = b2luigi.ListParameter(default=[0,1], 
                                            hashed=True,
                                            hash_function=join_list_hash_function)
    
    n_test_points = b2luigi.IntParameter(default=50, significant=False,description='Number of test points')
    
    job_name = 'toyLinearity'


    htcondor_settings = {
        'request_cpus': 1,
        'request_memory': '16000 MB',
        '+RequestRuntime': 3600,
    }

    def requires(self):
        lin_start = self.linearity_parameter_bonds[0]
        lin_end = self.linearity_parameter_bonds[1]
        
        n_max_toys_per_job = 10
        n_parameters = 12 - len(self.samples_toFix)
        
        # randomize the initial parameters
        for test_point in range(self.n_test_points):
            rng = np.random.default_rng(test_point)
            toy_pars = list(rng.uniform(lin_start,lin_end,n_parameters))
            
            for part in range(int(np.ceil(self.n_toys_per_point / n_max_toys_per_job))):
                yield self.clone(pyhf_toy_mle_part_fitTask,
                                 temp_workspace = self.temp_workspace,
                                 test_workspace = self.test_workspace,
                                 samples_toFix = self.samples_toFix,
                                 init_toy_pars = toy_pars,
                                 part=part,
                                 n_toys=n_max_toys_per_job)

    def output(self):
        yield self.add_to_output('linearity_toy_results.yaml')
        n_plot = 12 - len(self.samples_toFix)
        for i in range(n_plot):
            yield self.add_to_output(f'toy_fit_linearity_{i}.pdf')

    def run(self):
        import fit_utils

        # summarize results
        merged_toy_results_dict = {}
        failed_fits = 0
        for input_file in tqdm(self.get_input_file_names('toy_part_results.json')):
            with open(input_file, 'r') as f:
                in_dict = json.load(f)
                failed_fits += in_dict['failed_fits']
                par_names = in_dict['poi']
                # Convert the list to a tuple to make it hashable, keys in a dict
                truth = tuple(in_dict['toy_pars'])
                fitted = in_dict['results']['best_fit']

                # Check if the truth value already exists in the merged data
                if truth in merged_toy_results_dict:
                    # If it exists, append the fitted value to the existing entry
                    merged_toy_results_dict[truth].extend(fitted)
                else:
                    # If it doesn't exist, create a new key,value with the truth and the fitted
                    merged_toy_results_dict[truth] = fitted
        
        # average results for each test point
        processed_results = {'fitted':[],
                             'error':[],
                             'truth':[],
                             'poi': par_names}
        
        for truth, fits in merged_toy_results_dict.items():
            processed_results['truth'].append(truth)
            avg = np.mean(fits,axis=0).tolist()
            err = np.std(fits,axis=0).tolist()
            
            processed_results['fitted'].append(avg)
            processed_results['error'].append(err)
            

        out_dict = {
            'n_test_points': self.n_test_points,
            'n_toys_per_point': self.n_toys_per_point,
            'failed_fits': failed_fits,
            'toy_results': processed_results,
            'linearity_results': {},
        }

        par_to_ignore = [par+'_norm' for par in self.samples_toFix]
        nplot = 0
        
        for poi_index, poi in enumerate(processed_results['poi']):
            if poi in par_to_ignore:
                continue
                
            truth = np.array(processed_results['truth'])[:,poi_index]
            fitted = np.array(processed_results['fitted'])[:,poi_index]
            error = np.array(processed_results['error'])[:,poi_index]

            slope, intercept = fit_utils.minuit_linear(truth, fitted, error)

            out_dict['linearity_results'][poi] = {
                'fitted': fitted.tolist(),
                'error_bar': error.tolist(),
                'slope': [float(slope.n), float(slope.s)],
                'intercept': [float(intercept.n), float(intercept.s)],
            }
            
            colors = [
                    plotting_style.BrightColorScheme.blue_medium, 
                    plotting_style.BrightColorScheme.pink,
                    plotting_style.BrightColorScheme.teal
                ]

            #now make the plots:
            plotting_style.set_matplotlibrc_params()
            plotting.plot_linearity_test(
                truth,
                fitted,
                error,
                slope,
                intercept,
                bonds=[0,self.linearity_parameter_bonds[1]],
                color=colors[0],
                x_offset=[0],
#                 line_infos=line_infos,
#                 extra_info=fr'$\mu^{{B^{{+}}}}_{{\rm WA}}$'
#                 if poi == 'mu_wa_charged' else fr'$\mu^{{B^{{0}}}}_{{\rm WA}}$',
                title_info= poi,
                file_name=self.get_output_file_name(f'toy_fit_linearity_{nplot}.pdf'),
            )
            nplot+=1

        with open(self.get_output_file_name('linearity_toy_results.yaml'), 'w') as f:
            yaml.dump(out_dict, f)

        return


class pyhf_binning_fitTask(b2luigi.Task):
    """
    Task to summarise mle fit uncertainties on toys with different binning.
    """
    
    queue = 'l'
    
    samples_toFix = b2luigi.ListParameter(default=[], significant=True,
                                          hashed=True,
                                          hash_function=join_list_hash_function)
    
    binning = b2luigi.ListParameter(default=[10,81,5], significant=False,
                                    hashed=True,
                                    hash_function=join_list_hash_function,
                                    description='first_nbin, last_nbin+1, increment')
    
    n_toys_per_binning = b2luigi.IntParameter(default=10, 
                                            significant=False,
                                            description='Number of toys to generate for each test binning')
    
    create_templates = b2luigi.BoolParameter(default=True, significant=False)
    
    staterror = b2luigi.BoolParameter(default=False, significant=False)
    
    workspace_path = b2luigi.Parameter(default='',
                                       hashed=True,
                                       significant=False, description='relative path to pyhf workspace')
    
    job_name = 'toyBinning'


    htcondor_settings = {
        'request_cpus': 1,
        'request_memory': '16000 MB',
        '+RequestRuntime': 3600,
    }

    def requires(self):
        if self.create_templates:
            import uproot
            import sys
            sys.path.append('/home/belle/zhangboy/inclusive_R_D/')
            import utilities as util
            
            ## Loading Ntuples
            columns = util.all_relevant_variables + ['B0_CMS4_weMissM2']

            # Load template samples
            e_temp = uproot.concatenate([f'../Samples/Generic_MC15ri/e_channel/MC15ri_local_200fb/*.root:B0'],
                                      library="np",
                                      #cut=input_cut,
                                      filter_branch=lambda branch: branch.name in columns)

            df_e = pd.DataFrame(e_temp)

            # load MVA
            import lightgbm as lgb
            training_variables = util.training_variables
            bst_lgb = lgb.Booster(model_file=f'../BDTs/LightGBM/lgbm_multiclass.txt')
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
        else:
            pass
        
        # define binning
        nBins = np.arange(*self.binning) # 10,15,20...100
        
        for nMM2_bins in nBins:
            for nPDl_bins in nBins:
                workspace_path = f'temp/binning_{nMM2_bins}_{nPDl_bins}.json'
                
                if self.create_templates:
                    # Define the bin edges
                    MM2_bins = np.linspace(-5, 10, nMM2_bins + 1)
                    p_D_l_bins = np.linspace(0.4, 5, nPDl_bins + 1)

                    # create templates
                    te=util.get_dataframe_samples_new(df_bestSelected_e, 'e', template=False)
                    indices_threshold_3,temp_asimov_e,temp_asimov_merged_e = util.create_templates(
                        samples=te, bins=[MM2_bins, p_D_l_bins], 
                        variables=['B0_CMS3_weMissM2','p_D_l'],
                        bin_threshold=1,merge_threshold=10,
                        sample_to_exclude=[#'bkg_FakeD','bkg_TDFl','bkg_continuum','bkg_combinatorial','bkg_singleBbkg',
                                           'bkg_fakeTracks','bkg_other_TDTl','bkg_other_signal'])

                    # load,update,save example workspace
                    spec = cabinetry.workspace.load(self.workspace_path)
                    spec = util.update_workspace(workspace=spec,
                                 temp_asimov_sets=[temp_asimov_e],
                                 staterror=self.staterror)
                    cabinetry.workspace.save(spec, workspace_path)
                else:
                    pass
                
                # submit fit task
                n_parameters = 12 - len(self.samples_toFix)
                toy_pars = [1]*n_parameters

                yield self.clone(pyhf_toy_mle_part_fitTask,
                                 samples_toFix = self.samples_toFix,
                                 workspace_path = workspace_path,
                                 test_fakeD_sideband = False,
                                 init_toy_pars = toy_pars,
                                 n_toys=self.n_toys_per_binning,
                                 binning=[int(nMM2_bins), int(nPDl_bins)])

    def output(self):
        yield self.add_to_output('binning_toy_results.csv')

    def run(self):

        # average results for each test binning
        processed_results = {'nMM2_bins':[],
                             'nPDl_bins':[]}

        for input_file in tqdm(self.get_input_file_names('toy_part_results.json')):
            with open(input_file, 'r') as f:
                in_dict = json.load(f)
                successful_fits = in_dict['atempted_fits'] - in_dict['failed_fits']
                
                # calculate the mean and error of hesse
                hesse = in_dict['results']['uncertainty']
                avg = np.mean(hesse,axis=0).tolist()
                if successful_fits==0:
                    err = [0]*len(avg)
                else:
                    err = (np.std(hesse,axis=0)/np.sqrt(successful_fits)).tolist()
                
                # Append the avg values to the corresponding key in the dictionary
                pars_hesse_avg = [par+'_hesse_avg' for par in in_dict['poi']]
                pars_hesse_err = [par+'_hesse_err' for par in in_dict['poi']]
                for i, par in enumerate(pars_hesse_avg):
                    if par in processed_results:
                        processed_results[par].append(avg[i])
                    else:
                        processed_results[par] = [avg[i]]
                        
                for i, par in enumerate(pars_hesse_err):
                    if par in processed_results:
                        processed_results[par].append(err[i])
                    else:
                        processed_results[par] = [err[i]]
                
                processed_results['nMM2_bins'].append(in_dict['binning'][0])
                processed_results['nPDl_bins'].append(in_dict['binning'][1])
        
        # save the result to a csv file
        df = pd.DataFrame.from_dict(processed_results)
        df.to_csv(self.get_output_file_name('binning_toy_results.csv'),index=False)

        return

class pyhf_toys_wrapper(b2luigi.WrapperTask):
    """
    Wrapper for submitting all the pyhf asimov toys tasks.
    """

    samples_toFix = b2luigi.ListParameter(default=[], significant=True,
                                          hashed=True,
                                          hash_function=join_list_hash_function)
    temp_workspace = b2luigi.Parameter(default='',
                                       hashed=True,
                                       description='relative path to the template workspace')
    test_workspace = b2luigi.Parameter(default='',
                                       hashed=True,
                                       description='relative path to the test workspace')
    
    def requires(self):
    
#         yield self.clone(pyhf_binning_fitTask,
#                          binning = [10,76,5],
#                          samples_toFix = self.samples_toFix,
#                          n_toys_per_point = 25,
#                          workspace_path = self.workspace_path,
#                          create_templates = False,
#                          staterror = True)
    
        yield self.clone(pyhf_toy_fitTask,
                         temp_workspace = self.temp_workspace,
                         test_workspace = self.test_workspace,
                         samples_toFix = self.samples_toFix,
                         n_total_toys = 2000,
                         init_toy_pars = [1]*12,
                         normalise_by_uncertainty = True)
        
        yield self.clone(pyhf_linearity_fitTask,
                         temp_workspace = self.temp_workspace,
                         test_workspace = self.test_workspace,
                         samples_toFix = self.samples_toFix,
                         linearity_parameter_bonds = [0.8,1.2],
                         n_toys_per_point = 30,
                         n_test_parameters = 12,
                         n_test_points = 40)


if __name__ == '__main__':

    # set_b2luigi_settings('weak_annihilation_settings/weak_annihilation_settings.yaml')
    
    b2luigi.process(
        pyhf_toys_wrapper(test_workspace='2d_ws_SR_e_60_60.json',
                          temp_workspace='2d_ws_SR_e_60_60_SBFakeD.json',
                          samples_toFix = ['bkg_fakeD','bkg_TDFl',
                                           'bkg_continuum',
                                           'bkg_combinatorial',
                                           'bkg_singleBbkg',
                                           #'bkg_fakeTracks',
                                          ]),
        workers=int(1e4),
        batch=True,
        ignore_additional_command_line_args=False
        # set it to True to use additional args for this version
#         workers=1,
#         batch=False,
    )
