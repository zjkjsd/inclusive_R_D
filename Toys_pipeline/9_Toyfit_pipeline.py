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
import itertools
import json
import yaml
import logging
#import pickle


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
    
    workspace_path = b2luigi.Parameter(default='R_D_2d_workspace.json',
                                       hashed=True,
                                       description='relative path to pyhf workspace')
    
    init_toy_pars = b2luigi.ListParameter(default=[1]*6, 
                                          hashed=True,
                                          hash_function=join_list_hash_function)
    
    n_toys = b2luigi.IntParameter(default=100, description='Number of toys to generate')

    part = b2luigi.IntParameter(
        default=0, description='Needed to be able to run N jobs (each with a part number) with n toys each')

    store_full = b2luigi.BoolParameter(default=True, significant=False)

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
        
        # load workspace
        spec = cabinetry.workspace.load(self.workspace_path)
        model, data = cabinetry.model_utils.model_and_data(spec)
        
        # set initialisation for generating the asimov sample       
        toy_pars = model.config.suggested_init()
        toy_pars[:len(self.init_toy_pars)] = self.init_toy_pars

        # generate the toys:
        pdf_toy = model.make_pdf(pyhf.tensorlib.astensor(toy_pars))
        toys = pdf_toy.sample((self.n_toys,))
        
#         par_vals = par_vals.tolist()
#         par_bounds = par_bounds.tolist()
        
        # prepare containers for fit results
        fit_results = {
            'best_twice_nll': [],
            'pval': [],
            'best_fit': []
        }
        if self.store_full:
#             fit_results['main_data'] = []
#             fit_results['aux_data'] = []
            fit_results['uncertainty'] = []
            fit_results['minos_uncertainty_up'] = []
            fit_results['minos_uncertainty_down'] = []
            minos_parameters = model.config.par_names[:len(self.init_toy_pars)]
                
        failed_fits = 0
        attempted_fits = 0
        successful_fits = 0
        
        # fit toys
        with tqdm(total=self.n_toys, desc='Fitting toys') as pbar:
            while attempted_fits < self.n_toys:
                data = toys[attempted_fits]
                
                try:
                    if self.store_full: # minuit and minos
                        try:
                            pyhf.set_backend('jax', 'scipy')
                            init_pars = pyhf.infer.mle.fit(data=data, pdf=model).tolist()

                        except:
                            init_pars = toy_pars


                        pyhf.set_backend('jax', 'minuit')
                        res = cabinetry.fit.fit(
                            model,
                            data=data,
                            init_pars=init_pars,
                            # par_bounds=par_bounds,
                            goodness_of_fit=True,
                            minos=minos_parameters,
                        )

                        # save fit results
                        fit_results['best_twice_nll'].append(res.best_twice_nll)
                        fit_results['pval'].append(res.goodness_of_fit)
                        fit_results['best_fit'].append(res.bestfit[:len(self.init_toy_pars)].tolist())

                        main_data, aux_data = model.fullpdf_tv.split(pyhf.tensorlib.astensor(data))
#                         fit_results['main_data'].append(main_data.tolist())
#                         fit_results['aux_data'].append(aux_data.tolist())
                        fit_results['uncertainty'].append(res.uncertainty[:len(self.init_toy_pars)].tolist())

                        fit_results['minos_uncertainty_up'].append(
                            [abs(res.minos_uncertainty[x][1]) for x in res.labels[:len(self.init_toy_pars)]])
                        fit_results['minos_uncertainty_down'].append(
                            [abs(res.minos_uncertainty[x][0]) for x in res.labels[:len(self.init_toy_pars)]])
                            
                    else: # scipy only
                        pyhf.set_backend('jax', 'scipy')
                        bestfit_pars, twice_nll = pyhf.infer.mle.fit(data=data, pdf=model, return_fitted_val=True)
                        
                        # save fit results
                        fit_results['best_twice_nll'].append(twice_nll.tolist())
                        fit_results['best_fit'].append(bestfit_pars[:len(self.init_toy_pars)].tolist())
                    
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
            'poi': model.config.par_names[:len(self.init_toy_pars)],
            'toy_pars': self.init_toy_pars,
            'n_toys': self.n_toys,
            'part': self.part,
            'results': fit_results,
            'failed_fits': failed_fits,
            'atempted_fits': attempted_fits,
        }

        with open(self.get_output_file_name('toy_part_results.json'), 'w') as f:
            json.dump(out_dict, f, indent=4)
        return


class pyhf_toy_asimov_fitTask(b2luigi.Task):
    """
    Task to summarise mle fits on toys of the asimov sample on a prepared pyhf spec.
    """
    queue = 'l'
    
    workspace_path = b2luigi.Parameter(default='R_D_2d_workspace.json',
                                       hashed=True,
                                       description='relative path to pyhf workspace')
    
    n_total_toys = b2luigi.IntParameter(default=100, description='Number of toys to generate')

    normalise_by_uncertainty = b2luigi.BoolParameter(default=True,
                                                     description='Whether to calculate mean or (mean-mu)/err')

    produce_plots = b2luigi.BoolParameter(default=True, significant=False)
    
    job_name = 'toyGauss'
    htcondor_settings = {
        'request_cpus': 1,
        'request_memory': '4000 MB',
        '+RequestRuntime': 3600,
    }

    def requires(self):
        n_max_toys_per_job = 5 if self.normalise_by_uncertainty else 10
        for part in range(int(np.ceil(self.n_total_toys / n_max_toys_per_job))):
            yield self.clone(pyhf_toy_mle_part_fitTask,
                             workspace_path = self.workspace_path,
                             init_toy_pars = [1]*6,
                             part=part,
                             n_toys=n_max_toys_per_job,
                             store_full=True if self.normalise_by_uncertainty else False)
      
            
    def output(self):
        yield self.add_to_output('toy_results.json')
        if self.produce_plots:
            yield self.add_to_output('toy_fit_pulls_0.pdf')
            yield self.add_to_output('toy_fit_pulls_1.pdf')
            yield self.add_to_output('toy_fit_pulls_2.pdf')
            yield self.add_to_output('toy_fit_pulls_3.pdf')
            yield self.add_to_output('toy_fit_pulls_4.pdf')
            yield self.add_to_output('toy_fit_pulls_5.pdf')

    def run(self):
        import zfit_utils

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
        
#         fitted = np.array(merged_toy_results_dict['best_fit'][:,:6])
#         truth = merged_toy_results_dict['toy_pars']
#         diff = fitted - truth
        
        
        for poi_index, poi in enumerate(merged_toy_results_dict['poi']):
                
            fitted = np.array(merged_toy_results_dict['best_fit'])[:,poi_index]
            truth = merged_toy_results_dict['toy_pars'][poi_index]
            diff = fitted - truth
            
            # calculate pulls
            if self.normalise_by_uncertainty:
                hesse_error = np.array(merged_toy_results_dict['uncertainty'])[:,poi_index]
                pulls = diff / hesse_error
                if 'minos_uncertainty_up' in merged_toy_results_dict.keys():
                    # take minos errors
                    minos_up = np.array(merged_toy_results_dict['minos_uncertainty_up'])[:,poi_index]
                    minos_down = np.array(merged_toy_results_dict['minos_uncertainty_down'])[:,poi_index]
                    pulls = np.where(diff > 0, diff / minos_up, diff / minos_down)
                else:
                    # hesse error
                    hesse_error = np.array(merged_toy_results_dict['uncertainty'])[:,poi_index]
                    pulls = diff / hesse_error
            else:
                pulls = diff

            mu, sigma = zfit_utils.zfit_gauss(pulls)

            out_dict['gauss_results'][poi] = {
                'mu': [float(mu.n), float(mu.s)],
                'sigma': [float(sigma.n), float(sigma.s)],
                'corr': uncertainties.correlation_matrix([mu, sigma]).tolist(),
                'simple_mean': np.mean(pulls),
                'simple_std': np.std(pulls)
            }

            
            if self.produce_plots:
                plotting.plot_toy_gaussian(
                    pulls,
                    mu=mu if abs(sigma.n)<5 else uncertainties.ufloat(0, 0.1),
                    sigma=sigma if abs(sigma.n)<5 else uncertainties.ufloat(1, 0.1),
                    vertical_lines=([0,]),
                    xlabel='$(\mu-\mu_{in}) /\sigma_{\mu}$' if self.normalise_by_uncertainty else r'$\mu-\mu_{in}$',
                    center_bins_on_zero=self.normalise_by_uncertainty,
                    gauss_color=(plotting_style.BrightColorScheme.pink
                                 if poi_index<3 else plotting_style.BrightColorScheme.orange),
                    extra_info=None if abs(sigma.n)<5 else 'Fit failed',
                    title_info= poi,
                    file_name=self.get_output_file_name(f'toy_fit_pulls_{poi_index}.pdf'),
                )

        with open(self.get_output_file_name('toy_results.json'), 'w') as f:
            json.dump(out_dict, f, indent=4)
        return


class pyhf_linearity_asimov_fitTask(b2luigi.Task):
    """
    Task to summarise mle fits on toys of the asimov sample on a prepared pyhf spec.
    """
    
    queue = 'l'
    
    workspace_path = b2luigi.Parameter(default='R_D_2d_workspace.json',
                                       hashed=True,
                                       significant=False, description='relative path to pyhf workspace')

    n_toys_per_point = b2luigi.IntParameter(default=100, 
                                            description='Number of toys to generate for each test point')

    n_test_points = b2luigi.IntParameter(default=50, description='Number of test points')
    
    normalise_by_uncertainty = b2luigi.BoolParameter(default=False,
                                                     description='Whether to calculate mean or (mean-mu)/err')
    
    job_name = 'toyLinearity'


    htcondor_settings = {
        'request_cpus': 1,
        'request_memory': '16000 MB',
        '+RequestRuntime': 3600,
    }

    def requires(self):
        for part in range(self.n_test_points):
            rng = np.random.default_rng(part)
            
            yield self.clone(pyhf_toy_mle_part_fitTask,
                             workspace_path = self.workspace_path,
                             init_toy_pars = list(rng.random(6)),
                             part=part,
                             n_toys=self.n_toys_per_point,
                             store_full=True if self.normalise_by_uncertainty else False)

    def output(self):
        yield self.add_to_output('linearity_toy_results.yaml')
        yield self.add_to_output('toy_fit_linearity_0.pdf')
        yield self.add_to_output('toy_fit_linearity_1.pdf')
        yield self.add_to_output('toy_fit_linearity_2.pdf')
        yield self.add_to_output('toy_fit_linearity_3.pdf')
        yield self.add_to_output('toy_fit_linearity_4.pdf')
        yield self.add_to_output('toy_fit_linearity_5.pdf')

    def run(self):
        import zfit_utils
        
        # summarize results
        processed_results = {}
        failed_fits = 0
        for input_file in tqdm(self.get_input_file_names('toy_part_results.json')):
            with open(input_file, 'r') as f:
                in_dict = json.load(f)
                failed_fits += in_dict['failed_fits']
                
                avg = np.mean(in_dict['results']['best_fit'],axis=0).tolist()
                err = np.std(in_dict['results']['best_fit'],axis=0).tolist()
                tru = in_dict['toy_pars']
                poi = in_dict['poi']
                
                if 'fitted' not in processed_results.keys():
                    processed_results['fitted'] = [avg]
                    processed_results['error'] = [err]
                    processed_results['truth'] = [tru]
                    processed_results['poi'] = poi
                else:
                    processed_results['fitted'].extend([avg])
                    processed_results['error'].extend([err])
                    processed_results['truth'].extend([tru])

        out_dict = {
            'n_test_points': self.n_test_points,
            'n_toys_per_point': self.n_toys_per_point,
            'failed_fits': failed_fits,
            'toy_results': processed_results,
            'linearity_results': {},
        }
        

        for poi_index, poi in enumerate(processed_results['poi']):        
            truth = np.array(processed_results['truth'])[:,poi_index]
            fitted = np.array(processed_results['fitted'])[:,poi_index]
            error = np.array(processed_results['error'])[:,poi_index]

            slope, intercept = zfit_utils.zfit_linear(truth, fitted, error)

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
                color=colors[0],
                x_offset=[0],
#                 line_infos=line_infos,
#                 extra_info=fr'$\mu^{{B^{{+}}}}_{{\rm WA}}$'
#                 if poi == 'mu_wa_charged' else fr'$\mu^{{B^{{0}}}}_{{\rm WA}}$',
                title_info= poi,
                file_name=self.get_output_file_name(f'toy_fit_linearity_{poi_index}.pdf'),
            )

        with open(self.get_output_file_name('linearity_toy_results.yaml'), 'w') as f:
            yaml.dump(out_dict, f)

        return




class pyhf_toys_wrapper(b2luigi.WrapperTask):
    """
    Wrapper for submitting all the pyhf asimov toys tasks.
    """
    workspace_path = b2luigi.Parameter(default='R_D_2d_workspace.json',
                                       hashed=True,
                                       significant=False, description='relative path to pyhf workspace')

    def requires(self):
    
        yield self.clone(pyhf_toy_asimov_fitTask,
                         workspace_path = self.workspace_path,
                         n_total_toys = 10000,
                         normalise_by_uncertainty = True,
                         produce_plots = True,
                         store_full = True)
        
        yield self.clone(pyhf_linearity_asimov_fitTask,
                         workspace_path = self.workspace_path,
                         n_toys_per_point = 50,
                         n_test_points = 50,
                         normalise_by_uncertainty = False,
                         store_full = False)


if __name__ == '__main__':

    # set_b2luigi_settings('weak_annihilation_settings/weak_annihilation_settings.yaml')

    b2luigi.process(
        pyhf_toys_wrapper(workspace_path='R_D_2d_workspace.json'),
        workers=int(1e4),
        batch=True,
#         workers=1,
#         batch=False,
    )
