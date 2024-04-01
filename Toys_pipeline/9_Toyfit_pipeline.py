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
    
    fix_fakeD_inFit = b2luigi.BoolParameter(default=False, significant=True)
    
    test_fakeD_sideband = b2luigi.BoolParameter(default=False, significant=True)
    
    signalRegion_workspace_fakeDTest = b2luigi.Parameter(
        default='R_D_2d_workspace_withFakeDinSR_forSidebandTest_reweight.json',
        significant=False,
        hashed=True,
        description='relative path to pyhf workspace')
    
    init_toy_pars = b2luigi.ListParameter(default=[1]*7, 
                                          hashed=True,
                                          hash_function=join_list_hash_function)
    
    n_toys = b2luigi.IntParameter(default=10, significant=False,description='Number of toys to generate')

    part = b2luigi.IntParameter(
        default=0, description='Needed to be able to run N jobs (each with a part number) with n toys each')

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
        fix_par = [False,False,False,False,False,self.fix_fakeD_inFit]
        
        spec = cabinetry.workspace.load(self.workspace_path)
        model, _ = cabinetry.model_utils.model_and_data(spec)
        minos_parameters = model.config.parameters[:len(self.init_toy_pars)]
        if self.fix_fakeD_inFit:
            minos_parameters.remove('bkg_fakeD_norm')
        fakeD_par_slice = model.config.par_map['bkg_fakeD_norm']['slice']
        
        if self.test_fakeD_sideband:
            spec_test = cabinetry.workspace.load(self.signalRegion_workspace_fakeDTest)
            model_test, _ = cabinetry.model_utils.model_and_data(spec_test)
            
            # set initialisation for generating the asimov sample       
            toy_pars = model_test.config.suggested_init()
            toy_pars[:len(self.init_toy_pars)] = self.init_toy_pars
            # generate the toys:
            pdf_toy = model_test.make_pdf(pyhf.tensorlib.astensor(toy_pars))
            
        else:
            toy_pars = model.config.suggested_init()
            toy_pars[:len(self.init_toy_pars)] = self.init_toy_pars
            pdf_toy = model.make_pdf(pyhf.tensorlib.astensor(toy_pars))
        
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
                        fix_pars = fix_par,
                        # par_bounds=par_bounds,
                        goodness_of_fit=True,
                        minos=minos_parameters,
                    )

                    # save fit results
                    fit_results['best_twice_nll'].append(res.best_twice_nll)
                    fit_results['pval'].append(res.goodness_of_fit)
                    bes = res.bestfit[:len(self.init_toy_pars)].tolist()
                    err = res.uncertainty[:len(self.init_toy_pars)].tolist()
                    if self.fix_fakeD_inFit:
                        bes = bes[:fakeD_par_slice.start] + bes[fakeD_par_slice.stop:]
                        err = err[:fakeD_par_slice.start] + err[fakeD_par_slice.stop:]
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
            'poi': minos_parameters,
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
                                       significant=True,
                                       hashed=True,
                                       description='relative path to pyhf workspace')
    
    fix_fakeD_inFit = b2luigi.BoolParameter(default=False, significant=True)
    
    test_fakeD_sideband = b2luigi.BoolParameter(default=False, significant=True)
    
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
                             workspace_path = self.workspace_path,
                             fix_fakeD_inFit = self.fix_fakeD_inFit,
                             test_fakeD_sideband = self.test_fakeD_sideband,
                             init_toy_pars = self.init_toy_pars,
                             part=part,
                             n_toys=n_max_toys_per_job)
      
            
    def output(self):
        yield self.add_to_output('toy_results.json')
        if self.fix_fakeD_inFit:
            n_plot = len(self.init_toy_pars) - 1
        else:
            n_plot = len(self.init_toy_pars)
        for i in range(n_plot):
            yield self.add_to_output(f'toy_fit_pulls_{i}.pdf')

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
                                       significant=True, description='relative path to pyhf workspace')
    
    fix_fakeD_inFit = b2luigi.BoolParameter(default=False, significant=True)
    
    test_fakeD_sideband = b2luigi.BoolParameter(default=False, significant=True)
    
    n_toys_per_point = b2luigi.IntParameter(default=10, 
                                            significant=False,
                                            description='Number of toys to generate for each test point')
    
    linearity_parameter_bonds = b2luigi.ListParameter(default=[0,1], 
                                            hashed=True,
                                            hash_function=join_list_hash_function)

    n_test_parameters = b2luigi.IntParameter(default=7, significant=False,description='Number of poi')
    
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
        
        for test_point in range(self.n_test_points):
            rng = np.random.default_rng(test_point)
            toy_pars = list(rng.uniform(lin_start,lin_end,self.n_test_parameters))
            
            for part in range(int(np.ceil(self.n_toys_per_point / n_max_toys_per_job))):
                yield self.clone(pyhf_toy_mle_part_fitTask,
                                 workspace_path = self.workspace_path,
                                 fix_fakeD_inFit = self.fix_fakeD_inFit,
                                 test_fakeD_sideband = self.test_fakeD_sideband,
                                 init_toy_pars = toy_pars,
                                 part=part,
                                 n_toys=n_max_toys_per_job)

    def output(self):
        yield self.add_to_output('linearity_toy_results.yaml')
        if self.fix_fakeD_inFit:
            n_plot = self.n_test_parameters - 1
        else:
            n_plot = self.n_test_parameters
        for i in range(n_plot):
            yield self.add_to_output(f'toy_fit_linearity_{i}.pdf')

    def run(self):
        import zfit_utils
        
        spec = cabinetry.workspace.load(self.workspace_path)
        model, data = cabinetry.model_utils.model_and_data(spec)

        fakeD_par_slice = model.config.par_map['bkg_fakeD_norm']['slice']
        poi_names = model.config.parameters[:self.n_test_parameters]
        if self.fix_fakeD_inFit:
            poi_names.remove('bkg_fakeD_norm')
        
        # summarize results
        merged_results = {}
        failed_fits = 0
        for input_file in tqdm(self.get_input_file_names('toy_part_results.json')):
            with open(input_file, 'r') as f:
                in_dict = json.load(f)
                failed_fits += in_dict['failed_fits']
                
                tru = in_dict['toy_pars']
                if self.fix_fakeD_inFit:
                    tru = tru[:fakeD_par_slice.start] + tru[fakeD_par_slice.stop:]
                    
                truth = tuple(tru)  # Convert the list to a tuple to make it hashable
                fitted = in_dict['results']['best_fit']

                # Check if the truth value already exists in the merged data
                if truth in merged_results:
                    # If it exists, append the fitted value to the existing entry
                    merged_results[truth].extend(fitted)
                else:
                    # If it doesn't exist, create a new entry with the truth value and fitted value
                    merged_results[truth] = fitted
                
        processed_results = {'fitted':[],
                             'error':[],
                             'truth':[],
                             'poi': poi_names}
        for truth, fits in merged_results.items():
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
                bonds=[0,self.linearity_parameter_bonds[1]],
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
                                       description='relative path to pyhf workspace')
    fix_fakeD_inFit = b2luigi.BoolParameter(default=False, significant=False)
    test_fakeD_sideband = b2luigi.BoolParameter(default=False, significant=False)

    def requires(self):
    
        yield self.clone(pyhf_toy_asimov_fitTask,
                         workspace_path = self.workspace_path,
                         fix_fakeD_inFit = self.fix_fakeD_inFit,
                         test_fakeD_sideband = self.test_fakeD_sideband,
                         n_total_toys = 10000,
                         init_toy_pars = [0.12]*8,
                         normalise_by_uncertainty = True)
        
        yield self.clone(pyhf_linearity_asimov_fitTask,
                         workspace_path = self.workspace_path,
                         fix_fakeD_inFit = self.fix_fakeD_inFit,
                         test_fakeD_sideband = self.test_fakeD_sideband,
                         linearity_parameter_bonds = [0.01,0.2],
                         n_toys_per_point = 50,
                         n_test_parameters = 8,
                         n_test_points = 80)


if __name__ == '__main__':

    # set_b2luigi_settings('weak_annihilation_settings/weak_annihilation_settings.yaml')

    b2luigi.process(
        pyhf_toys_wrapper(workspace_path='R_D_2d_workspace_withDsideband_reweight.json',
                          fix_fakeD_inFit = True,
                          test_fakeD_sideband = True),
        workers=int(1e4),
        batch=True,
#         workers=1,
#         batch=False,
    )
