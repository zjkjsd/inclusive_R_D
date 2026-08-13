<!-- #region -->
<img src="https://www.belle2.org/common/logo.png" alt="Belle II Logo" width="20%">

# inclusive_R_D
Inclusive tagging measurement of R(D) at Belle II

## General procedure

 1. Use run dependent generic & signal MC16rd and proc16 data on the grid
 2. Prepare reconstruction scripts to run on the grid in the `Recon_script`
 3. Download Ntuples from the grid to `Samples`
 4. Run offline scripts, e.g. BDT_training.py, BBbkg_weights_optuna_minuit.py
 5. Use pyhf to extract yields, see in `Notebooks`; use Fit_toys to validate fitting procedures 
 6. Use plotting notebooks to check data/mc agreement at various control regions

## Offline procedure

|Procedure|Purpose|
|:---|:---|
|1. python3 2_LightGBM_tuner.py<br>or 8_XGBoost_tuner.py|Tune hyperparameters with optuna of multiclass models.|
|2. python3 3_LightGBM_training.py<br>or 8_XGBoost_training.py|Train multiclass models.|
|...|...|

## Required libraries
 1. `iminuit`
 2. `cabinetry`
 3. `optuna`
<!-- #endregion -->
