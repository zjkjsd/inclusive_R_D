<!-- #region -->
# inclusive_R_D
Inclusive tagging measurement of R(D) and R(D*) at Belle II

## General procedure

 1. Generate small run independent signal MC and test reconstruction locally in the `Samples/Signal_MC_ROEx1`
 2. Prepare reconstruction scripts to run on the grid in the `Recon_script`
 3. Download Ntuples from the grid to `Samples`
 4. Merge Ntuples by `hadd merged.root file1.root file2.root ... fileN.root`
 5. Run offline scripts 1-8. Script 3-5 are deprecated.


## Offline procedure

|Procedure|Purpose|
|:---|:---|
|1. python3 Apply_DecayHash_BCS.py|Apply decayhash/offline cuts and save `parquet` files for signal/generic_bb <br>and `root` files for continuum in `Samples`|
|2. python3 Prepare_Training_Samples.py|Use MCtruth and decayhash to select signal and particular background for BDT training<br>save output root files to `BDTs`|
|Deprecated|Deprecated|
|3. `cd BDTs`<br>`basf2_mva_merge_mc`|Merge signal and different bkg to prepare train/test samples for BDTs.|
|4. BDT_Grid_Search.py and BDT_Training.py|Grid search hyperparameters and train the first 3 BDTs (CS,DTCFake, BFake)|
|5. Add_Labels_or_combine.py<br>`basf2_mva_expert`|Add labels to all files in `BDTs` (signal, continuum, DTCFake, BFake)<br>Apply 3BDT weights and get file_applied.root for each type|
|6. Add_Labels_or_combine.py|Combine 3 BDT outputs with spectators to file_applied.root for the 4th BDT training |
|7. `basf2_mva_merge_mc`|Merge signal_applied and all bkg_applied to prepare the 4th BDT train/test|
|8. BDT_Grid_Search.py and BDT_Training.py|Train the 4th BDT|
|9. Add_Labels_or_combine.py<br>`basf2_mva_expert`|Add labels to all files in `Samples` (bb, qq)<br>Apply 3BDT weights and get file_applied.root for each type|
|10. Add_Labels_or_combine.py<br>`basf2_mva_expert`|Combine spectators and 3BDT outputs in file_applied.root <br>Apply 4th BDT weights and get file_applied_2.root for each type|
|Continue from step 2|Continue from step 2|
|3. python3 7_LightGBM_tuner.py<br>or 8_XGBoost_tuner.py|Tune hyperparameters with optuna of multiclass models.|
|4. python3 7_LightGBM_training.py<br>or 8_XGBoost_training.py|Train multiclass models.|

## Required libraries
 1. `plotly`
 2. `mplhep`-0.3.23
 3. `pyarrow`-10.0.0
 4. `iminuit`-2.18.0
 5. `cabinetry`
 6. `optuna`-3.1.0
 7. `h2o`-3.40.0.1
<!-- #endregion -->
