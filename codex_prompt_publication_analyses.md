# Codex Prompt: Publication Analyses for MeshFlowNet

## Project Context

MeshFlowNet is a GraphCast-style icosahedral mesh GNN for 15-day-ahead heat wave prediction over CONUS. It uses direct 15-day prediction (single forward pass, no autoregressive rollout) with a persistence residual output head (`y = x_t + model_residual`). The model is trained with 5-fold leave-k-years-out cross-validation over MJJAS 1981-2023 (43 years).

**Work directory:** `/blue/nessie/mostafarezaali/Teleconnection/`

**Key files:**
- `cfm_mesh_train_direct.py` — main training/export script
- `mesh_backbone.py` — MeshFlowNet GNN architecture
- `mode_dispatch.py` — loss computation, forward pass dispatch
- `icosahedral_mesh.py` — mesh construction
- `physics_losses.py` — physics loss terms

**Data files:**
- Training data: `VDM_Training_Data_Extended_v2.nc` — 6579 timesteps, 621x1405 grid, 0.04° PRISM. Target variable: `t2m_prism` (daily max temperature). 55.2% land / 44.8% ocean mask.
- Global teleconnection: `Global_Coarse_Conditions_Extended.nc` — 59 variables, 181x360, 1° global
- Hindcast outputs per fold: `hindcast_stats/hindcast_tac_stats_cvfold{0-4}_test.npz`
- Paper data per fold: `hindcast_paper_data/hindcast_sample_summary_cvfold{0-4}_test.npz`
- Paper maps per fold: `hindcast_paper_data/hindcast_selected_maps_cvfold{0-4}_test.npz`
- Monthly stats per fold: `hindcast_paper_data/hindcast_monthly_stats_cvfold{0-4}_test.npz`
- Norm stats per fold: `data_cache/norm_stats_direct15_cv5_val{V}_test{T}.npz`
- Climatology per fold: `data_cache/local_daily_climo30_direct15_cv5_val{V}_test{T}.npy` — shape (367, 621, 1405), float16, z-score units
- Land mask: derived from `t2m_prism` (NaN over ocean), 55.2% land

**Existing hindcast NPZ structure** (from `hindcast_tac_stats_cvfold{N}_test.npz`):
```
pred_sum, truth_sum, pred_sq_sum, truth_sq_sum, pred_truth_sum,
persist_sum, persist_sq_sum, persist_truth_sum, count
```
All shape (621, 1405), float64. These are per-pixel accumulated sufficient statistics for Pearson correlation. The TAC is computed as:
```python
cov = pred_truth_sum - (pred_sum * truth_sum) / count
var_pred = pred_sq_sum - pred_sum**2 / count
var_truth = truth_sq_sum - truth_sum**2 / count
corr = cov / sqrt(var_pred * var_truth)
tac = nanmean(corr[land_mask])
```

**Fold structure (cv_stride=5):**
- Fold 0: test=[1981,1986,1991,1996,2001,2006,2011,2016,2021], val=next group, train=remaining 3 groups
- Fold 1: test=[1982,1987,1992,1997,2002,2007,2012,2017,2022]
- Fold 2: test=[1983,1988,1993,1998,2003,2008,2013,2018,2023]
- Fold 3: test=[1984,1989,1994,1999,2004,2009,2014,2019]
- Fold 4: test=[1985,1990,1995,2000,2005,2010,2015,2020]

Each fold's test set is predicted by a model that never trained on those years. The 5 folds together cover all 43 years exactly once (stitched hindcast).

**Compute environment:** HiPerGator, `hpg-b200` partition, `nessie` QOS, conda env `torch_b200`. Always use `python3 -u` for unbuffered output. SLURM account: `gis4123`.

**Image dimensions:** `IMAGE_SIZE = (621, 1405)`, CONUS lat range 25-50°N, lon range -130 to -65°W, 0.04° resolution.

**Important constants:**
- LEAD_TIME = 15
- base_date for time index: 1981-05-01
- MJJAS = May 1 through Sep 30 each year
- Z-score normalization: fold-specific mean ~28.2°C, std ~6.1°C

---

## Task 1: Ensemble Predictions with Calibrated Uncertainty (CRPS)

Create a script `ensemble_crps_analysis.py` that:

1. **Loads the 5 best-TAC checkpoints** from all folds:
   - `trained_cfm_direct15_cvfold{0-4}_best_tac.pth`
   - Each checkpoint contains `ema_state_dict` (preferred) or `model_state_dict`

2. **For each fold's test set**, generates predictions from the fold's own model. This is already done in the export. The goal here is to compute CRPS from the existing single-member deterministic predictions.

3. **For a multi-seed ensemble approach** (the real deliverable):
   - Create a SLURM script that retrains each fold 5 times with different `torch.manual_seed` values (seeds 42, 137, 256, 512, 1024)
   - Each retrained model produces a separate prediction for the same test samples
   - The 5-member ensemble per fold gives a spread for CRPS computation

4. **CRPS computation** (energy form for ensemble):
   ```python
   def ensemble_crps(ensemble_preds, truth, mask):
       """
       ensemble_preds: (M, N, H, W) — M members, N samples
       truth: (N, H, W)
       mask: (H, W) binary land mask
       Returns: scalar CRPS averaged over samples and land pixels
       """
       M = ensemble_preds.shape[0]
       # E|X - y|
       abs_diff = np.abs(ensemble_preds - truth[None]).mean(axis=0)
       # E|X - X'|
       spread = 0.0
       count = 0
       for i in range(M):
           for j in range(i+1, M):
               spread += np.abs(ensemble_preds[i] - ensemble_preds[j])
               count += 1
       spread = spread / max(count, 1)
       crps = abs_diff - 0.5 * spread
       # Average over land pixels and samples
       land = mask > 0.5
       return float(np.mean(crps[:, land]))
   ```

5. **Reliability diagram**: bin forecast probabilities of exceeding the 90th percentile against observed frequencies. Use the ensemble spread to derive forecast probabilities.

6. **Output**: Save results to `ensemble_crps_results.npz` with fields: `crps_per_fold`, `crps_stitched`, `persistence_crps_stitched`, `climatology_crps_stitched`, `reliability_bins`, `reliability_observed_freq`, `reliability_forecast_freq`.

**Note:** Until the multi-seed ensemble is trained, compute a single-member CRPS (which equals MAE) from existing hindcast data as a placeholder. Structure the code so swapping in ensemble predictions later is trivial.

---

## Task 2: Per-Pixel TAC Skill Maps with Physical Interpretation

Create a script `tac_skill_maps.py` that:

1. **Loads hindcast stats from all 5 folds:**
   ```python
   for fold in range(5):
       stats = np.load(f'hindcast_stats/hindcast_tac_stats_cvfold{fold}_test.npz')
   ```

2. **Computes per-pixel stitched TAC** by summing the sufficient statistics across all 5 folds:
   ```python
   stitched = {key: sum(fold_stats[key] for fold in folds) for key in stat_keys}
   ```
   Then compute Pearson correlation from the stitched sums. This gives a (621, 1405) map of model TAC and persistence TAC.

3. **Computes skill maps:**
   - `tac_model`: per-pixel model TAC (stitched)
   - `tac_persistence`: per-pixel persistence TAC (stitched)
   - `tac_skill = tac_model - tac_persistence`: skill over persistence
   - `tac_ratio = tac_model / (tac_persistence + 1e-8)`: relative skill

4. **Generates publication-quality figures** using cartopy with CONUS Lambert Conformal projection:

   **Figure A — Model TAC map:**
   - Colormap: `RdYlBu_r`, range [-0.1, 0.3]
   - State boundaries, coastlines
   - Title: "MeshFlowNet 15-Day TAC (Stitched 5-Fold CV, 1981-2023)"

   **Figure B — Persistence TAC map:**
   - Same colormap and range
   - Title: "Persistence Baseline TAC"

   **Figure C — Skill map (model minus persistence):**
   - Colormap: `RdBu_r` centered at 0, range [-0.15, 0.15]
   - Hatching where skill > 0 (model beats persistence)
   - Title: "TAC Skill (MeshFlowNet minus Persistence)"

   **Figure D — Regional skill summary:**
   - Compute mean TAC and mean skill for NOAA climate regions (Northeast, Southeast, Midwest, Great Plains, Southwest, Northwest, West)
   - Bar chart: model TAC vs persistence TAC per region
   - Use approximate bounding boxes for regions:
     - Northeast: lat 37-48, lon -80 to -67
     - Southeast: lat 25-37, lon -90 to -75
     - Midwest: lat 37-48, lon -104 to -80
     - Great Plains: lat 25-48, lon -104 to -95
     - Southwest: lat 25-37, lon -120 to -104
     - Northwest: lat 42-50, lon -125 to -110
     - West: lat 37-42, lon -125 to -104

5. **Physical interpretation overlays:**
   - Overlay known teleconnection influence boundaries (PNA positive loading region: ~lat 35-55, lon -130 to -100)
   - Annotate regions where skill is highest/lowest with physical explanations in figure caption

6. **Output:** Save figures as 300 DPI PNGs and PDFs to `paper_figures/`. Save the per-pixel TAC arrays to `paper_figures/tac_skill_maps_data.npz`.

**Cartopy setup on HiPerGator:**
```bash
conda activate torch_b200
pip install cartopy --break-system-packages  # or conda install -c conda-forge cartopy
```

If cartopy is unavailable, fall back to matplotlib with manual CONUS lat/lon axes (no projection). The data is already on a regular lat-lon grid.

---

## Task 3: Proper Baselines (Climatology, Linear Regression, Operational S2S)

Create a script `compute_baselines.py` that implements three baselines using the same 5-fold CV splits and TAC evaluation as MeshFlowNet.

### Baseline 1: Climatology

For each fold, the climatology baseline predicts the 30-day running daily mean (already computed and saved per fold):
```python
climo = np.load(f'data_cache/local_daily_climo30_direct15_cv5_val{V}_test{T}.npy')  # (367, 621, 1405)
```

For each test sample with target DOY `d`, the climatology prediction is `climo[d]`. Compute TAC for climatology predictions vs truth across all test samples, same as the model evaluation.

The climatology TAC should be ~0.0 by construction (predicting the mean removes all anomaly signal), but compute it to verify.

Also compute climatology MSE and CRPS (MAE) as reference points.

### Baseline 2: Ridge Regression on Global Teleconnection Fields

For each fold:

1. **Extract features**: For each training sample at time `t`, extract:
   - The day-0 persistence field `x_t` (flattened land pixels, ~265k values — too large, so use PCA)
   - The 59 global fields at time `t`, spatially averaged into 8 regions:
     - Nino3.4 box (5S-5N, 170W-120W) — SST
     - Indian Ocean Dipole (IOD) boxes
     - North Pacific (30-60N, 150E-150W)
     - North Atlantic (30-60N, 80W-0)
     - Tropical Atlantic (15S-15N, 60W-0)
     - Southern Ocean (60S-30S, all longitudes)
     - Arctic (60N-90N, all longitudes)
     - Full tropics (30S-30N, all longitudes)
   - This gives 59 vars x 8 regions = 472 features, plus DOY sin/cos = 474 features

2. **Fit per-pixel ridge regression**:
   ```python
   from sklearn.linear_model import Ridge
   from sklearn.decomposition import PCA

   # Reduce persistence field to 50 PCs
   pca = PCA(n_components=50).fit(X_persist_train)
   X_persist_pcs = pca.transform(X_persist_train)

   # Full feature matrix: [persist_PCs, global_region_means, doy_sin, doy_cos]
   X_train = np.column_stack([X_persist_pcs, global_features, doy_features])

   # Fit one ridge model per pixel (only land pixels)
   for pixel_idx in land_pixel_indices:
       ridge = Ridge(alpha=1.0)
       ridge.fit(X_train, y_train[:, pixel_idx])
       y_pred[:, pixel_idx] = ridge.predict(X_test)
   ```

   Alternative (faster): fit a single multivariate ridge using the full target field projected onto 100 PCs, then reconstruct.

3. **Compute TAC** for ridge predictions using the same fold-specific climatology.

### Baseline 3: Operational S2S Reference Numbers

This is not a model you train. Instead, create a reference table from published literature:

```python
# From Pegion et al. (2019), SubX project; 
# Zhu et al. (2014), CFS/GEFS operational skill
# Specific numbers for week-3 T2max anomaly correlation over CONUS land, MJJAS

s2s_references = {
    'SubX_ensemble_mean': {
        'tac_week2': 0.15,  # approximate from published figures
        'tac_week3': 0.05,  # near zero skill at week 3
        'source': 'Pegion et al. 2019, BAMS',
    },
    'ECMWF_S2S': {
        'tac_week3': 0.08,
        'source': 'Vitart 2017, QJRMS',
    },
    'UFS_prototype': {
        'tac_week3_extratropical_land_summer': 'near zero',
        'source': 'NOAA/EMC operational verification',
    },
}
```

Search the literature for exact numbers. The key comparison: MeshFlowNet stitched TAC at day 15 vs these operational benchmarks at comparable lead times. If exact numbers are unavailable for the same metric/domain/season, note the caveats.

**Output:** Save all baseline results to `baselines_results.npz` with fields: `climo_tac_stitched`, `climo_mse_stitched`, `ridge_tac_stitched`, `ridge_tac_per_fold`, `ridge_mse_stitched`, per-pixel TAC maps for each baseline.

Create a summary comparison table as a LaTeX-formatted file `baselines_table.tex`:

```
Model               | TAC (stitched) | MSE  | CRPS
---------------------|---------------|------|------
Climatology          | X.XXX         | X.XX | X.XX
Persistence          | X.XXX         | X.XX | X.XX
Ridge Regression     | X.XXX         | X.XX | X.XX
MeshFlowNet (ours)   | X.XXX         | X.XX | X.XX
SubX ens. mean       | ~0.05         | -    | -
```

---

## Task 4: Bootstrap Significance Testing

Create a script `bootstrap_significance.py` that:

1. **Loads per-sample predictions from all 5 folds.** You need sample-level data, not just the accumulated sufficient statistics. The `hindcast_paper_data/hindcast_sample_summary_cvfold{N}_test.npz` files contain per-sample info. Check what fields are available:
   ```python
   data = np.load('hindcast_paper_data/hindcast_sample_summary_cvfold0_test.npz', allow_pickle=True)
   print(list(data.keys()))
   ```

   If per-sample spatial fields aren't stored (likely too large), re-run the export with a flag that saves per-sample TAC contributions. Alternatively, use a block bootstrap on the accumulated statistics.

2. **Block bootstrap on TAC** (preferred approach, works with accumulated stats):

   The stitched hindcast covers 43 years x ~119 days/year = ~5117 samples. Block bootstrap by year:

   ```python
   def block_bootstrap_tac(per_year_stats, land_mask, n_bootstrap=10000, rng_seed=42):
       """
       per_year_stats: dict mapping year -> {pred_sum, truth_sum, ..., count} arrays
       Returns: (model_tac_samples, persistence_tac_samples, skill_samples)
       """
       rng = np.random.default_rng(rng_seed)
       years = sorted(per_year_stats.keys())
       n_years = len(years)

       model_tacs = np.zeros(n_bootstrap)
       persist_tacs = np.zeros(n_bootstrap)

       for b in range(n_bootstrap):
           # Resample years with replacement
           boot_years = rng.choice(years, size=n_years, replace=True)

           # Sum sufficient statistics across resampled years
           boot_stats = {key: sum(per_year_stats[y][key] for y in boot_years)
                         for key in stat_keys}

           # Compute TAC from summed stats
           model_tacs[b] = compute_tac_from_stats(boot_stats, land_mask, prefix='pred')
           persist_tacs[b] = compute_tac_from_stats(boot_stats, land_mask, prefix='persist')

       skill = model_tacs - persist_tacs
       return model_tacs, persist_tacs, skill
   ```

3. **To get per-year stats**, modify the export or write a post-hoc script that re-runs inference on each fold's test set and accumulates stats per year instead of per fold. This requires:
   - Loading each fold's checkpoint
   - Running inference on the test set
   - Grouping samples by year
   - Accumulating sufficient statistics per year

   Create a script `export_per_year_stats.py` that does this. It should reuse the existing `_accumulate_tac_stats` function from `cfm_mesh_train_direct.py`. Output: `hindcast_stats/per_year_tac_stats.npz` containing a dict with 43 entries (one per year), each holding the 9 sufficient statistic arrays.

4. **Statistical tests to report:**

   ```python
   # 95% CI on model TAC
   model_ci = np.percentile(model_tacs, [2.5, 97.5])

   # 95% CI on persistence TAC
   persist_ci = np.percentile(persist_tacs, [2.5, 97.5])

   # 95% CI on skill (model - persistence)
   skill_ci = np.percentile(skill, [2.5, 97.5])

   # p-value: fraction of bootstrap samples where skill <= 0
   p_value = np.mean(skill <= 0)

   # Effect size: mean skill / std skill
   cohens_d = np.mean(skill) / (np.std(skill) + 1e-8)
   ```

5. **Per-pixel significance map:**
   Run the same block bootstrap per pixel (or for computational efficiency, per 4x4 pixel block). Generate a map showing where model TAC significantly exceeds persistence TAC at p < 0.05 (Bonferroni or FDR corrected for multiple comparisons).

   ```python
   from statsmodels.stats.multitest import multipletests

   # pixel_p_values: (621, 1405) array of p-values from bootstrap
   reject, corrected_p, _, _ = multipletests(
       pixel_p_values[land_mask].ravel(),
       alpha=0.05,
       method='fdr_bh'  # Benjamini-Hochberg FDR
   )
   ```

6. **Output figures:**

   **Figure A — Bootstrap distribution:**
   - Histogram of `skill` (model TAC minus persistence TAC) from 10,000 bootstrap samples
   - Vertical line at 0 (no skill)
   - Shaded 95% CI
   - Annotate with p-value

   **Figure B — Significance map:**
   - Per-pixel: colored where model significantly beats persistence (FDR-corrected p < 0.05)
   - Gray where not significant
   - Overlay state boundaries

7. **Output data:** Save to `bootstrap_results.npz` with fields: `model_tac_samples`, `persist_tac_samples`, `skill_samples`, `model_ci_95`, `persist_ci_95`, `skill_ci_95`, `p_value`, `cohens_d`, `pixel_significance_map`.

---

## Implementation Notes

- All scripts should be standalone Python files runnable on HiPerGator with `python3 -u script.py`.
- Use `#!/usr/bin/env python3` shebang.
- Import paths: scripts run from `/blue/nessie/mostafarezaali/Teleconnection/`.
- For any script that needs the model or dataset classes, add the work directory to sys.path and import from `cfm_mesh_train_direct`.
- Use `argparse` for any configurable parameters.
- Save all figures to `paper_figures/` directory (create if not exists).
- Save all numeric results as `.npz` files.
- Print a summary table to stdout at the end of each script.
- For SLURM scripts, use: partition `hpg-b200`, QOS `nessie`, account `gis4123`, conda env `torch_b200`.
- Always set `NCCL_NET_MERGE_LEVEL=LOC`, `NCCL_IB_DISABLE=1`, `NCCL_NVLS_ENABLE=0` in SLURM scripts.
- Use `matplotlib.use('Agg')` at the top of every plotting script (no display on HPC).
- Target: 300 DPI PNG + PDF for all figures.

## Execution Order

1. `tac_skill_maps.py` — runs immediately on existing hindcast stats (no GPU needed)
2. `compute_baselines.py` — climatology baseline runs immediately; ridge regression needs CPU time (~1 hour)
3. `export_per_year_stats.py` — needs GPU, re-runs inference on all 5 folds (~30 min per fold on 1 GPU)
4. `bootstrap_significance.py` — runs on output of step 3 (CPU only, ~10 min)
5. `ensemble_crps_analysis.py` — placeholder version runs immediately; full ensemble needs retraining (5 seeds x 5 folds = 25 training runs)
