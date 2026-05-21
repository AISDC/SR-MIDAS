# SR-MIDAS

Super-resolution workflow for [MIDAS](https://github.com/marinerhemant/MIDAS) Far-Field High-Energy X-ray Diffraction (FF-HEDM) analysis.

SR-MIDAS provides pipelines for - (a) using pre-trained CNN models, and (b) training new super-resolution models - to enhance the spatial resolution of 2D diffraction patches extracted from FF-HEDM datasets. It integrates with the [MIDAS](https://github.com/marinerhemant/MIDAS) FF-HEDM grain reconstruction routine for improved overlapping peak detection and more precise peak localization for FF-HEDM analysis.

**v0.1.3** — End-to-end all-GPU per-frame pipeline: preprocessing (dark/flood/mask/ring-threshold), connected-component labeling, patch extraction, and the SRx2→4→8 cascade now stay on-device through to peak fitting, eliminating the per-stage CPU round-trips of v0.1.2. The sub-pixel `shiftYpx`/`shiftZpx` correction that maps SR-grid coords to MIDAS pixel-center convention is auto-computed from `srfac` — no `shift_YZ_pos` config entry needed.

**v0.1.2** — Introduced GPU-accelerated peak fitting via batched pseudo-Voigt with `torch.compile` JIT on CUDA, replacing the per-patch `scipy.curve_fit` CPU path and giving ~1000× speedup at the fitting stage. Enabled by default; falls back to CPU automatically when no CUDA device is available.

---

### Publication

Artcile: [Super-resolution model for overlapping peak detection and improved spatial resolution in high-energy diffraction microscopy](https://doi.org/10.1016/j.matdes.2026.115917)

DOI: https://doi.org/10.1016/j.matdes.2026.115917

Authors: Dishant Beniwal, Jun-Sang Park, Peter Kenesei, Rajkumar Kettimuthu, Antonino Miceli, Hemant Sharma

---

## Table of Contents

- [Installation](#installation)
- [Workflow Overview](#workflow-overview)
- [Add SR-MIDAS to MIDAS `ff_MIDAS.py`](#adding-sr-midas-worflow-to-midas-ff-hedm-workflow)
- [Running MIDAS `ff_MIDAS.py` with super-resolution enabled](#running-midas-ff-hedm-analysis-with-super-resolution-workflow-enabled)
- [Command-Line Interface](#command-line-interface)
  - [Run SR Processing on MIDAS Data](#1-run-sr-processing-on-midas-data)
  - [Create Peakbank](#2-create-peakbank)
  - [Create Patchstore](#3-create-patchstore)
  - [Train a Model](#4-train-a-model)
  - [Hyperparameter Optimization](#5-hyperparameter-optimization)
  - [Predict on a Patchstore](#6-predict-on-a-patchstore)
  - [Create Predicted Patchstore](#7-create-predicted-patchstore)
- [Python API](#python-api)
- [Pretrained Models](#pretrained-models)
- [SR Config File](#sr-config-file)
- [Architecture String Format](#architecture-string-format)
- [GPU Peak Fitting](#gpu-peak-fitting)

---

## Installation

```bash
pip install sr-midas
```

For hyperparameter optimization support (requires [Optuna](https://optuna.org/)):

```bash
pip install "sr-midas[optuna]"
```

For development:

```bash
git clone <repo>
cd SR-MIDAS
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.4.

---

## Workflow Overview

The library provides two independent use paths:

**Path A — Use pretrained models directly on MIDAS data**

```
MIDAS .zip  →  sr-midas-process  →  per-frame peak CSV files
```

**Path B — Train your own models from MIDAS data**

```
MIDAS .zip
    ↓
sr-midas-create-peakbank      # extract fitted peaks into a peakbank CSV
    ↓
sr-midas-create-patchstore    # synthesize multi-resolution patch pairs for training
    ↓
sr-midas-train                # train a CNNSR model
    ↓
sr-midas-predict              # evaluate predictions on a patchstore
    ↓
sr-midas-process              # apply trained model to full MIDAS data
```

---

## Adding SR-MIDAS worflow to MIDAS FF-HEDM workflow
NOTE: This requires modification of ff_MIDAS.py file and should be carried out very carefully.

In `ff_MIDAS.py` file, add following additional arguments to argument paser inside the `main()` function:

```Python
    # Adding additional arguments for SR-MIDAS workflow --------------------------

    parser.add_argument("-runSR", type=int, required=False, default=0, 
                        help="(default=0) To enable super-resolution workflow, set to 1.")

    parser.add_argument("-srfac", type=int, required=False, default=8, 
                        help="(default 8) Super resolution factor. Options: [2, 4, 8]")

    parser.add_argument("-SRconfig_path", type=str, required=False, default="auto", 
                        help="(default 'auto') Full path to the super resolution configuration (.json) file.\
                        If not provided, the default configuraton built into the SR-MIDAS will be used")

    parser.add_argument("-saveSRpatches", type=int, required=False, default=0, 
                        help="(default 0) To save predicted SR patches, set to 1.")

    parser.add_argument("-saveFrameGoodCoords", type=int, required=False, default=0, 
                        help="(default 0) To save GoodCoords map for each frame, set to 1.\
                            The goodCoords map filters the pixels that belong to rings")

    # --------------------------
```

Create following additional variables from arguments:

```Python
    # Additional variables for SR-MIDAS workflow --------------------------
    runSR = args.runSR
    srfac = args.srfac
    SRconfig_path = args.SRconfig_path
    saveSRpatches = args.saveSRpatches
    saveFrameGoodCoords = args.saveFrameGoodCoords
    # --------------------------

```

Within the `main()` function, pass following additional variables when calling `process_layer(....)` for both the batch mode and standard mode:

```Python
    try:
        process_layer(...,
            runSR=runSR,
            srfac=srfac,
            SRconfig_path=SRconfig_path,
            saveSRpatches=saveSRpatches,
            saveFrameGoodCoords=saveFrameGoodCoords
        )
```

The last step is to modify the `process_layer()` function to create a route for super-resolution workflow.
- modify the function definition to include the additional variables
- modify the peak search section to include an additional pathway

```Python
def process_layer(.....,
                  runSR: int, srfac: int, SRconfig_path: str, saveSRpatches: int, saveFrameGoodCoords: int,
                  grains_file: str = '') -> None:

    ....
    ....
    # Process peaks if required
    if provide_input_all == 0:
        if _should_run('peak_search'):
            if do_peak_search == 1:
                ...
                try:
                    ...
                except Exception as e:
                    raise ...

            # Add the modification route for SR-MIDAS workflow --------------------------
            elif (do_peak_search == 0) and (runSR == 1):

                logger.info(f"Running super resolution. Time till now: {time.time() - t0}")
                from sr_midas.pipeline.sr_process import run_sr_process
                
                sr_process_args = {}
                sr_process_args['midasZarrDir'] = result_dir
                sr_process_args['srfac'] = srfac

                if SRconfig_path != 'auto':
                    # 'SRconfig_path' variable will be passed only if it is not the default value
                    sr_process_args['SRconfig_path'] = SRconfig_path 

                sr_process_args['saveSRpatches'] = saveSRpatches
                sr_process_args['saveFrameGoodCoords'] = saveFrameGoodCoords

                try:
                    run_sr_process(**sr_process_args)
                    ph5.mark('peak_search')
                except Exception as e:
                    logger.error(f"Failed to execute SR-MIDAS workflow: {e}")
                    return None
            # --------------------------
            
            else:
                ...

```

## Running MIDAS FF-HEDM analysis with super-resolution workflow enabled

If your MIDAS installation is compatible with running SR-MIDAS, the super-resolution workflow can be enabled by declaring two additional arguments in the command line for MIDAS FF-HEDM analysis.
`- doPeakSearch 0` : This disables the MIDAS peak fitting.
`- runSR 1` : This triggers the SR workflow and hands over the peak fitting task to SR-MIDAS pipeline 

```bash
python ../MIDAS/FF-HEDM/workflows/ff_MIDAS.py \
-paramFN ps.txt \
-nCPUs 50 \
-resultFolder MidasRecon \
-numFrameChunks 100 \
-preProcThresh 60 \
-dataFN ../data.h5 \
-doPeakSearch 0 \
-runSR 1 \
```

The above command will SR worflow with default in-built configuration in SR-MIDAS. You can pass optional arguments to control your SR workflow:

| Key | Description | Default | Options |
|---|---|---|---|
| `-srfac` | super-resolution factor | 8 | {2, 4, 8} |
| `-SRconfig_path` | Path to .json configuration file for super-resolution workflow | "auto" | Full path to .json SR configuration file |
| `-saveSRpatches` | Controls if predicted SR patches are saved (if =1) | 0 | {0, 1} |
| `-saveFrameGoodCoords` | Controls if GoodCoords map is saved for each frame (if =1) | 0 | {0, 1} |
| `-use_gpu` | Controls if GPU-accelerated peak fitting is used (if =1, requires CUDA) | 1 | {0, 1} |

Note: Enable 'saveSRpatches' and 'saveFrameGoodCoords' only if you want to debug the SR workflow since these will take significant disk space.

## Command-Line Interface

### 1. Run SR Processing on MIDAS Data

Applies the full super-resolution pipeline to a MIDAS experiment directory. Reads detector frames from the `.MIDAS.zip` file, detects diffraction spots, applies cascaded SR, fits peak positions, and writes per-frame peak CSV files back to the MIDAS `Temp/` directory.

```bash
sr-midas-process -midasZarrDir /path/to/analysis_dir/
```

With a custom SR config pointing to your own trained models:

```bash
sr-midas-process \
    -midasZarrDir /path/to/analysis_dir/ \
    -srfac 8 \
    -SRconfig /path/to/sr_config.json
```

| Argument | Default | Description |
|---|---|---|
| `-midasZarrDir` | required | Directory containing the `.MIDAS.zip` zarr file |
| `-srfac` | `8` | Super-resolution factor: `2`, `4`, or `8` |
| `-SRconfig` | bundled `cnnsr_sr_config.json` | Path to SR config `.json` or `.txt` file |
| `-saveSRpatches` | `1` | Save SR patch arrays to disk (`1`=yes, `0`=no) |
| `-saveFrameGoodCoords` | `1` | Save per-frame coordinate maps (`1`=yes, `0`=no) |
| `-use_gpu` | `1` | Use GPU-accelerated peak fitting when CUDA is available (`1`=yes, `0`=no) |

When `-SRconfig` is not provided, the bundled `cnnsr_sr_config.json` is used automatically, which points to the pretrained cascaded models included in the package.

**Output:** Per-frame peak CSV files written to `{midasZarrDir}/Temp/`, in the same format as standard MIDAS `*_PS.csv` files. A copy of the resolved SR config is saved to `{midasZarrDir}/SR_out/sr_config.json`.

---

### 2. Create Peakbank

Extracts MIDAS-fitted peaks from one or more analysis directories into a filtered CSV (the *peakbank*).

```bash
sr-midas-create-peakbank -config in_create_peakbank.json
```

The JSON config file specifies all parameters:

```json
{
    "midas_dir": [
        "/path/to/analysis_dir_1",
        "/path/to/analysis_dir_2"
    ],
    "peakbank_savedir": "out_peakbank/",
    "peakbank_savename": "df_peakbank.csv",
    "cvsz": 20,
    "srfac": 1,
    "I_thresh": 99.0,
    "peak_recon_err_threshold": 0.5,
    "save_exp_patches": "False",
    "dir_ignore": [".DS_Store"],
    "save_frame_gen": "False"
}
```

| Key | Description |
|---|---|
| `midas_dir` | List of MIDAS analysis directories (each must contain a `.MIDAS.zip` and a `Temp/` folder with `*_PS.csv` files) |
| `peakbank_savedir` | Output directory |
| `peakbank_savename` | Output CSV filename |
| `cvsz` | Canvas size (pixels) for patch reconstruction |
| `srfac` | SR factor for peak reconstruction (typically `1`) |
| `I_thresh` | Intensity percentile threshold; peaks below this are excluded |
| `peak_recon_err_threshold` | Max allowed peak reconstruction error |
| `save_exp_patches` | Save experimental patches alongside CSV (`"True"` / `"False"`) |
| `save_frame_gen` | Save synthesized frames for inspection (`"True"` / `"False"`) |

**Output:** `{peakbank_savedir}/{peakbank_savename}` — a CSV with one row per accepted peak, containing columns: `Ypx`, `Zpx`, `Rpx`, `EtaDeg`, `IMax`, `IntegInt`, `peak_recon_err`, and fitted peak shape parameters.

---

### 3. Create Patchstore

Synthesizes a set of multi-resolution patch pairs (at x1, x2, x4, x8) from the peakbank for training and evaluation.

```bash
sr-midas-create-patchstore \
    -peakbankPath out_peakbank/df_peakbank.csv \
    -savedir out_patchstore/ \
    -savename patchstore.h5 \
    -cvsz 20 \
    -srfac_list 1 2 4 8 \
    -n_patches 60000 \
    -patch_size 20 \
    -I_thresh 99.0
```

A JSON config file can also be provided via `-config` to override or supply any of the above arguments.

**Output:** An HDF5 file (`.h5`) with the following structure:

```
patchstore.h5
├── patchArr/
│   ├── SRx1    (N, 1, 20, 20)   ← native resolution patches
│   ├── SRx2    (N, 1, 40, 40)
│   ├── SRx4    (N, 1, 80, 80)
│   └── SRx8    (N, 1, 160, 160) ← highest resolution
├── patchInfo/                    ← per-patch metadata (nPeaks, ISum, position)
├── peaksLocInPatch/              ← peak pixel locations within each patch
├── peaksParameters/              ← fitted peak shape parameters
└── pstCreationArgs/              ← arguments used to create this file
```

---

### 4. Train a Model

Trains a CNNSR model to predict high-resolution patches from low-resolution inputs.

```bash
sr-midas-train cnnsr \
    -expName myexp \
    -pst out_patchstore/patchstore.h5 \
    -inSRx 1 \
    -outSRx 8 \
    -arch "64-5-r_32-5-r_1-5-s" \
    -lr 1e-4 \
    -mbsz 64 \
    -maxItr 1000 \
    -ecVal 0.01 \
    -ecItr 50 \
    -trainOutDir out_models/
```

| Argument | Required | Description |
|---|---|---|
| `cnnsr` | — | Model type subcommand |
| `-expName` | Yes | Experiment name (becomes output directory name) |
| `-pst` | Yes | Path to patchstore `.h5` file |
| `-inSRx` | Yes | Input patch SR factor (e.g. `1`) |
| `-outSRx` | Yes | Target output SR factor (e.g. `8`) |
| `-arch` | Yes | CNN architecture string (see [Architecture String Format](#architecture-string-format)) |
| `-lr` | Yes | Learning rate |
| `-mbsz` | Yes | Mini-batch size |
| `-maxItr` | Yes | Total training epochs |
| `-ecVal` | Yes | Loss threshold for convergence check |
| `-ecItr` | Yes | Number of initial epochs used for convergence check |
| `-trainOutDir` | Yes | Root output directory |
| `-lossF` | No | Loss function: `mse` (default) or `mae` |
| `-trainFrac` | No | Fraction of data for training (default: `0.8`) |
| `-nwork` | No | DataLoader workers (default: `1`) |
| `-useRch` | No | Include Radius channel as input (default: `false`) |
| `-useEtach` | No | Include Eta channel as input (default: `false`) |
| `-inPstPath` | No | Separate input patchstore (if different from `-pst`) |
| `-outPstPath` | No | Separate target patchstore (if different from `-pst`) |
| `-loadChkpt` | No | Path to checkpoint to initialize weights from |

**Output:** `{trainOutDir}/{expName}-itrOut/`
- `mod-it{epoch}.pth` — model checkpoint saved every epoch
- `_train_args.json` — all training arguments
- `_train_log.log` — per-epoch loss and L2-norm statistics

**Cascaded training example** (three separate models for x1→x2→x4→x8):

```bash
# Stage 1: x1 → x2
sr-midas-train cnnsr -expName x1_x2 -pst patchstore.h5 \
    -inSRx 1 -outSRx 2 -arch "64-5-r_32-5-r_1-5-s" \
    -lr 1e-4 -mbsz 64 -maxItr 5000 -ecVal 0.01 -ecItr 50 -trainOutDir out_models/

# Stage 2: x2 → x4
sr-midas-train cnnsr -expName x2_x4 -pst patchstore.h5 \
    -inSRx 2 -outSRx 4 -arch "64-5-r_32-5-r_1-5-s" \
    -lr 1e-4 -mbsz 64 -maxItr 1000 -ecVal 0.01 -ecItr 50 -trainOutDir out_models/

# Stage 3: x4 → x8
sr-midas-train cnnsr -expName x4_x8 -pst patchstore.h5 \
    -inSRx 4 -outSRx 8 -arch "64-5-r_32-5-r_1-5-s" \
    -lr 1e-4 -mbsz 64 -maxItr 1000 -ecVal 0.01 -ecItr 50 -trainOutDir out_models/
```

---

### 5. Hyperparameter Optimization

Searches for optimal CNNSR architecture and training hyperparameters using Optuna.

**Requires:** `pip install "sr-midas[optuna]"`

```bash
sr-midas-hp-optimize cnnsr \
    -pst out_patchstore/patchstore.h5 \
    -inSRx 1 \
    -outSRx 8 \
    -n_trials 50 \
    -n_itrs 20 \
    -output_base_dir optuna_results/
```

| Argument | Required | Description |
|---|---|---|
| `cnnsr` | — | Model type subcommand |
| `-pst` | Yes | Path to patchstore `.h5` file |
| `-inSRx` | Yes | Input SR factor |
| `-outSRx` | Yes | Output SR factor |
| `-n_trials` | Yes | Number of Optuna trials |
| `-n_itrs` | Yes | Epochs per trial |
| `-trainFrac` | No | Training fraction (default: `0.8`) |
| `-patience` | No | Early stopping patience (default: `5`) |
| `-study_name` | No | Optuna study name (default: `cnnsr_optimization`) |
| `-output_base_dir` | No | Results directory (default: `optuna_results`) |

---

### 6. Predict on a Patchstore

Runs a trained model on patches in a patchstore and saves predictions as `.npy` files. Useful for evaluating model accuracy against known ground-truth patches.

**Single-model mode** (x1 → x8 directly):

```bash
sr-midas-predict cnnsr \
    -pstPath out_patchstore/patchstore.h5 \
    -inSRx 1 \
    -outSRx 8 \
    -modDir out_models/myexp-itrOut \
    -modItr 999 \
    -saveDir out_predictions/
```

**Cascaded mode** (x1→x2→x4→x8 via three models):

```bash
sr-midas-predict cnnsr \
    -pstPath out_patchstore/patchstore.h5 \
    -inSRx 1 \
    -outSRx 8 \
    -cascade \
    -x2ModDir out_models/x1_x2-itrOut -x2ModItr 4975 \
    -x4ModDir out_models/x2_x4-itrOut -x4ModItr 999 \
    -x8ModDir out_models/x4_x8-itrOut -x8ModItr 999 \
    -saveDir out_predictions/
```

| Argument | Description |
|---|---|
| `-pstPath` | Input patchstore `.h5` |
| `-inSRx` | SR factor of input patches |
| `-outSRx` | Target SR factor |
| `-saveDir` | Output directory |
| `-saveName` | Output filename (default: `predictions.npy`) |
| `-bsz` | Inference batch size (default: `200`) |
| `-cascade` | Flag: use cascaded prediction |
| `-modDir` / `-modItr` | Single-model directory and checkpoint iteration |
| `-x2/x4/x8ModDir` / `-x2/x4/x8ModItr` | Per-stage directory and iteration (cascade mode) |

**Output:** `.npy` file(s) with predicted patches. In cascade mode, three files are saved: `{saveName}_SRx2.npy`, `{saveName}_SRx4.npy`, `{saveName}_SRx8.npy`.

---

### 7. Create Predicted Patchstore

Runs a trained model on an existing patchstore and saves the predictions as a new HDF5 patchstore. Useful for cascaded training workflows where the next-stage model trains on predicted (rather than ground-truth) inputs.

```bash
sr-midas-create-pred-pst \
    -pstPath out_patchstore/patchstore.h5 \
    -saveDir out_pred_patchstore/ \
    -saveName pred_pst_x2.h5 \
    -trainedModDir out_models/x1_x2-itrOut \
    -trainedModItr 4975 \
    -srfacIn 1 \
    -srfacOut 2 \
    -bsz 500
```

---

## Python API

Key functions are importable directly from the `sr_midas` package:

```python
import torch
import sr_midas
from sr_midas.pipeline.sr_process import run_sr_process

# Run the full SR pipeline (GPU peak fitting enabled by default)
run_sr_process(
    midasZarrDir="/path/to/analysis_dir/",
    srfac=8,
    use_gpu=1,   # set to 0 to force CPU-only peak fitting
)

# Load a trained CNNSR model
mod, mod_args, channels = sr_midas.load_trained_CNNSR(
    mod_dir="/path/to/model-itrOut",
    mod_itr=999,
    torch_devs=torch.device("cpu"),
)

# Load a patchstore
patch_arr, df_info, peaks_loc, peak_params, creation_args = \
    sr_midas.load_patchstore_h5data("patchstore.h5")

# Predict with a single x1→x8 model
from sr_midas import predict_CNNSR_singleMod
X = patch_arr["SRx1"]   # shape (N, 1, 20, 20)
mods_to_use = {"SRx8": {"mod_dir": "/path/to/model-itrOut", "mod_itr": 999}}
predictions = predict_CNNSR_singleMod(X, mods_to_use, batch_size=200)
# predictions: shape (N, 1, 160, 160), values in [0, 1]

# Predict with cascaded models (x1→x2→x4→x8)
from sr_midas import predict_CNNSR
mods_to_use = {
    "SRx2": {"mod_dir": "/path/to/x1_x2-itrOut", "mod_itr": 4975},
    "SRx4": {"mod_dir": "/path/to/x2_x4-itrOut", "mod_itr": 999},
    "SRx8": {"mod_dir": "/path/to/x4_x8-itrOut", "mod_itr": 999},
}
SRx2, SRx4, SRx8 = predict_CNNSR(X, mods_to_use, batch_size=200)

# Access bundled pretrained model paths
from sr_midas import get_model_dir, MODEL_NAMES, MODEL_ITRS
mod_dir = get_model_dir("x1_x8")   # returns absolute Path to bundled model directory
itr     = MODEL_ITRS["x1_x8"]      # 1300
```

---

## Pretrained Models

Four pretrained CNNSR models are bundled with the package:

| Key | Mapping | Checkpoint iteration | Use case |
|---|---|---|---|
| `x1_x2` | x1 → x2 | 4975 | Stage 1 of cascaded SR |
| `x2_x4` | x2 → x4 | 999 | Stage 2 of cascaded SR |
| `x4_x8` | x4 → x8 | 999 | Stage 3 of cascaded SR |
| `x1_x8` | x1 → x8 | 1300 | Single-model direct SR |

`sr-midas-process` uses the cascaded pretrained models (`x1_x2`, `x2_x4`, `x4_x8`) by default when no `-SRconfig` is provided.

---

## SR Config File

`sr-midas-process` is controlled by an SR config JSON file. When no file is provided, the bundled `cnnsr_sr_config.json` is used. To use your own trained models, copy the template below and pass it via `-SRconfig`:

```json
{
    "minEta": 6,
    "minPxCount": 2,
    "skipFitIfExists": "yes",
    "fitPeakShapePV": "no",
    "R_deviation": 10.0,
    "lrsz": 20,
    "edge_bound_cutoff_fac": 2.5,
    "batch_size": 400,
    "mods_to_use": {
        "SRx2": {"mod_dir": "/full/path/to/x1_x2-itrOut", "mod_itr": 4975},
        "SRx4": {"mod_dir": "/full/path/to/x2_x4-itrOut", "mod_itr": 999},
        "SRx8": {"mod_dir": "/full/path/to/x4_x8-itrOut", "mod_itr": 999}
    },
    "spot_find_args": {
        "threshold": 30,
        "patch_size": 20
    },
    "peak_find_args": {
        "min_d":             {"SRx1": 2,  "SRx2": 2,   "SRx4": 4,  "SRx8": 5},
        "thresh_rel":        {"SRx1": 0.2,"SRx2": 0.2, "SRx4": 0.2,"SRx8": 0.1},
        "gauss_filter_sigma":{"SRx1": 0,  "SRx2": 0,   "SRx4": 0,  "SRx8": 0},
        "median_filter_size":{"SRx1": 1,  "SRx2": 1,   "SRx4": 1,  "SRx8": 1},
        "peak_crop_size":    {"SRx1": 2,  "SRx2": 8,   "SRx4": 10, "SRx8": 20},
        "pvfit_int_thresh":  {"SRx1": 60, "SRx2": 20,  "SRx4": 15, "SRx8": 10}
    }
}
```

> **Sub-pixel shift correction** *(v0.1.3+)*: SR-MIDAS's internal SR grid places the leading edge of native pixel `Y00` at grid position `Y00 + 0`, while MIDAS treats integer pixel index `Y00` as the pixel center. The compensating shift `-(srfac - 1) / (2 * srfac)` (e.g. `-0.4375` at SRx8) is now computed automatically from `srfac` inside `sr_midas.physics.coord_transform.pixel_center_shift_for_srfac` and applied as the last step of fitting — no `shift_YZ_pos` block is needed in the config. Any `shift_YZ_pos` entries left in user configs from v0.1.2 or earlier are ignored.

| Key | Description |
|---|---|
| `mods_to_use` | Full paths and checkpoint iterations for each SR stage model |
| `lrsz` | Low-resolution patch size (must match the patchstore used for training) |
| `batch_size` | Number of patches processed per inference batch |
| `skipFitIfExists` | Skip re-fitting frames whose output CSV already exists (`"yes"` / `"no"`) |
| `fitPeakShapePV` | Fit peaks with pseudo-Voigt shape (`"yes"`) or center-of-mass only (`"no"`) |
| `R_deviation` | Max deviation (pixels) from the expected ring radius for a valid peak |
| `spot_find_args` | Intensity threshold and patch size for initial spot detection |
| `peak_find_args` | Per-SR-level parameters for peak finding (min distance, relative threshold, filter sizes, crop size) |

---

## Architecture String Format

The `-arch` argument for `sr-midas-train` encodes the CNN layer stack as an underscore-separated list of `{channels}-{kernel_size}-{activation}` descriptors.

```
"64-5-r_32-5-r_1-5-s"
 ^^^^^^^^^  ^^^^^^^^^  ^^^^^^^^^
 layer 1    layer 2    layer 3
 64ch,5×5   32ch,5×5   1ch,5×5
 ReLU       ReLU       Sigmoid
```

Supported activations: `r` (ReLU), `s` (Sigmoid), `lr` (LeakyReLU), `t` (Tanh).

All convolutions use same-padding, so spatial dimensions are preserved throughout. The final layer should output 1 channel with Sigmoid activation to produce normalized predictions in [0, 1].

| String | Layers | Notes |
|---|---|---|
| `"64-5-r_32-5-r_1-5-s"` | 3 layers | Standard SRCNN-style architecture |
| `"64-3-r_64-3-r_32-3-r_1-3-s"` | 4 layers | Deeper network with smaller receptive field |
| `"8-3-r_1-3-s"` | 2 layers | Minimal network for testing |

---

## GPU Peak Fitting

*Added in v0.1.2 · End-to-end all-GPU pipeline in v0.1.3*

SR-MIDAS includes GPU-accelerated 2D pseudo-Voigt peak fitting as an alternative to the per-patch `scipy.curve_fit` (TRF) CPU path. When enabled, all patches in a detector frame are fitted simultaneously on the GPU using a batched Adam optimizer with `torch.compile` JIT compilation.

### How it works

1. **GPU peak detection** — peaks are detected using a fully vectorized max-pooling approach with plateau suppression (equivalent to `skimage.feature.peak_local_max`)
2. **Batched pseudo-Voigt fitting** — a differentiable 2D pseudo-Voigt model is evaluated for all patches in parallel, with parameter bounds enforced via sigmoid projection
3. **torch.compile acceleration** — the fitting kernel is JIT-compiled on first use (one-time warmup cost), providing significant speedup for subsequent frames

### v0.1.3 runtime changes — end-to-end all-GPU per-frame pipeline

In v0.1.2 only the peak-fitting stage ran on GPU; preprocessing, connected-component labeling, patch extraction, and the SR cascade still bounced through CPU/numpy between stages. v0.1.3 keeps each frame on-device from raw read to final fit:

- **`pipeline/_gpu_preprocess.py`** *(new)* — per-frame preprocessing on GPU: image transforms, dark subtraction, flood division, beam-current scaling, bad-pixel zeroing, mask, ring thresholding. Static correction arrays (dark, flood, mask, ring-Nr map) are uploaded once at pipeline startup via `upload_correction_frames` instead of being re-transferred every frame. Also provides `gpu_ccl` — batched connected-component labeling via iterative 3×3 min-pooling — and `extract_patches_gpu`.
- **`pipeline/_gpu_sr_cascade.py`** *(new)* — cascaded SRx2 → SRx4 → SRx8 CNN inference with no per-stage CPU round-trip. `np.repeat` upsampling is replaced with `torch.repeat_interleave`, per-stage max-normalization runs on device, and the SRx8 intensity rescaling is applied with the patch sums kept as a tensor. Output is handed straight to the GPU peak fitter.
- **`pipeline/_gpu_peakfit.py`** — extended to consume the cascade output tensor directly. Coordinates that were previously list/numpy at the call site are auto-coerced.
- **`pipeline/sr_process.py`** — when `use_gpu=1` and CUDA is available, the per-frame path runs `preprocess_frame_gpu → gpu_ccl → extract_patches_gpu → cascade_sr_gpu → gpu_fit_frame_patches` with `torch.cuda.synchronize()` between stages for accurate timing; the legacy CPU path (`use_gpu=0` or no CUDA) is unchanged. The sub-pixel `shiftYpx`/`shiftZpx` correction is now auto-computed from `srfac` (see SR Config File above) instead of read from the config.

Numerics: bit-for-bit identical to the v1 CPU-bouncing cascade aside from floating-point ordering differences inside the CNN (PyTorch is free to pick different reduction orders on GPU vs CPU). With CUDA deterministic mode disabled (the default in `sr_process.py`), small last-decimal differences are expected; YCen/ZCen agree with the MIDAS CPU reference to 5 decimal places on the bundled test frame.

### Usage

GPU peak fitting is **enabled by default** (`use_gpu=1`). No additional configuration is required — it uses the same `peak_find_args` parameters from the SR config file.

**Behavior:**
- If `use_gpu=1` and a CUDA GPU is available → all-GPU per-frame pipeline is used
- If `use_gpu=0` → CPU peak fitting is used (original `scipy.curve_fit` path)
- If no CUDA GPU is available → falls back to CPU with a warning

To disable GPU fitting explicitly:

```bash
sr-midas-process -midasZarrDir /path/to/analysis_dir/ -use_gpu 0
```

Or in Python:

```python
from sr_midas.pipeline.sr_process import run_sr_process
run_sr_process(midasZarrDir="/path/to/analysis_dir/", use_gpu=0)
```

### Requirements

- PyTorch ≥ 2.4 with CUDA support (no additional dependencies beyond what SR-MIDAS already requires)
- A CUDA-capable GPU
