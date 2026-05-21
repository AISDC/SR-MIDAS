"""Automated end-to-end CNNSR training workflow.

Takes a MIDAS zip directory as input and produces trained cascaded CNNSR
models (x2, x4, x8) plus a ready-to-use sr_config.json for sr_process.

Pipeline:
    MIDAS zip -> peakbank -> patchstore -> train x2 -> pred-pst x2
    -> train x4 -> pred-pst x4 -> train x8 -> sr_config.json
"""

import os
import re
import sys
import json
import time
import glob
import logging

import numpy as np
import torch
import torch.nn as nn
import h5py
from copy import deepcopy

from sr_midas.synthesis.peakbank import create_peakbank
from sr_midas.synthesis.patchstore_gen import create_patchstore
from sr_midas.models.cnnsr.train import train_cnnsr
from sr_midas.models.cnnsr.load import load_trained_CNNSR
from sr_midas.data.patchstore import load_patchstore_h5data, midas_Zarr_zip
from sr_midas.data.upscale import upscale

SEP = os.sep

# ── Default configuration ────────────────────────────────────────────────────
# These match the hyperparameters used to train the bundled pretrained models.

DEFAULTS = {
    # Peakbank
    "cvsz": 50,
    "srfac": 16,
    "I_thresh": 30,
    "peak_recon_err_threshold": 0.25,

    # Patchstore
    "n_patches": 60000,
    "patch_size": 20,
    "srfac_list": "1-2-4-8",
    "n_peaks_per_patch": "1-2-3-4-5",
    "peak_sep_min": 3.0,
    "peak_sep_max": 15.0,
    "var_R": 3.0,
    "err_cut": 0.25,
    "integ_int_cut": 0.0,
    "midas_I_thresh": 30.0,
    "peak_I_min": 10.0,
    "peak_I_max": 100000.0,
    "sr_I_thresh_fac": 0.001,

    # Training
    "arch": "256-5-r_128-5-r_64-5-r_32-5-r_16-5-r_8-5-r_1-5-s",
    "lr": 0.0001,
    "loss_fn": "mse",
    "batch_size": 512,
    "max_itr_x2": 5000,
    "max_itr_x4": 1000,
    "max_itr_x8": 1000,
    "train_frac": 0.8,
    "ec_val": 0.02,
    "ec_itr": 10,
    "n_workers": 1,
    "pred_batch_size": 500,
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_best_iteration(mod_dir):
    """Find the highest iteration checkpoint in a model output directory."""
    pth_files = glob.glob(os.path.join(mod_dir, "mod-it*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No mod-it*.pth files found in {mod_dir}")
    itrs = []
    for f in pth_files:
        match = re.search(r"mod-it(\d+)\.pth", os.path.basename(f))
        if match:
            itrs.append(int(match.group(1)))
    return max(itrs)


def _extract_beam_center(midas_dir):
    """Extract beam center (Ypx_BC, Zpx_BC) from the MIDAS zip file."""
    zip_files = [f for f in os.listdir(midas_dir) if f.endswith(".MIDAS.zip")]
    if not zip_files:
        raise FileNotFoundError(
            f"No .MIDAS.zip file found in {midas_dir}"
        )
    zip_path = os.path.join(midas_dir, zip_files[0])
    zf, _, _, _ = midas_Zarr_zip(zip_path)
    params = zf["analysis"]["process"]["analysis_parameters"]
    Ypx_BC = float(params["YCen"][0])
    Zpx_BC = float(params["ZCen"][0])
    return Ypx_BC, Zpx_BC


def _create_pred_patchstore(pst_path, mod_dir, mod_itr,
                             srfac_in, srfac_out,
                             save_path, batch_size=500):
    """Create a predicted patchstore by running a trained model on input patches."""
    if torch.cuda.is_available():
        torch_devs = torch.device("cuda")
    elif torch.backends.mps.is_available():
        torch_devs = torch.device("mps")
    else:
        torch_devs = torch.device("cpu")

    sr_mod, sr_mod_args, sr_mod_ch = load_trained_CNNSR(
        mod_dir=mod_dir, mod_itr=mod_itr, torch_devs=torch_devs
    )

    patch_arr, *_ = load_patchstore_h5data(pst_path, only_patch_arrays=True)
    X_in = patch_arr[f"SRx{srfac_in}"]
    X_in = X_in[:, sr_mod_ch, :, :]

    upscale_factor = int(srfac_out / srfac_in)
    X = np.zeros(shape=(X_in.shape[0], len(sr_mod_ch),
                         X_in.shape[2] * upscale_factor,
                         X_in.shape[3] * upscale_factor))

    for i in range(len(X)):
        Xi_upscaled = upscale(X_in[i, 0, :, :], srfac_in, srfac_out)
        max_val = np.max(Xi_upscaled)
        X[i, 0, :, :] = Xi_upscaled / max_val if max_val > 0 else Xi_upscaled
    del X_in

    n_patches = len(X)
    n_batches = n_patches // batch_size

    SRx_pred = np.empty(
        (n_patches, 1, X.shape[2], X.shape[3]), dtype=np.float32
    )
    with torch.no_grad():
        for i in range(n_batches + 1):
            i_s = i * batch_size
            i_f = min((i + 1) * batch_size, n_patches)
            if i_s < n_patches:
                X_batch = torch.from_numpy(
                    X[i_s:i_f].astype(np.float32)
                ).to(torch_devs)
                SRx_pred[i_s:i_f] = sr_mod(X_batch).detach().cpu().numpy()
    del X

    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with h5py.File(save_path, "w") as h5file:
        group = h5file.create_group("patchArr")
        group.create_dataset(f"SRx{srfac_out}", data=SRx_pred, dtype=np.float32)


def _generate_sr_config(output_dir, x2_mod_dir, x2_itr,
                         x4_mod_dir, x4_itr,
                         x8_mod_dir, x8_itr):
    """Generate an sr_config.json that points to the trained models."""
    import importlib.resources as ilr
    config_file = ilr.files("sr_midas.models.cnnsr") / "cnnsr_sr_config.json"
    with config_file.open("r") as f:
        sr_config = json.load(f)

    sr_config["mods_to_use"]["SRx2"]["mod_dir"] = os.path.abspath(x2_mod_dir)
    sr_config["mods_to_use"]["SRx2"]["mod_itr"] = x2_itr
    sr_config["mods_to_use"]["SRx4"]["mod_dir"] = os.path.abspath(x4_mod_dir)
    sr_config["mods_to_use"]["SRx4"]["mod_itr"] = x4_itr
    sr_config["mods_to_use"]["SRx8"]["mod_dir"] = os.path.abspath(x8_mod_dir)
    sr_config["mods_to_use"]["SRx8"]["mod_itr"] = x8_itr

    config_path = os.path.join(output_dir, "sr_config.json")
    with open(config_path, "w") as f:
        json.dump(sr_config, f, indent=4)
    return config_path


# ── Main orchestration ───────────────────────────────────────────────────────

def run_auto_train(config_path):
    """Run the full automated training pipeline.

    Args:
        config_path (str): Path to JSON config file. Required keys:
            - midas_dir (list of str): MIDAS data directories with .MIDAS.zip files
            - output_dir (str): Output directory for all artifacts
            All other keys have sensible defaults (see DEFAULTS dict).

    Returns:
        str: Path to the generated sr_config.json
    """
    t_total = time.time()

    # ── 1. Load and validate config ──
    with open(config_path, "r") as f:
        user_config = json.load(f)

    cfg = {**DEFAULTS, **user_config}

    if "midas_dir" not in cfg:
        raise ValueError("Config must include 'midas_dir' (list of MIDAS data directories)")
    if "output_dir" not in cfg:
        raise ValueError("Config must include 'output_dir' (output directory path)")

    if isinstance(cfg["midas_dir"], str):
        cfg["midas_dir"] = [cfg["midas_dir"]]

    output_dir = os.path.abspath(cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    # Set up logging
    log_path = os.path.join(output_dir, "auto_train.log")
    logger = logging.getLogger("auto_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info("=" * 60)
    logger.info("SR-MIDAS AUTOMATED TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Config file: {os.path.abspath(config_path)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"MIDAS directories: {cfg['midas_dir']}")

    # Save resolved config for reproducibility
    resolved_cfg_path = os.path.join(output_dir, "auto_train_config_resolved.json")
    with open(resolved_cfg_path, "w") as f:
        json.dump(cfg, f, indent=4)

    # ── 2. Extract beam center from MIDAS zip ──
    logger.info("")
    logger.info("-" * 40)
    logger.info("STEP 0: Extracting beam center from MIDAS zip")
    ts = time.time()
    Ypx_BC, Zpx_BC = _extract_beam_center(cfg["midas_dir"][0])
    logger.info(f"  Beam center: Ypx_BC={Ypx_BC}, Zpx_BC={Zpx_BC}")
    logger.info(f"  Time: {time.time() - ts:.2f} s")

    # ── 3. Create peakbank ──
    logger.info("")
    logger.info("-" * 40)
    logger.info("STEP 1: Creating peakbank")
    peakbank_dir = os.path.join(output_dir, "peakbank")
    peakbank_path = os.path.join(peakbank_dir, "peakbank.csv")

    if os.path.exists(peakbank_path):
        logger.info(f"  SKIPPED (already exists): {peakbank_path}")
    else:
        os.makedirs(peakbank_dir, exist_ok=True)
        ts = time.time()
        peakbank_config = {
            "midas_dir": cfg["midas_dir"],
            "peakbank_savedir": peakbank_dir,
            "peakbank_savename": "peakbank.csv",
            "peak_recon_err_threshold": cfg["peak_recon_err_threshold"],
            "cvsz": cfg["cvsz"],
            "srfac": cfg["srfac"],
            "I_thresh": cfg["I_thresh"],
            "save_exp_patches": False,
            "dir_ignore": [],
            "save_frame_gen": False,
        }
        create_peakbank(peakbank_config)
        logger.info(f"  Saved: {peakbank_path}")
        logger.info(f"  Time: {time.time() - ts:.2f} s")

    # ── 4. Create patchstore ──
    logger.info("")
    logger.info("-" * 40)
    logger.info("STEP 2: Creating patchstore")
    patchstore_dir = os.path.join(output_dir, "patchstore")
    patchstore_path = os.path.join(patchstore_dir, "patchstore.h5")

    if os.path.exists(patchstore_path):
        logger.info(f"  SKIPPED (already exists): {patchstore_path}")
    else:
        os.makedirs(patchstore_dir, exist_ok=True)
        ts = time.time()
        patchstore_args = {
            "peakbank": peakbank_path,
            "saveName": "patchstore.h5",
            "saveDir": patchstore_dir,
            "nPatch": cfg["n_patches"],
            "lrsz": cfg["patch_size"],
            "cvsz": cfg["cvsz"],
            "srfacSource": cfg["srfac"],
            "srfacAll": cfg["srfac_list"],
            "nPeak": cfg["n_peaks_per_patch"],
            "pSepMin": cfg["peak_sep_min"],
            "pSepMax": cfg["peak_sep_max"],
            "varR": cfg["var_R"],
            "errCut": cfg["err_cut"],
            "integIntCut": cfg["integ_int_cut"],
            "midasIthresh": cfg["midas_I_thresh"],
            "peakImin": cfg["peak_I_min"],
            "peakImax": cfg["peak_I_max"],
            "Ypx_BC": Ypx_BC,
            "Zpx_BC": Zpx_BC,
            "srIthreshFac": cfg["sr_I_thresh_fac"],
            "config": None,
        }
        create_patchstore(patchstore_args)
        logger.info(f"  Saved: {patchstore_path}")
        logger.info(f"  Time: {time.time() - ts:.2f} s")

    # ── 5. Train x1 -> x2 model ──
    logger.info("")
    logger.info("-" * 40)
    logger.info("STEP 3: Training x1 -> x2 model")
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    x2_exp_name = "x1_x2"
    x2_mod_dir = os.path.join(models_dir, f"{x2_exp_name}-itrOut")

    if os.path.exists(x2_mod_dir) and glob.glob(os.path.join(x2_mod_dir, "mod-it*.pth")):
        logger.info(f"  SKIPPED (already trained): {x2_mod_dir}")
    else:
        ts = time.time()
        train_cnnsr({
            "expName": x2_exp_name,
            "pst": patchstore_path,
            "inSRx": 1,
            "outSRx": 2,
            "useRch": "false",
            "useEtach": "false",
            "arch": cfg["arch"],
            "lr": cfg["lr"],
            "lossF": cfg["loss_fn"],
            "mbsz": cfg["batch_size"],
            "maxItr": cfg["max_itr_x2"],
            "trainFrac": cfg["train_frac"],
            "nwork": cfg["n_workers"],
            "ecVal": cfg["ec_val"],
            "ecItr": cfg["ec_itr"],
            "inPstPath": None,
            "outPstPath": None,
            "loadChkpt": None,
            "trainOutDir": models_dir,
        })
        logger.info(f"  Saved: {x2_mod_dir}")
        logger.info(f"  Time: {time.time() - ts:.2f} s")

    x2_best_itr = _find_best_iteration(x2_mod_dir)
    logger.info(f"  Best iteration: {x2_best_itr}")

    # ── 6. Create x2 predicted patchstore ──
    logger.info("")
    logger.info("-" * 40)
    logger.info("STEP 4: Creating x2 predicted patchstore")
    pred_pst_dir = os.path.join(output_dir, "pred_patchstores")
    os.makedirs(pred_pst_dir, exist_ok=True)
    x2pred_path = os.path.join(pred_pst_dir, "x2pred.h5")

    if os.path.exists(x2pred_path):
        logger.info(f"  SKIPPED (already exists): {x2pred_path}")
    else:
        ts = time.time()
        _create_pred_patchstore(
            pst_path=patchstore_path,
            mod_dir=x2_mod_dir,
            mod_itr=x2_best_itr,
            srfac_in=1,
            srfac_out=2,
            save_path=x2pred_path,
            batch_size=cfg["pred_batch_size"],
        )
        logger.info(f"  Saved: {x2pred_path}")
        logger.info(f"  Time: {time.time() - ts:.2f} s")

    # ── 7. Train x2 -> x4 model ──
    logger.info("")
    logger.info("-" * 40)
    logger.info("STEP 5: Training x2pred -> x4 model")
    x4_exp_name = "x2pred_x4"
    x4_mod_dir = os.path.join(models_dir, f"{x4_exp_name}-itrOut")

    if os.path.exists(x4_mod_dir) and glob.glob(os.path.join(x4_mod_dir, "mod-it*.pth")):
        logger.info(f"  SKIPPED (already trained): {x4_mod_dir}")
    else:
        ts = time.time()
        train_cnnsr({
            "expName": x4_exp_name,
            "pst": patchstore_path,
            "inSRx": 2,
            "outSRx": 4,
            "useRch": "false",
            "useEtach": "false",
            "arch": cfg["arch"],
            "lr": cfg["lr"],
            "lossF": cfg["loss_fn"],
            "mbsz": cfg["batch_size"],
            "maxItr": cfg["max_itr_x4"],
            "trainFrac": cfg["train_frac"],
            "nwork": cfg["n_workers"],
            "ecVal": cfg["ec_val"],
            "ecItr": cfg["ec_itr"],
            "inPstPath": x2pred_path,
            "outPstPath": None,
            "loadChkpt": None,
            "trainOutDir": models_dir,
        })
        logger.info(f"  Saved: {x4_mod_dir}")
        logger.info(f"  Time: {time.time() - ts:.2f} s")

    x4_best_itr = _find_best_iteration(x4_mod_dir)
    logger.info(f"  Best iteration: {x4_best_itr}")

    # ── 8. Create x4 predicted patchstore ──
    logger.info("")
    logger.info("-" * 40)
    logger.info("STEP 6: Creating x4 predicted patchstore")
    x4pred_path = os.path.join(pred_pst_dir, "x4pred.h5")

    if os.path.exists(x4pred_path):
        logger.info(f"  SKIPPED (already exists): {x4pred_path}")
    else:
        ts = time.time()
        _create_pred_patchstore(
            pst_path=x2pred_path,
            mod_dir=x4_mod_dir,
            mod_itr=x4_best_itr,
            srfac_in=2,
            srfac_out=4,
            save_path=x4pred_path,
            batch_size=cfg["pred_batch_size"],
        )
        logger.info(f"  Saved: {x4pred_path}")
        logger.info(f"  Time: {time.time() - ts:.2f} s")

    # ── 9. Train x4 -> x8 model ──
    logger.info("")
    logger.info("-" * 40)
    logger.info("STEP 7: Training x4pred -> x8 model")
    x8_exp_name = "x4pred_x8"
    x8_mod_dir = os.path.join(models_dir, f"{x8_exp_name}-itrOut")

    if os.path.exists(x8_mod_dir) and glob.glob(os.path.join(x8_mod_dir, "mod-it*.pth")):
        logger.info(f"  SKIPPED (already trained): {x8_mod_dir}")
    else:
        ts = time.time()
        train_cnnsr({
            "expName": x8_exp_name,
            "pst": patchstore_path,
            "inSRx": 4,
            "outSRx": 8,
            "useRch": "false",
            "useEtach": "false",
            "arch": cfg["arch"],
            "lr": cfg["lr"],
            "lossF": cfg["loss_fn"],
            "mbsz": cfg["batch_size"],
            "maxItr": cfg["max_itr_x8"],
            "trainFrac": cfg["train_frac"],
            "nwork": cfg["n_workers"],
            "ecVal": cfg["ec_val"],
            "ecItr": cfg["ec_itr"],
            "inPstPath": x4pred_path,
            "outPstPath": None,
            "loadChkpt": None,
            "trainOutDir": models_dir,
        })
        logger.info(f"  Saved: {x8_mod_dir}")
        logger.info(f"  Time: {time.time() - ts:.2f} s")

    x8_best_itr = _find_best_iteration(x8_mod_dir)
    logger.info(f"  Best iteration: {x8_best_itr}")

    # ── 10. Generate sr_config.json ──
    logger.info("")
    logger.info("-" * 40)
    logger.info("STEP 8: Generating sr_config.json")
    sr_config_path = _generate_sr_config(
        output_dir=output_dir,
        x2_mod_dir=x2_mod_dir, x2_itr=x2_best_itr,
        x4_mod_dir=x4_mod_dir, x4_itr=x4_best_itr,
        x8_mod_dir=x8_mod_dir, x8_itr=x8_best_itr,
    )
    logger.info(f"  Saved: {sr_config_path}")

    # ── Summary ──
    t_elapsed = time.time() - t_total
    logger.info("")
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {t_elapsed:.1f} s ({t_elapsed/60:.1f} min)")
    logger.info("")
    logger.info("Trained models:")
    logger.info(f"  x1->x2 : {x2_mod_dir} (iteration {x2_best_itr})")
    logger.info(f"  x2->x4 : {x4_mod_dir} (iteration {x4_best_itr})")
    logger.info(f"  x4->x8 : {x8_mod_dir} (iteration {x8_best_itr})")
    logger.info("")
    logger.info("To use these models with sr_process:")
    logger.info(f"  sr-midas-process -midasZarrDir <your_data> -SRconfig {sr_config_path}")
    logger.info("")
    logger.info(f"Full log: {log_path}")

    return sr_config_path
