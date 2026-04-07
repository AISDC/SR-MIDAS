"""MIDAS super-resolution processing pipeline.

Refactored from: SR-MIDAS/super_res_process.py (execution block, lines 808-1514)
All function definitions are imported from their canonical library modules.

GPU acceleration: when use_gpu=1 (default) and a CUDA GPU is available, peak
fitting is performed using batched GPU pseudo-Voigt fitting via _gpu_peakfit.py.
This replaces the per-patch scipy.curve_fit loop with a ~1000x faster batched
Adam optimizer on GPU while producing equivalent peak positions.
"""

import os
import sys
import json
import math
import shutil
import time

import numpy as np
import pandas as pd
import zarr
import torch
from torch.amp import autocast
from scipy.ndimage import gaussian_filter, median_filter
from skimage.feature import peak_local_max
from copy import deepcopy

from sr_midas.utils.io import (NumpyEncoder, setup_logging, read_hkls_csv,
                                parse_sr_config_txt, update_nested_dict)
from sr_midas.physics.detector import (create_rotation_matrices, create_distortion_map,
                                        ringNr_map_on_detector)
from sr_midas.physics.peaks2d import pseudoVoigt2d_diffLGwidth
from sr_midas.models.cnnsr import load_trained_CNNSR
from sr_midas.pipeline._patch_ops import (patches_from_detector_frame,
                                           weighted_center_of_mass, com_peak_coords,
                                           watershed_peaks, multi_pv_fit)

SEP = os.sep


def run_sr_process(midasZarrDir, srfac=8, SRconfig_path=None,
                   saveSRpatches=1, saveFrameGoodCoords=1,
                   use_gpu=1):
    """Run the MIDAS super-resolution processing pipeline.

    Loads a MIDAS zarr zip file, applies cascaded CNN super-resolution to
    detector patches, fits peak shapes (center-of-mass or pseudo-Voigt),
    and writes per-frame peak CSV files to the MIDAS Temp/ directory.

    Args:
        midasZarrDir (str): Directory containing the MIDAS .zip zarr file
        srfac (int; default=8): Super-resolution factor (2, 4, or 8)
        SRconfig_path (str or None; default=None): Path to SR config (.json or .txt).
            When None, the bundled cnnsr_sr_config.json with pretrained models is used.
        saveSRpatches (int; default=1): 1 to save SR patches to disk, 0 to skip
        saveFrameGoodCoords (int; default=1): 1 to save goodCoords maps, 0 to skip
        use_gpu (int; default=1): 1 to use GPU-accelerated peak fitting when CUDA
            is available, 0 to always use the CPU path
    """

    if midasZarrDir[-1] != SEP:
        midasZarrDir = midasZarrDir + SEP

    t0 = time.time()
    t_run = 0

    ts = time.time()
    SRlogger = setup_logging(midasZarrDir)
    tf = time.time()
    t_run += tf - ts
    SRlogger.info(f"{'-'*5} Time to setup logging: {tf - ts:.5f} s | SR_run_time: {t_run:.5f} s")

    SRlogger.info(f"Searching for MIDAS zip file in dir: {midasZarrDir}")
    ts = time.time()
    midas_zip_filename = [i for i in os.listdir(midasZarrDir) if ".MIDAS.zip" in i][0]
    midas_zip_filepath = f"{midasZarrDir}{midas_zip_filename}"
    tf = time.time()
    t_run += tf - ts
    SRlogger.info(f"\t|Found MIDAS zip file: {midas_zip_filename}")
    SRlogger.info(f"{'-'*5} Time to find MIDAS zip file: {tf - ts:.5f} s | SR_run_time: {t_run:.5f} s")

    SRlogger.info(f"Loading Zarr file: {midas_zip_filename}")
    ts = time.time()
    try:
        zf = zarr.open(midas_zip_filepath, "r")
    except Exception:
        zf = zarr.open(zarr.storage.ZipStore(midas_zip_filepath, mode='r'), mode='r')

    SRlogger.info(f"\t|Extracting parameters needed for super-res workflow")
    zf_process_params = zf["analysis"]["process"]["analysis_parameters"]
    zf_scan_params = zf["measurement/process/scan_parameters"]

    sr_params = {}
    sr_params["Ypx_BC"] = float(zf_process_params["YCen"][0])
    sr_params["Zpx_BC"] = float(zf_process_params["ZCen"][0])
    sr_params["ringsToUse"] = list(zf_process_params["RingThresh"][:][:, 0].astype(int))
    sr_params["ringsThresh"] = list(zf_process_params["RingThresh"][:][:, 1].astype(float))
    sr_params["spacegroup"] = int(zf_process_params["SpaceGroup"][0])
    sr_params["lattParam"] = list(zf_process_params["LatticeParameter"][:].astype(float))
    sr_params["xrayLambda"] = float(zf_process_params["Wavelength"][0])
    sr_params["Lsd"] = float(zf_process_params["Lsd"][0])
    sr_params["pxSize"] = float(zf_process_params["PixelSize"][0])
    sr_params["tx"] = float(zf_process_params["tx"][0])
    sr_params["ty"] = float(zf_process_params["ty"][0])
    sr_params["tz"] = float(zf_process_params["tz"][0])
    sr_params["p0"] = float(zf_process_params["p0"][0])
    sr_params["p1"] = float(zf_process_params["p1"][0])
    sr_params["p2"] = float(zf_process_params["p2"][0])
    sr_params["p3"] = float(zf_process_params["p3"][0])
    sr_params["ring_width"] = float(zf_process_params["Width"][0])
    sr_params["omega_start"] = float(zf_scan_params["start"][0])
    sr_params["omega_stepsize"] = float(zf_scan_params["step"][0])
    sr_params["numPxY"] = int(zf["exchange"]["data"].shape[2])
    sr_params["numPxZ"] = int(zf["exchange"]["data"].shape[1])

    sr_params["padding"] = 6
    if "Padding" in zf_process_params:
        sr_params["padding"] = int(zf_process_params["Padding"][0])

    sr_params["SkipFrame"] = 0
    if "SkipFrame" in zf_process_params:
        sr_params["SkipFrame"] = int(zf_process_params["SkipFrame"][0])

    sr_params["ImTransOpt"] = 0
    if "ImTransOpt" in zf_process_params:
        sr_params["ImTransOpt"] = int(zf_process_params["ImTransOpt"][0])

    SRlogger.info(f"\t|Loading Zarr data")
    exData = np.array(zf["exchange"]["data"])[sr_params["SkipFrame"]:]

    del zf
    tf = time.time()
    t_run += tf - ts
    SRlogger.info(f"\t|Nr of frames={exData.shape[0]}; after skipping {sr_params['SkipFrame']} frames")
    SRlogger.info(f"{'-'*5} Time to load midas zip file: {tf - ts:.5f} s | SR_run_time: {t_run:.5f} s")

    ts = time.time()

    if SRconfig_path is None:
        import importlib.resources as ilr
        SRlogger.info("Loading SR configuration: bundled cnnsr_sr_config.json (pretrained models)")
        config_file = ilr.files("sr_midas.models.cnnsr") / "cnnsr_sr_config.json"
        with config_file.open("r") as f:
            sr_config = json.load(f)
        pretrained_base = ilr.files("sr_midas.models.cnnsr.pretrained")
        for sr_level in ["SRx2", "SRx4", "SRx8"]:
            dir_name = sr_config["mods_to_use"][sr_level]["mod_dir"]
            sr_config["mods_to_use"][sr_level]["mod_dir"] = str(pretrained_base / dir_name)
    elif SRconfig_path.endswith(".json"):
        SRlogger.info(f"Loading SR configuration file: {SRconfig_path}")
        with open(SRconfig_path, "r") as f:
            sr_config = json.load(f)
    elif SRconfig_path.endswith(".txt"):
        SRlogger.info(f"Loading SR configuration file: {SRconfig_path}")
        sr_config = parse_sr_config_txt(SRconfig_path)
    else:
        SRlogger.error(f"Invalid SR config file: {SRconfig_path}. Must be .json or .txt.")
        sys.exit(1)

    sr_config_savepath = os.path.join(midasZarrDir, "SR_out", "sr_config.json")
    with open(sr_config_savepath, "w") as f:
        json.dump(sr_config, f, indent=4, cls=NumpyEncoder)
    SRlogger.info(f"\t|SR config saved to: {sr_config_savepath}")

    sr_params_savepath = os.path.join(midasZarrDir, "SR_out", "sr_params.json")
    with open(sr_params_savepath, "w") as f:
        json.dump(sr_params, f, indent=4, cls=NumpyEncoder)
    SRlogger.info(f"\t|SR params saved to: {sr_params_savepath}")

    tf = time.time()
    t_run += tf - ts
    SRlogger.info(f"{'-'*5} Time to load & save SR config: {tf - ts:.5f} s | SR_run_time: {t_run:.5f} s")

    SRlogger.info(f"Reading ring positions from '{midasZarrDir}hkls.csv'")
    ts = time.time()
    df_hkls = read_hkls_csv(f"{midasZarrDir}hkls.csv")
    rings_Rpx = (df_hkls["Radius"].unique() / sr_params["pxSize"]).tolist()
    rings_to_use_Rpx = [rings_Rpx[int(i) - 1] for i in sr_params['ringsToUse']]
    sr_params["rings_to_use_Rpx"] = rings_to_use_Rpx
    tf = time.time()
    t_run += tf - ts
    SRlogger.info(f"Ring radii (px): {rings_to_use_Rpx}")
    SRlogger.info(f"{'-'*5} Time to calculate ring radii: {tf - ts:.5f} s | SR_run_time: {t_run:.5f} s")

    SRlogger.info(f"Creating detector RingNr map")
    ts = time.time()
    RingNrmap = ringNr_map_on_detector(sr_params)
    RingNrmap_savepath = os.path.join(midasZarrDir, "SR_out", "RingNrmap.npy")
    np.save(RingNrmap_savepath, RingNrmap)
    tf = time.time()
    t_run += tf - ts
    SRlogger.info(f"{'-'*5} Time to create RingNr map: {tf - ts:.5f} s | SR_run_time: {t_run:.5f} s")

    ts = time.time()
    peaks_csv_dir = f"{midasZarrDir}Temp{SEP}"
    if sr_config["skipFitIfExists"].lower() in ["no", "n", "false", "f", "0"]:
        if os.path.isdir(peaks_csv_dir):
            shutil.rmtree(peaks_csv_dir)
            SRlogger.info(f"Deleted existing 'Temp' folder")
        os.mkdir(peaks_csv_dir)
        SRlogger.info(f"Created new 'Temp' folder")
    else:
        if not os.path.isdir(peaks_csv_dir):
            os.mkdir(peaks_csv_dir)
            SRlogger.info(f"Created 'Temp' folder")
        else:
            SRlogger.info(f"Temp folder already exists. Skipping frames that have csv files.")
    tf = time.time()
    t_run += tf - ts

    SRlogger.info(f"Selecting device for ML models")
    ts = time.time()
    if torch.cuda.is_available():
        torch_devs = torch.device("cuda")
    elif torch.backends.mps.is_available():
        torch_devs = torch.device("mps")
    else:
        torch_devs = torch.device("cpu")
    tf = time.time()
    t_run += tf - ts
    SRlogger.info(f"\t|Device selected: {torch_devs}")

    SRlogger.info(f"Loading the trained models")
    ts = time.time()
    with torch.no_grad():
        x2mod, x2mod_args, x2mod_ch = load_trained_CNNSR(
            mod_dir=sr_config["mods_to_use"]["SRx2"]["mod_dir"],
            mod_itr=sr_config["mods_to_use"]["SRx2"]["mod_itr"],
            torch_devs=torch_devs)
        x4mod, x4mod_args, x4mod_ch = load_trained_CNNSR(
            mod_dir=sr_config["mods_to_use"]["SRx4"]["mod_dir"],
            mod_itr=sr_config["mods_to_use"]["SRx4"]["mod_itr"],
            torch_devs=torch_devs)
        x8mod, x8mod_args, x8mod_ch = load_trained_CNNSR(
            mod_dir=sr_config["mods_to_use"]["SRx8"]["mod_dir"],
            mod_itr=sr_config["mods_to_use"]["SRx8"]["mod_itr"],
            torch_devs=torch_devs)
        x2mod.eval()
        x4mod.eval()
        x8mod.eval()
    tf = time.time()
    t_run += tf - ts
    SRlogger.info(f"{'-'*5} Time to load models: {tf - ts:.5f} s | SR_run_time: {t_run:.5f} s")

    # ── GPU peak fitting setup ──
    gpu_peakfit_enabled = False
    if use_gpu == 1 and torch_devs.type == "cuda":
        try:
            from sr_midas.pipeline._gpu_peakfit import gpu_fit_frame_patches, warmup_gpu_compile
            gpu_peakfit_enabled = True
            SRlogger.info(f"GPU peak fitting: ENABLED (CUDA available, use_gpu=1)")

            ts = time.time()
            SRlogger.info(f"\t|Warming up torch.compile for GPU peak fitting (one-time cost)...")
            warmup_gpu_compile(sr_config, sr_params, srfac, torch_devs)
            tf = time.time()
            t_run += tf - ts
            SRlogger.info(f"{'-'*5} Time to warmup GPU peak fitting: {tf - ts:.5f} s | SR_run_time: {t_run:.5f} s")
        except Exception as e:
            SRlogger.warning(f"GPU peak fitting: DISABLED (import failed: {e}). Falling back to CPU.")
            gpu_peakfit_enabled = False
    else:
        reason = "use_gpu=0" if use_gpu == 0 else f"no CUDA (device={torch_devs.type})"
        SRlogger.info(f"GPU peak fitting: DISABLED ({reason}). Using CPU peak fitting.")

    col_names = ["SpotID", "IntegratedIntensity", "Omega(degrees)", "YCen(px)", "ZCen(px)", "IMax",
                 "Radius(px)", "Eta(degrees)", "SigmaR", "SigmaEta", "NrPixels",
                 "TotalNrPixelsInPeakRegion", "nPeaks", "maxY", "maxZ", "diffY", "diffZ"]

    nr_frames = exData.shape[0]

    shiftYpx = sr_config["shift_YZ_pos"][f"SRx{srfac}"]["shiftYpx"]
    shiftZpx = sr_config["shift_YZ_pos"][f"SRx{srfac}"]["shiftZpx"]
    SRlogger.info(f"\t|shiftYpx: {shiftYpx}, shiftZpx: {shiftZpx}")

    for frame_i in range(0, nr_frames):
        t0_frame = time.time()
        SRlogger.info(f"{'='*60}")
        SRlogger.info(f"{'*'*5} Processing frame {frame_i} {'*'*5}")

        frame_i_csv_savename = f"{midas_zip_filename}_{str(frame_i+1).zfill(sr_params['padding'])}_PS.csv"
        frame_i_csv_savepath = f"{peaks_csv_dir}{frame_i_csv_savename}"

        df_peaks_frame_i = pd.DataFrame(columns=col_names)

        if sr_config["skipFitIfExists"].lower() in ["yes", "y", "true", "t", "1"] and os.path.isfile(frame_i_csv_savepath):
            SRlogger.info(f"{'*'*5}| Frame {frame_i} skipped (csv exists).")
            continue

        frame_arr = exData[frame_i]

        if sr_params["ImTransOpt"] == 1:
            frame_arr = np.flip(frame_arr, axis=1)
        if sr_params["ImTransOpt"] == 2:
            frame_arr = np.flip(frame_arr, axis=0)
        if sr_params["ImTransOpt"] == 3:
            frame_arr = np.transpose(frame_arr)

        ts = time.time()
        frame_goodCoords = np.zeros_like(frame_arr)
        for ring_i, ring_thresh in enumerate(sr_params["ringsThresh"]):
            valid_pixels_mask = (RingNrmap == ring_i) & (frame_arr >= ring_thresh)
            frame_goodCoords[valid_pixels_mask] = frame_arr[valid_pixels_mask]

        if saveFrameGoodCoords == 1:
            frame_goodCoords_save_dirpath = os.path.join(midasZarrDir, "SR_out", "frame_goodCoords")
            if not os.path.isdir(frame_goodCoords_save_dirpath):
                os.mkdir(frame_goodCoords_save_dirpath)
            frame_goodCoords_savepath = os.path.join(frame_goodCoords_save_dirpath,
                                                      f"{str(frame_i).zfill(sr_params['padding'])}_GC.npy")
            np.save(frame_goodCoords_savepath, frame_goodCoords)

        tf = time.time()
        t_run += tf - ts
        SRlogger.info(f"{'-'*5} Time to create goodCoords map: {tf - ts:.5f} s | SR_run_time: {t_run:.5f} s")

        ts = time.time()
        patches_exp, patches_Z00, patches_Y00, nr_pixels_in_patch = \
            patches_from_detector_frame(frame_goodCoords, sr_config, connectivity_dim=8)

        n_patches = len(patches_exp)

        if n_patches == 0:
            SRlogger.info(f"\t| No patches found in frame {frame_i}")
            df_peaks_frame_i.to_csv(frame_i_csv_savepath, sep="\t", index=False)
            continue

        patches_exp = np.expand_dims(patches_exp, axis=1)
        patches_Isum = np.sum(patches_exp, axis=(1, 2, 3)).tolist()

        tf = time.time()
        t_run += tf - ts
        SRlogger.info(f"\t| Nr. patches in frame {frame_i}: {n_patches}")
        SRlogger.info(f"{'-'*5} Time to extract patches: {tf - ts:.5f} s | SR_run_time: {t_run:.5f} s")

        # SRx2 model
        ts = time.time()
        upscale_fac = 2
        patches_subset = patches_exp[:, x2mod_ch, :, :]
        upscaled_patches = np.repeat(np.repeat(patches_subset, upscale_fac, axis=2), upscale_fac, axis=3)
        upscaled_patches = upscaled_patches / (upscale_fac * upscale_fac)
        max_vals = np.max(upscaled_patches, axis=(2, 3), keepdims=True)
        max_vals = np.where(max_vals == 0, 1, max_vals)
        Xin_x2 = upscaled_patches / max_vals
        tf = time.time()
        t_run += tf - ts

        ts = time.time()
        n_batches = n_patches // sr_config["batch_size"]
        with torch.no_grad():
            for i in range(0, n_batches + 1):
                i_s, i_f = i * sr_config["batch_size"], min((i + 1) * sr_config["batch_size"], n_patches)
                if i_s < n_patches:
                    X_batch = torch.from_numpy(Xin_x2[i_s:i_f].astype(np.float32)).to(torch_devs)
                    SRx2_pred_batch = x2mod.forward(X_batch).detach().cpu().numpy()
                    if i == 0: SRx2_pred = deepcopy(SRx2_pred_batch)
                    else: SRx2_pred = np.append(SRx2_pred, SRx2_pred_batch, axis=0)
        tf = time.time()
        t_run += tf - ts
        SRlogger.info(f"{'-'*5} Time to predict SRx2: {tf - ts:.5f} s | SR_run_time: {t_run:.5f} s")

        if srfac > 2.5:
            ts = time.time()
            upscale_fac = 2
            patches_subset = SRx2_pred[:, x4mod_ch, :, :]
            upscaled_patches = np.repeat(np.repeat(patches_subset, upscale_fac, axis=2), upscale_fac, axis=3)
            upscaled_patches = upscaled_patches / (upscale_fac * upscale_fac)
            max_vals = np.max(upscaled_patches, axis=(2, 3), keepdims=True)
            max_vals = np.where(max_vals == 0, 1, max_vals)
            Xin_x4 = upscaled_patches / max_vals
            tf = time.time()
            t_run += tf - ts

            ts = time.time()
            with torch.no_grad():
                for i in range(0, n_batches + 1):
                    i_s, i_f = i * sr_config["batch_size"], min((i + 1) * sr_config["batch_size"], n_patches)
                    if i_s < n_patches:
                        X_batch = torch.from_numpy(Xin_x4[i_s:i_f].astype(np.float32)).to(torch_devs)
                        SRx4_pred_batch = x4mod.forward(X_batch).detach().cpu().numpy()
                        if i == 0: SRx4_pred = deepcopy(SRx4_pred_batch)
                        else: SRx4_pred = np.append(SRx4_pred, SRx4_pred_batch, axis=0)
            tf = time.time()
            t_run += tf - ts
            SRlogger.info(f"{'-'*5} Time to predict SRx4: {tf - ts:.5f} s | SR_run_time: {t_run:.5f} s")

        if srfac > 4.5:
            ts = time.time()
            upscale_fac = 2
            patches_subset = SRx4_pred[:, x8mod_ch, :, :]
            upscaled_patches = np.repeat(np.repeat(patches_subset, upscale_fac, axis=2), upscale_fac, axis=3)
            upscaled_patches = upscaled_patches / (upscale_fac * upscale_fac)
            max_vals = np.max(upscaled_patches, axis=(2, 3), keepdims=True)
            max_vals = np.where(max_vals == 0, 1, max_vals)
            Xin_x8 = upscaled_patches / max_vals
            tf = time.time()
            t_run += tf - ts

            ts = time.time()
            if torch_devs.type == "cuda":
                with torch.no_grad(), autocast(torch_devs.type):
                    for i in range(0, n_batches + 1):
                        i_s, i_f = i * sr_config["batch_size"], min((i + 1) * sr_config["batch_size"], n_patches)
                        if i_s < n_patches:
                            X_batch = torch.from_numpy(Xin_x8[i_s:i_f].astype(np.float32)).to(torch_devs)
                            SRx8_pred_batch = x8mod.forward(X_batch)
                            current_sums = torch.sum(SRx8_pred_batch, dim=(1, 2, 3))
                            batch_target_sums = patches_Isum[i_s:i_f]
                            target_sums = torch.tensor(batch_target_sums, device=SRx8_pred_batch.device)
                            scaling_factors = (target_sums / current_sums).view(-1, 1, 1, 1)
                            SRx8_pred_batch = (SRx8_pred_batch * scaling_factors).detach().cpu().numpy()
                            if i == 0: SRx8_pred = deepcopy(SRx8_pred_batch)
                            else: SRx8_pred = np.append(SRx8_pred, SRx8_pred_batch, axis=0)
            else:
                with torch.no_grad():
                    for i in range(0, n_batches + 1):
                        i_s, i_f = i * sr_config["batch_size"], min((i + 1) * sr_config["batch_size"], n_patches)
                        if i_s < n_patches:
                            X_batch = torch.from_numpy(Xin_x8[i_s:i_f].astype(np.float32)).to(torch_devs)
                            SRx8_pred_batch = x8mod.forward(X_batch)
                            current_sums = torch.sum(SRx8_pred_batch, dim=(1, 2, 3))
                            batch_target_sums = patches_Isum[i_s:i_f]
                            target_sums = torch.tensor(batch_target_sums, device=SRx8_pred_batch.device)
                            scaling_factors = (target_sums / current_sums).view(-1, 1, 1, 1)
                            SRx8_pred_batch = (SRx8_pred_batch * scaling_factors).detach().cpu().numpy()
                            if i == 0: SRx8_pred = deepcopy(SRx8_pred_batch)
                            else: SRx8_pred = np.append(SRx8_pred, SRx8_pred_batch, axis=0)
            tf = time.time()
            t_run += tf - ts
            SRlogger.info(f"{'-'*5} Time to predict SRx8: {tf - ts:.5f} s | SR_run_time: {t_run:.5f} s")

        if srfac == 2: patches_to_fit = SRx2_pred
        if srfac == 4: patches_to_fit = SRx4_pred
        if srfac == 8: patches_to_fit = SRx8_pred

        if saveSRpatches == 1:
            SR_patch_save_dirpath = f"{midasZarrDir}SR_out{SEP}SR_patches{SEP}"
            if not os.path.isdir(SR_patch_save_dirpath):
                os.mkdir(SR_patch_save_dirpath)

            ts = time.time()
            np.save(f"{SR_patch_save_dirpath}{str(frame_i).zfill(sr_params['padding'])}_SRx1_exp.npy", patches_exp)
            np.save(f"{SR_patch_save_dirpath}{str(frame_i).zfill(sr_params['padding'])}_SRx2_pred.npy", SRx2_pred)
            if srfac > 2.5:
                np.save(f"{SR_patch_save_dirpath}{str(frame_i).zfill(sr_params['padding'])}_SRx4_pred.npy", SRx4_pred)
            if srfac > 4.5:
                np.save(f"{SR_patch_save_dirpath}{str(frame_i).zfill(sr_params['padding'])}_SRx8_pred.npy", SRx8_pred)
            tf = time.time()
            t_run += tf - ts

        ts = time.time()
        spotID = int(0)
        orig_frame_index = frame_i + sr_params["SkipFrame"]
        omega = sr_params["omega_start"] + (sr_params["omega_stepsize"] * orig_frame_index)

        R_patches = np.sqrt(
            (sr_params["Ypx_BC"] - (np.array(patches_Y00) + sr_config["spot_find_args"]["patch_size"] / 2))**2 +
            (sr_params["Zpx_BC"] - (np.array(patches_Z00) + sr_config["spot_find_args"]["patch_size"] / 2))**2
        )

        if gpu_peakfit_enabled:
            # ── GPU path: batched pseudo-Voigt fitting on all patches at once ──
            SRlogger.info(f"\t| GPU peak fitting: {n_patches} patches")
            df_rows, n_peaks_in_patches_list, spotID = gpu_fit_frame_patches(
                patches_to_fit, patches_Y00, patches_Z00,
                patches_exp, nr_pixels_in_patch,
                sr_params, sr_config, srfac,
                omega, shiftYpx, shiftZpx,
                torch_devs)
            for row in df_rows:
                df_peaks_frame_i.loc[len(df_peaks_frame_i)] = row

        else:
            # ── CPU path: original per-patch fitting (COM or PV based on config) ──
            n_peaks_in_patches_list = []
            for patch_i in range(len(patches_to_fit)):
                patch = patches_to_fit[patch_i, 0]
                Z00, Y00 = patches_Z00[patch_i], patches_Y00[patch_i]

                patch_for_locmax = gaussian_filter(patch, sigma=sr_config["peak_find_args"]["gauss_filter_sigma"][f"SRx{srfac}"])
                patch_for_locmax = median_filter(patch_for_locmax, size=sr_config["peak_find_args"]["median_filter_size"][f"SRx{srfac}"])

                locmax_peak_coords = peak_local_max(patch_for_locmax,
                                                    min_distance=sr_config["peak_find_args"]["min_d"][f"SRx{srfac}"],
                                                    threshold_rel=sr_config["peak_find_args"]["thresh_rel"][f"SRx{srfac}"])

                n_peaks_in_patch = len(locmax_peak_coords)
                n_peaks_in_patches_list.append(n_peaks_in_patch)

                if (str(sr_config["fitPeakShapePV"]).lower() in ["no", "n", "false", "f", "0"]) or \
                   (str(sr_config["fitPeakShapePV"]).lower() in ["auto", "partial", "mix"] and n_peaks_in_patch == 1):

                    bnd_peak_cutoff = sr_config["edge_bound_cutoff_fac"] * srfac
                    locmax_peak_coords = locmax_peak_coords[~(locmax_peak_coords[:, :] < bnd_peak_cutoff).any(1)]
                    locmax_peak_coords = locmax_peak_coords[~(locmax_peak_coords[:, :] > ((sr_config["lrsz"] * srfac) - bnd_peak_cutoff)).any(1)]

                    com_coords = com_peak_coords(patch, locmax_peak_coords,
                                                 threshold=0.7,
                                                 peak_crop_size=sr_config["peak_find_args"]["peak_crop_size"][f"SRx{srfac}"])

                    labelled_patch, peak_label_values, peaks_Isum, peaks_NrPixels = \
                        watershed_peaks(patch, locmax_peak_coords,
                                        mask_thresh=sr_config["peak_find_args"]["thresh_rel"][f"SRx{srfac}"])

                    n_peaks_in_patch_after_ws = len(com_coords)
                    for (peak_i_inner, peak_label) in zip(range(n_peaks_in_patch_after_ws), peak_label_values):
                        spotID += 1

                        peak_i_patch = patch * (labelled_patch == peak_label)

                        downscale_fac = int(srfac / 1)
                        h, w = peak_i_patch.shape
                        new_h, new_w = h // downscale_fac, w // downscale_fac
                        reshaped = peak_i_patch[:new_h * downscale_fac, :new_w * downscale_fac].reshape(
                            new_h, downscale_fac, new_w, downscale_fac)
                        peak_i_patch_SRx1 = np.sum(reshaped, axis=(1, 3))

                        IntegratedIntensity = np.sum(peak_i_patch_SRx1)
                        NrPixels = np.count_nonzero(peak_i_patch_SRx1 > 1E-2)
                        IMax = np.max(peak_i_patch_SRx1)
                        TotalNrPixelsInPeakRegion = nr_pixels_in_patch[patch_i]

                        YCen_px = Y00 + (com_coords[peak_i_inner][1] * (1 / srfac)) + float(shiftYpx)
                        ZCen_px = Z00 + (com_coords[peak_i_inner][0] * (1 / srfac)) + float(shiftZpx)

                        R = math.sqrt((sr_params["Ypx_BC"] - YCen_px)**2 + (sr_params["Zpx_BC"] - ZCen_px)**2)
                        Eta = math.degrees(math.acos((ZCen_px - sr_params["Zpx_BC"]) / R))
                        Eta = Eta * ((YCen_px - sr_params["Ypx_BC"]) / np.abs(YCen_px - sr_params["Ypx_BC"]))

                        SigmaR, SigmaEta = 1, 1

                        r_max_SRx1, c_max_SRx1 = np.unravel_index(np.argmax(peak_i_patch_SRx1), peak_i_patch_SRx1.shape)
                        maxY = Y00 + c_max_SRx1
                        maxZ = Z00 + r_max_SRx1
                        diffY = maxY - YCen_px
                        diffZ = maxZ - ZCen_px

                        peak_data = [spotID, IntegratedIntensity, omega, YCen_px, ZCen_px, IMax, R, Eta,
                                     SigmaR, SigmaEta, NrPixels, TotalNrPixelsInPeakRegion,
                                     n_peaks_in_patch, maxY, maxZ, diffY, diffZ]
                        df_peaks_frame_i.loc[spotID - 1] = peak_data

                if (str(sr_config["fitPeakShapePV"]).lower() in ["yes", "y", "true", "t", "1"]) or \
                   (str(sr_config["fitPeakShapePV"]).lower() in ["auto", "partial", "mix"] and n_peaks_in_patch >= 2):

                    try:
                        pvfit_peaks, _, _ = multi_pv_fit(patch, patches_Y00[patch_i], patches_Z00[patch_i],
                                                          sr_params["Ypx_BC"], sr_params["Zpx_BC"],
                                                          sr_config["lrsz"], srfac,
                                                          min_distance=sr_config["peak_find_args"]["min_d"][f"SRx{srfac}"],
                                                          threshold_rel=sr_config["peak_find_args"]["thresh_rel"][f"SRx{srfac}"],
                                                          gauss_filter_sigma=sr_config["peak_find_args"]["gauss_filter_sigma"][f"SRx{srfac}"],
                                                          median_filter_size=sr_config["peak_find_args"]["median_filter_size"][f"SRx{srfac}"],
                                                          lr_int_thresh=sr_config["peak_find_args"]["pvfit_int_thresh"][f"SRx{srfac}"])

                        n_peaks_in_patch = len(pvfit_peaks)
                        for peak_i_inner in range(n_peaks_in_patch):
                            spotID += 1
                            peak_prop = pvfit_peaks[peak_i_inner]
                            R, Eta = peak_prop["R(px)"], peak_prop["Eta(deg)"]

                            dpx_sr = 1 / srfac
                            Ypx = np.arange(Y00, Y00 + sr_config["spot_find_args"]["patch_size"], dpx_sr)
                            Zpx = np.arange(Z00, Z00 + sr_config["spot_find_args"]["patch_size"], dpx_sr)

                            grid_YY, grid_ZZ = np.meshgrid(Ypx, Zpx)
                            grid_RR = ((sr_params["Ypx_BC"] - grid_YY)**2 + (sr_params["Zpx_BC"] - grid_ZZ)**2)**0.5
                            grid_EE = np.rad2deg(np.arccos((grid_ZZ - sr_params["Zpx_BC"]) / grid_RR))
                            grid_EE = grid_EE * ((grid_YY - sr_params["Ypx_BC"]) / np.abs(grid_YY - sr_params["Ypx_BC"]))

                            peak_fit_patch = pseudoVoigt2d_diffLGwidth(grid_RR, grid_EE,
                                                                        y0=R, z0=Eta,
                                                                        ySigG=peak_prop["SigGR"], zSigG=peak_prop["SigGEta"],
                                                                        ySigL=peak_prop["SigLR"], zSigL=peak_prop["SigLEta"],
                                                                        LGmix=peak_prop["LGmix"], IMax=peak_prop["IMax"])

                            downscale_fac = int(srfac / 1)
                            h, w = peak_fit_patch.shape
                            new_h, new_w = h // downscale_fac, w // downscale_fac
                            reshaped = peak_fit_patch[:new_h * downscale_fac, :new_w * downscale_fac].reshape(
                                new_h, downscale_fac, new_w, downscale_fac)
                            peak_fit_patch_SRx1 = np.sum(reshaped, axis=(1, 3))

                            r_max_SRx1, c_max_SRx1 = np.unravel_index(np.argmax(peak_fit_patch_SRx1), peak_fit_patch_SRx1.shape)
                            maxY = Y00 + c_max_SRx1
                            maxZ = Z00 + r_max_SRx1

                            IntegratedIntensity = np.sum(peak_fit_patch)
                            YCen_px = peak_prop["Y(px)"] + shiftYpx
                            ZCen_px = peak_prop["Z(px)"] + shiftZpx
                            IMax = np.max(peak_fit_patch_SRx1)
                            SigmaR = max(peak_prop["SigGR"], peak_prop["SigLR"])
                            SigmaEta = max(peak_prop["SigGEta"], peak_prop["SigLEta"])
                            NrPixels = np.count_nonzero(peak_fit_patch_SRx1 * patches_exp[patch_i])
                            TotalNrPixelsInPeakRegion = nr_pixels_in_patch[patch_i]

                            diffY = maxY - YCen_px
                            diffZ = maxZ - ZCen_px
                            peak_data = [spotID, IntegratedIntensity, omega, YCen_px, ZCen_px, IMax, R, Eta,
                                         SigmaR, SigmaEta, NrPixels, TotalNrPixelsInPeakRegion,
                                         n_peaks_in_patch, maxY, maxZ, diffY, diffZ]
                            df_peaks_frame_i.loc[spotID - 1] = peak_data

                    except Exception as e:
                        SRlogger.warning(f"\tFrame {frame_i}: Patch {patch_i} skipped (pvfit failed: {e})")

        tf = time.time()
        t_run += tf - ts
        fit_method = "GPU" if gpu_peakfit_enabled else "CPU"
        SRlogger.info(f"{'-'*5} Time to fit peaks [{fit_method}] (frame {frame_i}): {tf - ts:.5f} s | SR_run_time: {t_run:.5f} s")

        df_peaks_frame_i.to_csv(frame_i_csv_savepath, sep="\t", index=False)

        if saveSRpatches == 1:
            patches_info_fp = f"{SR_patch_save_dirpath}{str(frame_i).zfill(sr_params['padding'])}_patches_info.csv"
            patches_info_df = pd.DataFrame({
                "Y00": patches_Y00,
                "Z00": patches_Z00,
                "nr_pixels_in_patch": nr_pixels_in_patch,
                "patches_Isum": patches_Isum,
                "n_peaks_in_patches": n_peaks_in_patches_list
            })
            patches_info_df.to_csv(patches_info_fp)
            SRlogger.info(f"\t| Saved patches info: {patches_info_fp}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MIDAS super-resolution processing pipeline")
    parser.add_argument("midas_dir", help="Directory containing MIDAS .zip zarr file and hkls.csv")
    parser.add_argument("-srfac", type=int, default=8, help="Super-resolution factor (2, 4, or 8)")
    parser.add_argument("-SRconfig_path", type=str, default=None,
                        help="Path to SR config file (.json or .txt). Default: bundled config.")
    parser.add_argument("-saveSRpatches", type=int, default=1, help="1 to save SR patches, 0 to skip")
    parser.add_argument("-saveFrameGoodCoords", type=int, default=1, help="1 to save goodCoords, 0 to skip")
    parser.add_argument("-use_gpu", type=int, default=1,
                        help="1 to use GPU-accelerated peak fitting when CUDA is available (default), "
                             "0 to use CPU-only fitting")
    args = parser.parse_args()

    run_sr_process(args.midas_dir, srfac=args.srfac, SRconfig_path=args.SRconfig_path,
                   saveSRpatches=args.saveSRpatches, saveFrameGoodCoords=args.saveFrameGoodCoords,
                   use_gpu=args.use_gpu)
