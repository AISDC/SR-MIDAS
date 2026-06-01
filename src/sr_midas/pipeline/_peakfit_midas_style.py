"""MIDAS-methodology pseudo-Voigt fit on SR-MIDAS super-resolved patches.

Alternative to `_gpu_peakfit.gpu_fit_frame_patches`. Faithfully ports the
fitting recipe from MIDAS's `FF_HEDM/src/PeaksFittingOMPZarrRefactor.c`
function `fit2DPeaks` (lines ~850-1080), adapted to consume the SR-MIDAS
SR-predicted patches instead of raw detector connected-component regions.

What this matches against the canonical MIDAS routine:
  * forward model: bg + sum_j IMax_j * (Mu_j * L + (1 - Mu_j) * G), with
    G and L factored in R and Eta exactly as in MIDAS;
  * BG is a fitted scalar per region, bounded [0, thresh];
  * sigmas (sGR, sLR, sGE, sLE) initialised via per-peak moment-based
    weighted variance with Voronoi partitioning over the region pixels;
  * R-, Eta-, sigma-bounds are region-dependent (RMin/RMax, EtaMin/EtaMax,
    maxRWidth, maxEtaWidth) instead of hard-coded constants;
  * optimizer: scipy `minimize(method="Nelder-Mead")` (closest match to
    MIDAS's NLopt LN_NELDERMEAD; both are downhill-simplex; iteration
    sequences differ but converge to the same minimum on smooth problems);
  * IntegratedIntensity[j] = sum_pixel (model_j + BG * 1[model_j > BG])
    -- adds the background back under each peak's footprint;
  * NrPixels[j] = count_pixel of pixels where the j-th peak's per-peak
    model exceeds BG.

What this differs from MIDAS by construction (input data is different):
  * the fit operates on the SR-predicted patch (lrsz*srfac x lrsz*srfac
    grid, with intensity per-patch rescaled so sum == native intensity
    sum) instead of raw detector pixels inside a connected-component
    region. BG bounds and pixel-count thresholds are scaled by srfac**2
    so that totals stay comparable to native-scale MIDAS quantities.

Performance: runs on CPU per-patch with scipy `minimize`. Slower than
`gpu_fit_frame_patches` -- intended for verification / methodology
comparison, not as the default production fitter. Use the
`peak_fit_method="midas_style"` selector in `sr_process.run_sr_process`
to engage it.
"""
from __future__ import annotations

import math
import time
from typing import Sequence

import numpy as np
import torch
from scipy.optimize import minimize


# ---------- coordinate grid (CPU, per-patch) --------------------------------

def _r_eta_grids_for_patch(Y00: float, Z00: float, lrsz: int, srfac: int,
                            Ypx_BC: float, Zpx_BC: float):
    """Build (R, Eta) grids for one SR patch at SR resolution.

    Mirrors `_gpu_peakfit.build_RE_grids` but per-patch on CPU.

    Returns:
        Rs, Etas: each (lrsz*srfac, lrsz*srfac) float64 numpy arrays.
    """
    dpx = 1.0 / srfac
    n_px = int(lrsz * srfac)
    offsets = np.arange(n_px, dtype=np.float64) * dpx
    Ypx = Y00 + offsets
    Zpx = Z00 + offsets
    grid_YY = np.broadcast_to(Ypx[None, :], (n_px, n_px)).copy()
    grid_ZZ = np.broadcast_to(Zpx[:, None], (n_px, n_px)).copy()
    dY = Ypx_BC - grid_YY
    dZ = Zpx_BC - grid_ZZ
    R = np.sqrt(dY * dY + dZ * dZ)
    cos_eta = np.clip((grid_ZZ - Zpx_BC) / np.maximum(R, 1e-12), -1.0, 1.0)
    Eta = np.rad2deg(np.arccos(cos_eta))
    sign_y = np.sign(grid_YY - Ypx_BC)
    sign_y[sign_y == 0] = 1.0
    Eta = Eta * sign_y
    return R, Eta


# ---------- forward model (CPU, vectorised over pixels in one region) -------

def _per_peak_pv(R_flat: np.ndarray, Eta_flat: np.ndarray, peak_params: np.ndarray):
    """Evaluate per-peak pseudo-Voigt intensity at every pixel.

    Args:
        R_flat, Eta_flat: (npx,) float64.
        peak_params:      (nPeaks, 8) float64 in MIDAS order
            (IMax, R, Eta, Mu, sGR, sLR, sGE, sLE).
    Returns:
        (nPeaks, npx) float64; sum over axis 0 is the model without BG.
    """
    IMax = peak_params[:, 0:1]
    Rp   = peak_params[:, 1:2]
    Etap = peak_params[:, 2:3]
    Mu   = peak_params[:, 3:4]
    sGR  = peak_params[:, 4:5]
    sLR  = peak_params[:, 5:6]
    sGE  = peak_params[:, 6:7]
    sLE  = peak_params[:, 7:8]

    dR = R_flat[None, :] - Rp
    dE = Eta_flat[None, :] - Etap
    R2 = dR * dR
    E2 = dE * dE
    G = np.exp(-0.5 * (R2 / (sGR * sGR) + E2 / (sGE * sGE)))
    L = 1.0 / ((1.0 + R2 / (sLR * sLR)) * (1.0 + E2 / (sLE * sLE)))
    return IMax * (Mu * L + (1.0 - Mu) * G)


def _residual_ssq(x: np.ndarray, R_flat: np.ndarray, Eta_flat: np.ndarray,
                  z_flat: np.ndarray, n_peaks: int):
    """SSR for MIDAS-style fit: x = [BG, (IMax,R,Eta,Mu,sGR,sLR,sGE,sLE)*nPeaks].

    Direct port of `peakFittingObjectiveFunction` in PeaksFittingOMPZarrRefactor.c
    (lines ~709-776). Same loss, same parameter layout.
    """
    bg = x[0]
    peak_params = x[1:].reshape(n_peaks, 8)
    per_peak = _per_peak_pv(R_flat, Eta_flat, peak_params)
    model = bg + per_peak.sum(axis=0)
    diff = model - z_flat
    return float(np.dot(diff, diff))


# ---------- per-peak quantities (matches MIDAS calculateIntegratedIntensity)

def _midas_intensity_and_nrpx(R_flat, Eta_flat, peak_params, bg):
    """Compute (IntegratedIntensity, NrPixels) per peak the MIDAS way.

    Port of `calculateIntegratedIntensity` (PeaksFittingOMPZarrRefactor.c
    ~782-845): for each (peak j, pixel i), if the peak's model intensity
    exceeds BG at that pixel, add (model + BG) and increment NrPixels.
    """
    per_peak = _per_peak_pv(R_flat, Eta_flat, peak_params)
    above_bg = per_peak > bg
    contrib = per_peak + bg * above_bg.astype(per_peak.dtype)
    integ = contrib.sum(axis=1)
    nrpx = above_bg.sum(axis=1)
    return integ.astype(np.float64), nrpx.astype(np.int64)


# ---------- moment-based per-peak width init (Voronoi-partitioned) ----------

def _voronoi_moment_widths(R_flat, Eta_flat, z_flat, peakRs, peakEtas, bg_est,
                            width_fallback):
    """Per-peak (sigmaR, sigmaEta) from weighted-variance over Voronoi cells.

    Mirrors PeaksFittingOMPZarrRefactor.c lines ~927-985: each pixel is
    assigned to the closest peak in (R, Eta) (Euclidean), pixels with
    (z - bg) <= 0 are skipped, then per-peak weighted variance gives the
    initial sigma. Falls back to `width_fallback` if a peak attracts no
    above-BG pixels.
    """
    n_peaks = len(peakRs)
    if n_peaks == 1:
        # Single peak owns every above-bg pixel
        val = z_flat - bg_est
        mask = val > 0
        if not mask.any():
            return [width_fallback] * n_peaks, [width_fallback] * n_peaks
        v = val[mask]
        dR = R_flat[mask] - peakRs[0]
        dE = Eta_flat[mask] - peakEtas[0]
        sW = float(v.sum())
        sR = math.sqrt(float((v * dR * dR).sum()) / sW) if sW > 0 else width_fallback
        sE = math.sqrt(float((v * dE * dE).sum()) / sW) if sW > 0 else width_fallback
        return [sR], [sE]

    # multi-peak: distance to each peak in (R, Eta)
    dR_all = R_flat[None, :] - np.asarray(peakRs)[:, None]   # (nP, npx)
    dE_all = Eta_flat[None, :] - np.asarray(peakEtas)[:, None]
    d2 = dR_all * dR_all + dE_all * dE_all
    closest = d2.argmin(axis=0)                              # (npx,)
    val = z_flat - bg_est
    mask = val > 0
    sigmaR = [width_fallback] * n_peaks
    sigmaE = [width_fallback] * n_peaks
    for j in range(n_peaks):
        sel = mask & (closest == j)
        if not sel.any():
            continue
        v = val[sel]
        dRj = R_flat[sel] - peakRs[j]
        dEj = Eta_flat[sel] - peakEtas[j]
        sW = float(v.sum())
        if sW <= 0:
            continue
        sigmaR[j] = math.sqrt(float((v * dRj * dRj).sum()) / sW)
        sigmaE[j] = math.sqrt(float((v * dEj * dEj).sum()) / sW)
    return sigmaR, sigmaE


# ---------- core: fit one SR patch the MIDAS way ----------------------------

def _fit_one_patch(R: np.ndarray, Eta: np.ndarray, z: np.ndarray,
                   peak_R0: list, peak_Eta0: list, peak_IMax0: list,
                   bg_thresh: float, max_evals_per_peak: int = 2000,
                   max_time_s: float = 30.0,
                   ftol: float = 1e-5, xtol: float = 1e-5):
    """Run the MIDAS fit on a single SR patch.

    Args:
        R, Eta, z: (H, W) float64 — pixel coords (native units) and
            SR-predicted intensities for the patch.
        peak_R0, peak_Eta0, peak_IMax0: per-peak initial radial position,
            azimuthal position (deg), and max intensity (already in SR-pixel
            intensity scale).
        bg_thresh: upper bound for fitted BG. For SR-MIDAS this is
            `(ring_pvfit_int_thresh) / srfac**2` to put BG in the same
            scale as the SR-patch pixel values.

    Returns:
        dict with keys: bg, peak_params (nP,8 in MIDAS order), nrPixels,
            integratedIntensity, fitRMSE, returnCode, retVal.
    """
    H, W = R.shape
    R_flat = R.ravel()
    Eta_flat = Eta.ravel()
    z_flat = z.ravel().astype(np.float64)
    npx = R_flat.size
    n_peaks = len(peak_R0)

    # region extents (MIDAS lines 868-905)
    RMin, RMax = float(R_flat.min()), float(R_flat.max())
    EMin, EMax = float(Eta_flat.min()), float(Eta_flat.max())
    maxRWidth   = max(0.1, (RMax - RMin) / 2.0 + 1.0)
    # MIDAS adjusts maxEtaWidth and subtracts 180 for wraparound regions
    maxEtaWidth = (EMax - EMin) / 2.0 + math.degrees(math.atan(2.0 / max(RMax + RMin, 1e-9)))
    if (EMax - EMin) > 180.0:
        maxEtaWidth -= 180.0
    maxEtaWidth = max(0.1, maxEtaWidth)

    # fallback uniform width estimate
    width = math.sqrt(npx / max(n_peaks, 1))
    if width > maxRWidth:
        width = maxRWidth

    # initial BG (MIDAS line 863): thresh / 2
    bg_init = bg_thresh / 2.0

    # moment-based per-peak widths
    sigmaR_init, sigmaE_init = _voronoi_moment_widths(
        R_flat, Eta_flat, z_flat, peak_R0, peak_Eta0, bg_init, width)

    # build initial-x and bounds vectors (MIDAS lines 859-1018)
    x0 = np.empty(1 + 8 * n_peaks, dtype=np.float64)
    lb = np.empty_like(x0)
    ub = np.empty_like(x0)

    x0[0] = bg_init
    lb[0] = 0.0
    ub[0] = max(bg_thresh, bg_init * 2.0 + 1e-6)  # ensure ub > lb

    for j in range(n_peaks):
        b = 1 + 8 * j
        IMax_j  = float(peak_IMax0[j])
        peakR   = float(peak_R0[j])
        peakEta = float(peak_Eta0[j])
        dEta    = math.degrees(math.atan(1.0 / max(peakR, 1e-9)))

        sGR = max(0.1, min(maxRWidth, float(sigmaR_init[j])))
        sGE = max(0.005, min(maxEtaWidth, float(sigmaE_init[j])))

        # init: IMax, R, Eta, Mu, sGR, sLR, sGE, sLE
        x0[b + 0] = IMax_j
        x0[b + 1] = peakR
        x0[b + 2] = peakEta
        x0[b + 3] = 0.5
        x0[b + 4] = sGR
        x0[b + 5] = sGR
        x0[b + 6] = sGE
        x0[b + 7] = sGE

        # bounds (MIDAS lines 998-1017)
        lb[b + 0] = IMax_j / 2.0
        ub[b + 0] = IMax_j * 5.0 + 1e-6
        lb[b + 1] = peakR - 1.0
        ub[b + 1] = peakR + 1.0
        lb[b + 2] = peakEta - dEta
        ub[b + 2] = peakEta + dEta
        lb[b + 3] = 0.0
        ub[b + 3] = 1.0
        lb[b + 4] = 0.01
        ub[b + 4] = 2.0 * maxRWidth
        lb[b + 5] = 0.01
        ub[b + 5] = 2.0 * maxRWidth
        lb[b + 6] = 0.005
        ub[b + 6] = 2.0 * maxEtaWidth
        lb[b + 7] = 0.005
        ub[b + 7] = 2.0 * maxEtaWidth

    # clamp x0 inside [lb, ub]
    x0 = np.clip(x0, lb, ub)

    bounds = list(zip(lb.tolist(), ub.tolist()))

    args = (R_flat, Eta_flat, z_flat, n_peaks)
    try:
        res = minimize(
            _residual_ssq, x0, args=args,
            method="Nelder-Mead", bounds=bounds,
            options={
                "xatol": xtol, "fatol": ftol,
                "maxiter": max_evals_per_peak * n_peaks,
                "maxfev":  max_evals_per_peak * n_peaks,
                "adaptive": True,
            },
        )
        x_opt = np.clip(res.x, lb, ub)  # belt-and-braces
        rc = 0 if res.success else 1
        ssq = float(res.fun)
    except Exception:
        x_opt = x0
        rc = -1
        ssq = float(_residual_ssq(x0, *args))

    bg = float(x_opt[0])
    peak_params = x_opt[1:].reshape(n_peaks, 8)
    fit_rmse = math.sqrt(ssq / max(npx, 1))

    integ, nrpx = _midas_intensity_and_nrpx(R_flat, Eta_flat, peak_params, bg)

    return {
        "bg": bg,
        "peak_params": peak_params,  # (nP, 8) MIDAS order
        "nrPixels": nrpx,
        "integratedIntensity": integ,
        "fitRMSE": fit_rmse,
        "returnCode": rc,
        "retVal": fit_rmse,
        "ssq": ssq,
    }


# ---------- frame-level orchestrator ----------------------------------------

def midas_style_fit_frame_patches(patches_to_fit_t, patches_Y00, patches_Z00,
                                  patches_exp_t, nr_pixels_in_patch,
                                  patches_Isum,
                                  sr_params, sr_config, srfac,
                                  omega, shiftYpx, shiftZpx,
                                  torch_devs, logger=None):
    """Drop-in alternative to `gpu_fit_frame_patches` using MIDAS methodology.

    Same input/output contract as the GPU Adam routine so it slots into the
    same per-frame pipeline in `sr_process.run_sr_process`. Per-patch fits
    run sequentially on CPU via scipy; for production-sized frames this is
    significantly slower than the GPU path -- intended for verification.

    Returns:
        df_rows: list[list[float]] with 29 columns each, matching MIDAS CSV.
        n_peaks_list: list[int], peaks detected per patch.
        spotID: int, total fitted peaks across the frame.
    """
    n_patches = int(patches_to_fit_t.shape[0])
    if n_patches == 0:
        return [], [], 0

    # ---- pull config bits we need ----
    lrsz       = int(sr_config["lrsz"])
    Ypx_BC     = float(sr_params["Ypx_BC"])
    Zpx_BC     = float(sr_params["Zpx_BC"])
    pkargs     = sr_config["peak_find_args"]
    lr_thresh  = float(pkargs["pvfit_int_thresh"][f"SRx{srfac}"])
    min_d      = int(pkargs["min_d"][f"SRx{srfac}"])
    thresh_rel = float(pkargs["thresh_rel"][f"SRx{srfac}"])

    # SR-pixel-equivalent BG cap: native threshold divided by srfac^2 so it
    # is in the same intensity scale as the SR-predicted pixel values
    # (cascade preserves total intensity per patch by construction).
    bg_thresh_sr = lr_thresh / (srfac * srfac)

    # ---- move per-patch tensors to CPU numpy once ----
    patches_np    = patches_to_fit_t[:, 0].detach().cpu().numpy().astype(np.float64)
    patches_exp_np = patches_exp_t.detach().cpu().numpy().astype(np.float64)
    Y00_np = np.asarray(patches_Y00 if not isinstance(patches_Y00, torch.Tensor)
                        else patches_Y00.detach().cpu().numpy(), dtype=np.float64)
    Z00_np = np.asarray(patches_Z00 if not isinstance(patches_Z00, torch.Tensor)
                        else patches_Z00.detach().cpu().numpy(), dtype=np.float64)
    nrpx_native = np.asarray(nr_pixels_in_patch if not isinstance(nr_pixels_in_patch, torch.Tensor)
                              else nr_pixels_in_patch.detach().cpu().numpy(), dtype=np.float64)
    isum_native = np.asarray(patches_Isum if not isinstance(patches_Isum, torch.Tensor)
                              else patches_Isum.detach().cpu().numpy(), dtype=np.float64)

    # ---- peak detection per patch via skimage peak_local_max (matches the
    #      semantics MIDAS uses for regional maxima) ----
    from skimage.feature import peak_local_max

    df_rows = []
    n_peaks_list = []
    spotID = 0
    t_start = time.time()

    for pi in range(n_patches):
        patch = patches_np[pi]            # (H, W) SR-resolution
        H, W = patch.shape
        Y00 = Y00_np[pi]
        Z00 = Z00_np[pi]

        # local-max peak detection on the SR patch
        loc = peak_local_max(patch, min_distance=min_d, threshold_rel=thresh_rel,
                             exclude_border=True)
        if loc.shape[0] == 0:
            argflat = int(patch.argmax())
            loc = np.array([[argflat // W, argflat % W]])

        # patch-local (row, col) -> native (Y, Z) coords
        # row index -> Z; col index -> Y (matches build_RE_grids convention)
        peak_Y = Y00 + loc[:, 1].astype(np.float64) / srfac
        peak_Z = Z00 + loc[:, 0].astype(np.float64) / srfac
        peak_R    = np.sqrt((Ypx_BC - peak_Y) ** 2 + (Zpx_BC - peak_Z) ** 2)
        cos_e     = np.clip((peak_Z - Zpx_BC) / np.maximum(peak_R, 1e-12), -1.0, 1.0)
        peak_Eta  = np.rad2deg(np.arccos(cos_e))
        sgn       = np.sign(peak_Y - Ypx_BC); sgn[sgn == 0] = 1.0
        peak_Eta *= sgn
        peak_IMax = patch[loc[:, 0], loc[:, 1]].astype(np.float64)

        # build grids and fit. To mirror MIDAS's connected-component fit
        # domain (and to keep wall time reasonable), restrict the residual
        # to pixels above the SR-scale background threshold -- pixels far
        # below threshold do not contribute information about peak shape
        # in the C MIDAS routine either (they are simply not part of any
        # connected component).
        R, Eta = _r_eta_grids_for_patch(Y00, Z00, lrsz, srfac, Ypx_BC, Zpx_BC)
        mask = patch > bg_thresh_sr
        if mask.sum() < 8 * peak_R.size:  # too few pixels to fit
            mask = patch > 0.0
        R_use   = R[mask]
        Eta_use = Eta[mask]
        z_use   = patch[mask]
        # Reshape into pseudo-(H, W) just to satisfy _fit_one_patch's API
        # (it only uses .ravel() internally so any 2D shape works).
        # We pass it as a 1xN 'patch' so .ravel() preserves the masked set.
        out = _fit_one_patch(
            R_use.reshape(1, -1), Eta_use.reshape(1, -1), z_use.reshape(1, -1),
            peak_R0=peak_R.tolist(),
            peak_Eta0=peak_Eta.tolist(),
            peak_IMax0=peak_IMax.tolist(),
            bg_thresh=bg_thresh_sr,
        )

        # ---- post-fit per-peak quantities ----
        pp = out["peak_params"]  # (nP, 8) MIDAS order (IMax, R, Eta, Mu, sGR, sLR, sGE, sLE)
        n_pk = pp.shape[0]
        n_peaks_list.append(int(n_pk))

        # Rescale BG and per-peak IMax from SR-pixel intensity scale to
        # native intensity scale (× srfac**2) so output values are
        # comparable to MIDAS native-pixel outputs and to the existing
        # GPU-Adam fitter (whose IMax is already at native scale via
        # avg-pool). IntegratedIntensity is already a SUM, so by
        # cascade-level intensity conservation it is unit-equivalent
        # across SR and native scale -- no rescale needed.
        sr_to_native = float(srfac * srfac)
        bg          = out["bg"] * sr_to_native
        fit_rmse    = out["fitRMSE"]
        return_code = out["returnCode"]
        ret_val     = out["retVal"]
        integ_pp    = out["integratedIntensity"]   # SR sum == native sum
        nrpx_pp_sr  = out["nrPixels"]              # count of SR pixels
        # Convert fitted per-peak IMax (param column 0) to native scale too
        pp[:, 0] = pp[:, 0] * sr_to_native

        # NrPixels at native scale (MIDAS reports native-pixel counts).
        # Each native pixel = srfac*srfac SR pixels, so divide and round.
        nrpx_pp_native = np.maximum(1, (nrpx_pp_sr // (srfac * srfac)).astype(np.int64))

        # rawIMax: max of NATIVE patch (matches GPU Adam routine line 599)
        raw_imax = float(patches_exp_np[pi, 0].max())

        for j in range(n_pk):
            IMax_j, R_j, Eta_j, Mu_j, sGR, sLR, sGE, sLE = pp[j]
            SigmaR   = max(sGR, sLR)
            SigmaEta = max(sGE, sLE)
            eta_rad  = math.radians(Eta_j)
            # Apply same shift correction as the existing routine
            YCen = Ypx_BC + R_j * math.sin(eta_rad) + float(shiftYpx)
            ZCen = Zpx_BC + R_j * math.cos(eta_rad) + float(shiftZpx)

            # maxY/maxZ: position of the peak's argmax pixel mapped to NATIVE coords
            # (same as the GPU routine's pooled-argmax in native coords).
            # Approximation: use peak (R, Eta) -> (Y, Z) rounded to native.
            maxY = round(YCen)
            maxZ = round(ZCen)
            diffY = float(maxY) - YCen
            diffZ = float(maxZ) - ZCen

            row = [
                float(spotID + j + 1),          # 0 SpotID (re-numbered after frame done)
                float(integ_pp[j]),             # 1 IntegratedIntensity (MIDAS-style w/ BG inclusion)
                float(omega),                   # 2 Omega
                float(YCen),                    # 3 YCen
                float(ZCen),                    # 4 ZCen
                float(IMax_j),                  # 5 IMax (fitted amplitude)
                float(R_j),                     # 6 Radius
                float(Eta_j),                   # 7 Eta
                float(SigmaR),                  # 8 SigmaR  = max(sGR, sLR)
                float(SigmaEta),                # 9 SigmaEta = max(sGE, sLE)
                float(nrpx_pp_native[j]),       # 10 NrPixels (native-scale, MIDAS-style)
                float(nrpx_native[pi]),         # 11 TotalNrPixelsInPeakRegion (native)
                float(n_pk),                    # 12 nPeaks
                float(maxY),                    # 13 maxY
                float(maxZ),                    # 14 maxZ
                float(diffY),                   # 15 diffY
                float(diffZ),                   # 16 diffZ
                raw_imax,                       # 17 rawIMax
                float(return_code),             # 18 returnCode (0 OK, 1 not converged, -1 except)
                float(ret_val),                 # 19 retVal (== fit RMSE)
                float(bg),                      # 20 BG  <-- FITTED, not zero
                float(sGR),                     # 21 SigmaGR
                float(sLR),                     # 22 SigmaLR
                float(sGE),                     # 23 SigmaGEta
                float(sLE),                     # 24 SigmaLEta
                float(Mu_j),                    # 25 MU
                float(isum_native[pi]),         # 26 RawSumIntensity
                0.0,                            # 27 maskTouched (not tracked here)
                float(fit_rmse),                # 28 FitRMSE
            ]
            df_rows.append(row)

        spotID += n_pk

    # Re-number SpotIDs sequentially over the frame
    for k, row in enumerate(df_rows):
        row[0] = float(k + 1)

    if logger is not None:
        dt = time.time() - t_start
        logger.info(f"\t| midas_style fit: {n_patches} patches, "
                    f"{len(df_rows)} peaks, {dt:.2f} s "
                    f"({1000*dt/max(n_patches,1):.1f} ms/patch)")

    return df_rows, n_peaks_list, len(df_rows)
