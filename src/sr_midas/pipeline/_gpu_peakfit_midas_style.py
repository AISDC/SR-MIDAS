"""GPU peak fit on SR patches using MIDAS-style methodology.

A hybrid of the existing `_gpu_peakfit.batched_adam_fit` (batched-Adam
infrastructure on GPU) and the MIDAS C/CUDA peak-fit recipe (per-region
background, moment-based initialisation, region-dependent bounds,
MIDAS `IntegratedIntensity`/`NrPixels` rules). Keeps the existing fast
GPU pipeline; adds a third `peak_fit_method = "gpu_midas_style"`
selector for cases where users want MIDAS-equivalent semantics
batched on the GPU.

Inspiration:
  * MIDAS C peak fit recipe: `FF_HEDM/src/PeaksFittingOMPZarrRefactor.c::
    fit2DPeaks` (per-region BG ∈ [0, thresh], moment-based per-peak
    Voronoi sigma init, region-dependent maxRWidth / maxEtaWidth bounds,
    R ± 1, Eta ± atan(1/R) bounds; `IntegratedIntensity[j] = Σ(model_j +
    BG · [model_j > BG])`; `NrPixels[j] = #{ model_j > BG }`).
  * midas_peakfit GPU layout: `~/MIDAS/packages/midas_peakfit/midas_peakfit/
    {model,lm,jacobian_triton}.py` -- batched LM over many regions with
    `1 + 8P` parameters per region (1 BG + 8 per peak), reparameterised
    bounds. We borrow the *parameter layout and BG handling* but keep
    Adam as the optimiser to reuse the existing torch.compile-friendly
    infrastructure here.

What this matches against the C MIDAS routine, on GPU:
  * forward: `bg + Σⱼ IMaxⱼ · (μⱼ L + (1-μⱼ) G)` with G and L factored
    in R and Eta -- algebraically identical;
  * BG is a fitted scalar per region with hard bound `[0, thresh / srfac²]`;
  * sigmas are initialised via per-peak Voronoi-partitioned weighted
    variance (batched over patches);
  * R, Eta, sigma bounds are derived from the per-patch R/Eta extents
    (matching MIDAS lines ~868-905);
  * IntegratedIntensity and NrPixels per peak use the MIDAS conditional
    `model > BG` rule.

What this still differs from MIDAS by construction (input data):
  * the residual is over SR-patch pixels (`lrsz·srfac × lrsz·srfac`) not
    a native CC region; we mask pixels below the SR-scale BG threshold
    to mimic MIDAS's CC support and to keep batched compute fast.

Output rows are 29-column MIDAS-schema rows with **BG and IMax rescaled
to native intensity units** (× `srfac²`) so they are directly comparable
to MIDAS C-routine outputs and to the existing GPU-Adam fitter (whose
IMax is at native scale via avg-pool).
"""
from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from sr_midas.pipeline._gpu_peakfit import (
    build_RE_grids,
    detect_peaks_and_init,
)


# ---------- batched moment-based sigma init (Voronoi-partitioned) ----------

def _voronoi_moment_widths_batched(R, Eta, z, peak_R, peak_Eta, n_peaks,
                                   bg_est, width_fallback):
    """Per-(patch, peak) (sigmaR, sigmaEta) from Voronoi-partitioned weighted
    variance of above-BG pixels. Vectorised over the patch batch.

    Args:
        R, Eta, z:  (B, npx) float tensors.
        peak_R, peak_Eta: (B, K) float — initial peak positions.
        n_peaks:    (B,) long, active peak count per patch.
        bg_est:     float, scalar background estimate used to gate pixels.
        width_fallback: (B,) float, per-patch fallback sigma when a peak
            attracts no above-BG pixels.

    Returns:
        sigmaR, sigmaEta: (B, K) float tensors.
    """
    B, K = peak_R.shape
    npx = R.shape[1]
    eps = 1e-9

    dR = R.unsqueeze(-1) - peak_R.unsqueeze(1)        # (B, npx, K)
    dE = Eta.unsqueeze(-1) - peak_Eta.unsqueeze(1)
    d2 = dR * dR + dE * dE

    # mask inactive peak slots so they cannot win the argmin
    active = (torch.arange(K, device=R.device).unsqueeze(0)
              < n_peaks.unsqueeze(1))                  # (B, K)
    d2 = d2.masked_fill(~active.unsqueeze(1), float("inf"))
    closest = d2.argmin(dim=-1)                        # (B, npx)

    val = (z - bg_est).clamp(min=0.0)                  # (B, npx)
    one_hot = F.one_hot(closest, num_classes=K).to(R.dtype)  # (B, npx, K)
    weight = one_hot * val.unsqueeze(-1)               # (B, npx, K)

    sumW   = weight.sum(dim=1)                         # (B, K)
    sumWR2 = (weight * dR * dR).sum(dim=1)
    sumWE2 = (weight * dE * dE).sum(dim=1)

    safe_sumW = sumW.clamp(min=eps)
    sigmaR = torch.where(sumW > eps,
                         torch.sqrt(sumWR2 / safe_sumW),
                         width_fallback.unsqueeze(-1).expand(B, K))
    sigmaE = torch.where(sumW > eps,
                         torch.sqrt(sumWE2 / safe_sumW),
                         width_fallback.unsqueeze(-1).expand(B, K))
    return sigmaR, sigmaE


# ---------- forward model with BG (per-peak so we can apply MIDAS rules) ----

def _pv_per_peak_with_grid(grid_RR, grid_EE, peak_params):
    """Evaluate per-peak pseudo-Voigt without BG.

    Args:
        grid_RR, grid_EE: (B, H, W) float.
        peak_params:      (B, K, 8) float in our internal order
            (Rpx, Eta, sGR, sGE, sLR, sLE, LGmix, IMax).
    Returns:
        (B, K, H, W) float -- per-peak PV intensity at each pixel.
    """
    Rpx = peak_params[..., 0].unsqueeze(-1).unsqueeze(-1)
    Eta = peak_params[..., 1].unsqueeze(-1).unsqueeze(-1)
    sGR = peak_params[..., 2].unsqueeze(-1).unsqueeze(-1)
    sGE = peak_params[..., 3].unsqueeze(-1).unsqueeze(-1)
    sLR = peak_params[..., 4].unsqueeze(-1).unsqueeze(-1)
    sLE = peak_params[..., 5].unsqueeze(-1).unsqueeze(-1)
    LG  = peak_params[..., 6].unsqueeze(-1).unsqueeze(-1)
    IM  = peak_params[..., 7].unsqueeze(-1).unsqueeze(-1)

    RR = grid_RR.unsqueeze(1)
    EE = grid_EE.unsqueeze(1)

    dR = RR - Rpx
    dE = EE - Eta
    G = IM * torch.exp(-0.5 * (dR * dR) / (sGR * sGR + 1e-10)
                       - 0.5 * (dE * dE) / (sGE * sGE + 1e-10))
    L = IM / ((1.0 + (dR * dR) / (sLR * sLR + 1e-10))
              * (1.0 + (dE * dE) / (sLE * sLE + 1e-10)))
    return LG * L + (1.0 - LG) * G


# ---------- bounded reparameterisation (reuse the sigmoid scheme) -----------

def _project(raw, lower, upper):
    return lower + (upper - lower) * torch.sigmoid(raw)


def _inv_project(params, lower, upper):
    eps = 1e-6
    t = ((params - lower) / (upper - lower + 1e-12)).clamp(eps, 1.0 - eps)
    return torch.log(t / (1.0 - t))


# ---------- batched Adam fit with BG and peak masking -----------------------
#
# Speed strategy mirrors `_gpu_peakfit._get_compiled_step()`: the per-step
# body is JIT-compiled with torch.compile(dynamic=True) and cached as a
# module-global. Inductor fuses the forward, the residual reduction, and
# (via the autograd graph) most of the backward into a small handful of
# CUDA kernels -- avoiding the per-iteration Python/launch overhead that
# dominated the eager version. Compile cost (~5-15 s) is paid once per
# process; steady-state per-patch wall time drops by ~20-40x on RTX A6000,
# making this routine comparable to the existing fast gpu_adam fitter.

_compiled_step_bg_fn = None


def _get_compiled_step_bg():
    """Lazy build + cache the compiled per-step body (BG + peaks)."""
    global _compiled_step_bg_fn
    if _compiled_step_bg_fn is None:
        @torch.compile(mode='default', dynamic=True)
        def _step(raw_bg, raw_peak, bg_lo, bg_hi, peak_lo, peak_hi,
                  target_masked, mask, grid_RR, grid_EE, active_b):
            """Forward + masked SSR for one Adam step.

            Inductor fuses sigmoid-reparam, per-peak PV evaluation, the
            sum-over-peaks, and the residual SSR into a small number of
            kernels. Autograd's backward then runs through the fused
            graph too, so a full iteration is ~one launch per kernel.
            """
            bg    = bg_lo + (bg_hi - bg_lo) * torch.sigmoid(raw_bg)
            peaks = peak_lo + (peak_hi - peak_lo) * torch.sigmoid(raw_peak)
            pv = _pv_per_peak_with_grid(grid_RR, grid_EE, peaks) * active_b
            model = bg.view(-1, 1, 1) + pv.sum(dim=1)
            diff  = (model - target_masked) * mask
            return (diff * diff).sum()
        _compiled_step_bg_fn = _step
    return _compiled_step_bg_fn


def _batched_adam_fit_with_bg(grid_RR, grid_EE, target, mask,
                              bg_init, bg_lower, bg_upper,
                              peak_init, peak_lower, peak_upper,
                              n_peaks_t, n_steps=20, lr=0.15,
                              use_compile=True):
    """Batched Adam over (BG + peaks) for many patches at once on GPU.

    Loss: sum of squared (model - target) over MASKED pixels (mask==1).

    Uses a torch.compile-cached step body so steady-state iterations
    avoid Python/launch overhead. Compile happens on first call and is
    reused across frames in the same process.
    """
    B, H, W = grid_RR.shape
    K = peak_init.shape[1]
    device = grid_RR.device

    raw_bg   = _inv_project(bg_init,   bg_lower,   bg_upper).clone().detach().requires_grad_(True)
    raw_peak = _inv_project(peak_init, peak_lower, peak_upper).clone().detach().requires_grad_(True)

    optim = torch.optim.Adam([raw_bg, raw_peak], lr=lr)
    target_m = target * mask

    active = (torch.arange(K, device=device).unsqueeze(0)
              < n_peaks_t.unsqueeze(1)).to(target.dtype)       # (B, K)
    active_b = active.unsqueeze(-1).unsqueeze(-1)             # (B, K, 1, 1)

    step_fn = _get_compiled_step_bg() if (use_compile and device.type == 'cuda') else None

    for _ in range(n_steps):
        optim.zero_grad(set_to_none=True)
        if step_fn is not None:
            loss = step_fn(raw_bg, raw_peak, bg_lower, bg_upper,
                           peak_lower, peak_upper, target_m, mask,
                           grid_RR, grid_EE, active_b)
        else:
            bg    = _project(raw_bg,   bg_lower,   bg_upper)
            peaks = _project(raw_peak, peak_lower, peak_upper)
            pv = _pv_per_peak_with_grid(grid_RR, grid_EE, peaks) * active_b
            model = bg.view(-1, 1, 1) + pv.sum(dim=1)
            diff  = (model - target_m) * mask
            loss  = (diff * diff).sum()
        loss.backward()
        optim.step()

    with torch.no_grad():
        bg    = _project(raw_bg,   bg_lower,   bg_upper)
        peaks = _project(raw_peak, peak_lower, peak_upper)
        pv = _pv_per_peak_with_grid(grid_RR, grid_EE, peaks) * active_b
        model = bg.view(B, 1, 1) + pv.sum(dim=1)
        diff  = (model - target_m) * mask
        per_patch_ssq = (diff * diff).sum(dim=(1, 2))
        n_active_px = mask.sum(dim=(1, 2)).clamp(min=1.0)
        per_patch_rmse = torch.sqrt(per_patch_ssq / n_active_px)

    return bg, peaks, pv, per_patch_rmse


# ---------- main entry point ------------------------------------------------

def gpu_midas_fit_frame_patches(patches_to_fit_t, patches_Y00, patches_Z00,
                                patches_exp_t, nr_pixels_in_patch,
                                patches_Isum,
                                sr_params, sr_config, srfac,
                                omega, shiftYpx, shiftZpx,
                                torch_devs, logger=None,
                                n_steps: int = 20, lr: float = 0.15,
                                use_compile: bool = True):
    """GPU MIDAS-style fit: same I/O contract as `gpu_fit_frame_patches`.

    Drop-in alternative to `_gpu_peakfit.gpu_fit_frame_patches`. Fits all
    patches in one frame in a single batched Adam loop on the GPU but
    with MIDAS-style BG / init / bounds / output rules.

    Returns:
        df_rows (list[list[float]]), n_peaks_list (list[int]), spotID (int).
    """
    n_patches = int(patches_to_fit_t.shape[0])
    if n_patches == 0:
        return [], [], 0

    # ---- config / params ----
    lrsz       = int(sr_config["lrsz"])
    Ypx_BC     = float(sr_params["Ypx_BC"])
    Zpx_BC     = float(sr_params["Zpx_BC"])
    pkargs     = sr_config["peak_find_args"]
    lr_thresh  = float(pkargs["pvfit_int_thresh"][f"SRx{srfac}"])
    min_d      = int(pkargs["min_d"][f"SRx{srfac}"])
    thresh_rel = float(pkargs["thresh_rel"][f"SRx{srfac}"])

    # SR-pixel BG cap (matches MIDAS thresh divided by srfac² to put BG
    # in SR-pixel intensity scale; cascade conserves total intensity).
    bg_thresh_sr = lr_thresh / (srfac * srfac)

    device = torch_devs
    dtype = torch.float32

    # ---- coerce inputs to torch on device ----
    def _t(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=torch.float32)
        return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)

    Y00_t = _t(patches_Y00)
    Z00_t = _t(patches_Z00)
    isum_t = _t(patches_Isum)
    nrpx_t = _t(nr_pixels_in_patch)

    patches_t = patches_to_fit_t[:, 0].contiguous()            # (B, H, W)
    H, W = patches_t.shape[1], patches_t.shape[2]
    B = patches_t.shape[0]

    # ---- coordinate grids (reused from existing module) ----
    grid_RR, grid_EE = build_RE_grids(Y00_t, Z00_t, lrsz, srfac, Ypx_BC, Zpx_BC, device)

    # ---- local-max peak detection (reuse) to get initial R, Eta, IMax ----
    init_loc, n_peaks_t, _lb_loc, _ub_loc = detect_peaks_and_init(
        patches_t, grid_RR, grid_EE, srfac,
        min_distance=min_d, threshold_rel=thresh_rel,
        edge_bound_cutoff_fac=0.0, exclude_border=True)
    K = init_loc.shape[1]
    peak_R0   = init_loc[..., 0]                                # (B, K)
    peak_Eta0 = init_loc[..., 1]
    peak_I0   = init_loc[..., 7]

    # ---- per-patch R/Eta extents and width caps (MIDAS lines 868-905) ----
    R_flat   = grid_RR.reshape(B, -1)
    E_flat   = grid_EE.reshape(B, -1)
    z_flat   = patches_t.reshape(B, -1)

    RMin = R_flat.amin(dim=1); RMax = R_flat.amax(dim=1)
    EMin = E_flat.amin(dim=1); EMax = E_flat.amax(dim=1)
    maxRWidth   = ((RMax - RMin) / 2.0 + 1.0).clamp(min=0.1)
    maxEtaWidth = ((EMax - EMin) / 2.0
                   + torch.atan(2.0 / (RMax + RMin).clamp(min=1e-9))
                       .mul_(180.0 / math.pi)).clamp(min=0.1)
    width_fb = torch.sqrt(R_flat.shape[1] / n_peaks_t.clamp(min=1).to(dtype))
    width_fb = torch.minimum(width_fb, maxRWidth)

    # ---- per-peak Voronoi-partitioned moment widths ----
    sigmaR_init, sigmaE_init = _voronoi_moment_widths_batched(
        R_flat, E_flat, z_flat, peak_R0, peak_Eta0, n_peaks_t,
        bg_est=bg_thresh_sr / 2.0,
        width_fallback=width_fb)
    sigmaR_init = torch.clamp(sigmaR_init, min=0.1).minimum(maxRWidth.unsqueeze(-1))
    sigmaE_init = torch.clamp(sigmaE_init, min=0.005).minimum(maxEtaWidth.unsqueeze(-1))

    # ---- bounds and init: BG (B,), peaks (B, K, 8) ----
    bg_init  = torch.full((B,), bg_thresh_sr / 2.0, device=device, dtype=dtype)
    bg_lower = torch.zeros((B,), device=device, dtype=dtype)
    bg_upper = torch.full((B,), max(bg_thresh_sr, 1e-6), device=device, dtype=dtype)

    peak_init  = torch.zeros((B, K, 8), device=device, dtype=dtype)
    peak_lower = torch.zeros((B, K, 8), device=device, dtype=dtype)
    peak_upper = torch.zeros((B, K, 8), device=device, dtype=dtype)

    # internal slot order: (R, Eta, sGR, sGE, sLR, sLE, LGmix, IMax)
    dEta_bounds = torch.atan(1.0 / peak_R0.clamp(min=1e-9)).mul(180.0 / math.pi)
    peak_init[..., 0] = peak_R0
    peak_init[..., 1] = peak_Eta0
    peak_init[..., 2] = sigmaR_init
    peak_init[..., 3] = sigmaE_init
    peak_init[..., 4] = sigmaR_init
    peak_init[..., 5] = sigmaE_init
    peak_init[..., 6] = 0.5
    peak_init[..., 7] = peak_I0.clamp(min=1e-3)

    peak_lower[..., 0] = peak_R0 - 1.0
    peak_lower[..., 1] = peak_Eta0 - dEta_bounds
    peak_lower[..., 2] = 0.01
    peak_lower[..., 3] = 0.005
    peak_lower[..., 4] = 0.01
    peak_lower[..., 5] = 0.005
    peak_lower[..., 6] = 0.0
    peak_lower[..., 7] = peak_I0.clamp(min=1e-3) / 2.0

    peak_upper[..., 0] = peak_R0 + 1.0
    peak_upper[..., 1] = peak_Eta0 + dEta_bounds
    peak_upper[..., 2] = 2.0 * maxRWidth.unsqueeze(-1)
    peak_upper[..., 3] = 2.0 * maxEtaWidth.unsqueeze(-1)
    peak_upper[..., 4] = 2.0 * maxRWidth.unsqueeze(-1)
    peak_upper[..., 5] = 2.0 * maxEtaWidth.unsqueeze(-1)
    peak_upper[..., 6] = 1.0
    peak_upper[..., 7] = peak_I0.clamp(min=1e-3) * 5.0 + 1e-6

    # clamp init inside bounds (defensive)
    peak_init = torch.minimum(torch.maximum(peak_init, peak_lower), peak_upper)

    # ---- MIDAS-style "CC region" mask: keep only SR pixels above BG threshold
    mask = (patches_t > bg_thresh_sr).to(dtype)
    # safety: if a patch has too few above-BG pixels, include all pixels
    enough = mask.sum(dim=(1, 2)) > (8 * n_peaks_t.to(dtype))
    fallback_mask = (patches_t > 0).to(dtype)
    mask = torch.where(enough.view(B, 1, 1).bool(), mask, fallback_mask)

    # ---- fit ----
    t0 = time.time()
    bg_fit, peaks_fit, pv_per_peak, per_patch_rmse = _batched_adam_fit_with_bg(
        grid_RR, grid_EE, patches_t, mask,
        bg_init, bg_lower, bg_upper,
        peak_init, peak_lower, peak_upper,
        n_peaks_t, n_steps=n_steps, lr=lr, use_compile=use_compile)
    if device.type == "cuda":
        torch.cuda.synchronize()
    fit_dt = time.time() - t0

    # ---- post-fit: MIDAS IntegratedIntensity / NrPixels rules ----
    with torch.no_grad():
        # per-peak model values per pixel are in SR-pixel intensity scale
        # MIDAS IntegratedIntensity[j] = Σ (per_peak_j + BG · [per_peak_j > BG])
        bg_view = bg_fit.view(B, 1, 1, 1)
        above_bg = (pv_per_peak > bg_view).to(dtype)
        contrib_sr = pv_per_peak + bg_view * above_bg
        # restrict to masked pixels (the "CC region" support)
        mask_b = mask.unsqueeze(1)
        integrated_sr = (contrib_sr * mask_b).sum(dim=(2, 3))       # (B, K)
        nrpx_sr       = (above_bg * mask_b).sum(dim=(2, 3))         # (B, K)

        # Pool per-peak model down to native resolution for IMax / argmax
        # (matches gpu_fit_frame_patches behaviour line 549-551)
        if srfac > 1:
            pooled = F.avg_pool2d(pv_per_peak.reshape(B * K, 1, H, W),
                                  kernel_size=srfac, stride=srfac,
                                  divisor_override=1)
            pv_native = pooled.reshape(B, K, H // srfac, W // srfac)
        else:
            pv_native = pv_per_peak
        Ws = pv_native.shape[-1]

        imax_native  = pv_native.amax(dim=(-2, -1))                  # (B, K)
        argflat      = pv_native.flatten(start_dim=-2).argmax(dim=-1)
        r_max        = (argflat // Ws).to(dtype)
        c_max        = (argflat %  Ws).to(dtype)
        maxY         = Y00_t.unsqueeze(1) + c_max
        maxZ         = Z00_t.unsqueeze(1) + r_max

        # rawIMax: max of native patch
        raw_imax = patches_exp_t.amax(dim=(-2, -1)).squeeze(1).unsqueeze(1).expand(B, K)

        # YCen / ZCen with shift correction
        R_fit   = peaks_fit[..., 0]
        Eta_fit = peaks_fit[..., 1]
        sGR     = peaks_fit[..., 2]
        sGE     = peaks_fit[..., 3]
        sLR     = peaks_fit[..., 4]
        sLE     = peaks_fit[..., 5]
        LGmix   = peaks_fit[..., 6]
        IMax_fit_sr = peaks_fit[..., 7]
        SigmaR   = torch.maximum(sGR, sLR)
        SigmaEta = torch.maximum(sGE, sLE)

        eta_rad = torch.deg2rad(Eta_fit)
        YCen = Ypx_BC + R_fit * torch.sin(eta_rad) + float(shiftYpx)
        ZCen = Zpx_BC + R_fit * torch.cos(eta_rad) + float(shiftZpx)
        diffY = maxY - YCen
        diffZ = maxZ - ZCen

        # rescale to native intensity scale (× srfac²) for output parity with
        # the C MIDAS routine and the existing GPU-Adam fitter
        sr_to_native = float(srfac * srfac)
        bg_native      = (bg_fit * sr_to_native).unsqueeze(1).expand(B, K)
        IMax_param     = IMax_fit_sr * sr_to_native      # fitted-param IMax in native units

        # NrPixels at native scale = (count of SR pixels above BG) / srfac²
        nrpx_native = (nrpx_sr / sr_to_native).clamp(min=1.0)

        fit_rmse_b = per_patch_rmse.unsqueeze(1).expand(B, K)
        omega_b    = torch.full((B, K), float(omega), device=device, dtype=dtype)
        nPeaks_b   = n_peaks_t.to(dtype).unsqueeze(1).expand(B, K)
        total_nrpx = nrpx_t.unsqueeze(1).expand(B, K)
        raw_isum   = isum_t.unsqueeze(1).expand(B, K)
        zeros_bk   = torch.zeros((B, K), device=device, dtype=dtype)

        active = (torch.arange(K, device=device).unsqueeze(0)
                  < n_peaks_t.unsqueeze(1))

        cols = [
            zeros_bk,            # 0 SpotID (renumbered after)
            integrated_sr,       # 1 IntegratedIntensity (SR-sum == native-sum)
            omega_b,             # 2 Omega
            YCen,                # 3 YCen
            ZCen,                # 4 ZCen
            IMax_param,          # 5 IMax (fitted amplitude, native scale)
            R_fit,               # 6 Radius
            Eta_fit,             # 7 Eta
            SigmaR,              # 8 SigmaR
            SigmaEta,            # 9 SigmaEta
            nrpx_native,         # 10 NrPixels (native count, MIDAS rule)
            total_nrpx,          # 11 TotalNrPixelsInPeakRegion
            nPeaks_b,            # 12 nPeaks
            maxY,                # 13 maxY
            maxZ,                # 14 maxZ
            diffY,               # 15 diffY
            diffZ,               # 16 diffZ
            raw_imax,            # 17 rawIMax
            zeros_bk,            # 18 returnCode (0 -- Adam doesn't report)
            fit_rmse_b,          # 19 retVal
            bg_native,           # 20 BG (fitted, native scale)
            sGR,                 # 21 SigmaGR
            sLR,                 # 22 SigmaLR
            sGE,                 # 23 SigmaGEta
            sLE,                 # 24 SigmaLEta
            LGmix,               # 25 MU
            raw_isum,            # 26 RawSumIntensity
            zeros_bk,            # 27 maskTouched
            fit_rmse_b,          # 28 FitRMSE
        ]
        rows_bk    = torch.stack(cols, dim=-1)
        rows_flat  = rows_bk.reshape(B * K, 29)
        active_flat = active.reshape(B * K)
        valid_rows = rows_flat[active_flat]
        total_peaks = valid_rows.shape[0]

        if total_peaks > 0:
            valid_rows[:, 0] = torch.arange(1, total_peaks + 1,
                                            device=device, dtype=valid_rows.dtype)

    rows_np = valid_rows.detach().cpu().numpy()
    df_rows = [list(row) for row in rows_np]
    n_peaks_list = n_peaks_t.detach().cpu().numpy().astype(int).tolist()

    if logger is not None:
        logger.info(f"\t| gpu_midas_style fit: {n_patches} patches, "
                    f"{total_peaks} peaks, {fit_dt:.2f} s "
                    f"({1000*fit_dt/max(n_patches,1):.2f} ms/patch); "
                    f"BG range [{float(bg_native.min()):.3f}, "
                    f"{float(bg_native.max()):.3f}], "
                    f"RMSE p95={float(per_patch_rmse.quantile(0.95)):.4f}")

    return df_rows, n_peaks_list, total_peaks
