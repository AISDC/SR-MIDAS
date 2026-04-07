"""GPU-accelerated 2D pseudo-Voigt peak fitting for the SR-MIDAS pipeline.

Provides batched GPU peak detection and Adam-based pseudo-Voigt fitting
as an alternative to the per-patch scipy.curve_fit (TRF) in _patch_ops.py.
When a CUDA GPU is available and use_gpu=1, sr_process.py routes peak
fitting through these functions instead of the sequential CPU path.

Key functions:
    build_RE_grids       - Batched R-Eta coordinate grid construction
    detect_peaks_and_init - GPU peak detection via max-pool + plateau suppression
    batched_adam_fit      - Batched bounded Adam optimizer with torch.compile
    gpu_fit_frame_patches - High-level entry point called from sr_process.py
"""

import numpy as np
import torch
import torch.nn.functional as F


# ─── Coordinate transforms (batched) ────────────────────────────────────────

def build_RE_grids(Y00, Z00, lrsz, srfac, Ypx_BC, Zpx_BC, device):
    """Build R-Eta coordinate grids for a batch of patches."""
    B = Y00.shape[0]
    dpx = 1.0 / srfac
    n_px = int(lrsz * srfac)
    offsets = torch.arange(n_px, device=device, dtype=torch.float32) * dpx

    Ypx = Y00.unsqueeze(1) + offsets.unsqueeze(0)
    Zpx = Z00.unsqueeze(1) + offsets.unsqueeze(0)

    grid_YY = Ypx.unsqueeze(1).expand(B, n_px, n_px)
    grid_ZZ = Zpx.unsqueeze(2).expand(B, n_px, n_px)

    dY = Ypx_BC - grid_YY
    dZ = Zpx_BC - grid_ZZ
    grid_RR = torch.sqrt(dY * dY + dZ * dZ)

    cos_eta = ((grid_ZZ - Zpx_BC) / grid_RR).clamp(-1.0, 1.0)
    grid_EE = torch.rad2deg(torch.acos(cos_eta))
    sign_y = torch.sign(grid_YY - Ypx_BC)
    sign_y = torch.where(sign_y == 0, torch.ones_like(sign_y), sign_y)
    grid_EE = grid_EE * sign_y

    return grid_RR, grid_EE


# ─── Pseudo-Voigt 2D model (batched, differentiable) ────────────────────────

def pseudo_voigt_2d_batch(grid_RR, grid_EE, params, n_peaks_per_patch,
                          threshold=0.0):
    """Evaluate multi-peak 2D pseudo-Voigt for a batch of patches."""
    B, H, W = grid_RR.shape
    max_peaks = params.shape[1]

    Rpx = params[:, :, 0].unsqueeze(-1).unsqueeze(-1)
    Eta = params[:, :, 1].unsqueeze(-1).unsqueeze(-1)
    sGR = params[:, :, 2].unsqueeze(-1).unsqueeze(-1)
    sGE = params[:, :, 3].unsqueeze(-1).unsqueeze(-1)
    sLR = params[:, :, 4].unsqueeze(-1).unsqueeze(-1)
    sLE = params[:, :, 5].unsqueeze(-1).unsqueeze(-1)
    LG  = params[:, :, 6].unsqueeze(-1).unsqueeze(-1)
    IM  = params[:, :, 7].unsqueeze(-1).unsqueeze(-1)

    RR = grid_RR.unsqueeze(1)
    EE = grid_EE.unsqueeze(1)

    dR_G = (RR - Rpx) / sGR
    dE_G = (EE - Eta) / sGE
    G = IM * torch.exp(-0.5 * dR_G * dR_G - 0.5 * dE_G * dE_G)

    dR_L = (RR - Rpx) / sLR
    dE_L = (EE - Eta) / sLE
    L = IM / ((1.0 + dR_L * dR_L) * (1.0 + dE_L * dE_L))

    PV = LG * L + (1.0 - LG) * G

    peak_mask = (torch.arange(max_peaks, device=params.device).unsqueeze(0)
                 < n_peaks_per_patch.unsqueeze(1))
    PV = PV * peak_mask.unsqueeze(-1).unsqueeze(-1).float()

    patches = PV.sum(dim=1)

    if threshold > 0:
        patches = torch.where(patches >= threshold, patches,
                              torch.zeros_like(patches))
    return patches


# ─── Bounded optimization via sigmoid ────────────────────────────────────────

def _project(raw, lower, upper):
    return lower + (upper - lower) * torch.sigmoid(raw)

def _inv_project(params, lower, upper):
    eps = 1e-6
    t = ((params - lower) / (upper - lower + 1e-12)).clamp(eps, 1.0 - eps)
    return torch.log(t / (1.0 - t))


# ─── Compiled optimization step ─────────────────────────────────────────────

_compiled_step_fn = None

def _get_compiled_step():
    """Get the compiled step function (created once, handles dynamic shapes)."""
    global _compiled_step_fn
    if _compiled_step_fn is None:
        @torch.compile(mode='default', dynamic=True)
        def _step(raw, target_flat, lb, ub, n_pk, grid_RR, grid_EE):
            params = _project(raw, lb, ub)
            pred = pseudo_voigt_2d_batch(grid_RR, grid_EE, params, n_pk)
            residuals = pred.reshape(pred.shape[0], -1) - target_flat
            return (residuals * residuals).sum()
        _compiled_step_fn = _step
    return _compiled_step_fn


# ─── Core optimizer ──────────────────────────────────────────────────────────

def batched_adam_fit(grid_RR, grid_EE, target, init_params,
                     n_peaks, lower, upper,
                     n_steps=20, lr=0.15, threshold=0.0,
                     use_compile=True):
    """Fit all patches in parallel using Adam + autograd + torch.compile."""
    B = target.shape[0]
    H, W = target.shape[1], target.shape[2]
    device = grid_RR.device

    raw = _inv_project(init_params, lower, upper).clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=lr)
    target_flat = target.reshape(B, -1)

    if use_compile and device.type == 'cuda':
        step_fn = _get_compiled_step()
    else:
        step_fn = None

    for step in range(n_steps):
        optimizer.zero_grad(set_to_none=True)

        if step_fn is not None:
            loss = step_fn(raw, target_flat, lower, upper, n_peaks, grid_RR, grid_EE)
        else:
            params = _project(raw, lower, upper)
            pred = pseudo_voigt_2d_batch(grid_RR, grid_EE, params, n_peaks, threshold)
            residuals = pred.reshape(B, -1) - target_flat
            loss = (residuals * residuals).sum()

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        params = _project(raw, lower, upper)
        pred = pseudo_voigt_2d_batch(grid_RR, grid_EE, params, n_peaks, threshold)
        costs = ((pred.reshape(B, -1) - target_flat) ** 2).mean(dim=1)

    return params, costs


# ─── Peak detection (fully vectorized) ──────────────────────────────────────

def detect_peaks_and_init(patches, grid_RR, grid_EE, srfac,
                          min_distance=3, threshold_rel=0.1,
                          edge_bound_cutoff_fac=0.0,
                          exclude_border=True):
    """Detect peaks via GPU max-pool — fully vectorized.

    Matches peak_local_max behavior: exclude_border drops peaks within
    min_distance of the boundary; plateau suppression ensures one pixel
    per equal-intensity plateau.
    """
    B, H, W = patches.shape
    device = patches.device
    max_peaks = 5

    kernel = 2 * min_distance + 1
    local_max = F.max_pool2d(patches.unsqueeze(1), kernel_size=kernel,
                              stride=1, padding=min_distance).squeeze(1)

    patch_max = patches.reshape(B, -1).max(dim=1).values
    thresh = threshold_rel * patch_max.unsqueeze(1).unsqueeze(2)
    is_peak = (patches == local_max) & (patches > thresh) & (patches > 0)

    # Plateau suppression: keep one pixel per equal-intensity plateau
    suppress = torch.zeros_like(is_peak)
    suppress[:, :, 1:]  |= (patches[:, :, 1:]  == patches[:, :, :-1])  & is_peak[:, :, :-1]
    suppress[:, 1:, :]  |= (patches[:, 1:, :]  == patches[:, :-1, :])  & is_peak[:, :-1, :]
    suppress[:, 1:, 1:] |= (patches[:, 1:, 1:] == patches[:, :-1, :-1]) & is_peak[:, :-1, :-1]
    suppress[:, 1:, :-1]|= (patches[:, 1:, :-1] == patches[:, :-1, 1:]) & is_peak[:, :-1, 1:]
    is_peak = is_peak & ~suppress

    # Border exclusion
    border = 0
    if exclude_border and min_distance > 0:
        border = min_distance
    if edge_bound_cutoff_fac > 0:
        border = max(border, int(np.ceil(edge_bound_cutoff_fac * srfac)))

    if border > 0 and border < min(H, W):
        edge_mask = torch.ones(H, W, device=device, dtype=torch.bool)
        edge_mask[:border, :] = False
        edge_mask[-border:, :] = False
        edge_mask[:, :border] = False
        edge_mask[:, -border:] = False
        is_peak = is_peak & edge_mask.unsqueeze(0)

    flat = patches.reshape(B, -1)
    flat_mask = is_peak.reshape(B, -1).float()
    masked = flat * flat_mask + (-1e9) * (1 - flat_mask)

    topk_vals, topk_idx = torch.topk(masked, k=max_peaks, dim=1)
    n_peaks = (topk_vals > 0).sum(dim=1).clamp(min=1).long()

    no_peaks = (topk_vals[:, 0] <= 0)
    if no_peaks.any():
        fb = flat.argmax(dim=1)
        topk_idx[no_peaks, 0] = fb[no_peaks]
        topk_vals[no_peaks, 0] = flat[no_peaks].gather(1, fb[no_peaks].unsqueeze(1)).squeeze(1)

    peak_R = grid_RR.reshape(B, -1).gather(1, topk_idx)
    peak_E = grid_EE.reshape(B, -1).gather(1, topk_idx)
    peak_I = flat.gather(1, topk_idx)

    active = torch.arange(max_peaks, device=device).unsqueeze(0) < n_peaks.unsqueeze(1)
    peak_R = torch.where(active, peak_R, peak_R[:, 0:1].expand_as(peak_R))
    peak_E = torch.where(active, peak_E, peak_E[:, 0:1].expand_as(peak_E))
    peak_I = torch.where(active, peak_I, torch.zeros_like(peak_I))

    dR, dEta = 2.0, 0.1
    dIMax = 400.0 / max(srfac, 1)

    init = torch.zeros(B, max_peaks, 8, device=device)
    init[:, :, 0] = peak_R;  init[:, :, 1] = peak_E
    init[:, :, 2] = 0.5;     init[:, :, 3] = 0.3
    init[:, :, 4] = 0.5;     init[:, :, 5] = 0.3
    init[:, :, 6] = 0.5;     init[:, :, 7] = peak_I

    lb = torch.zeros(B, max_peaks, 8, device=device)
    lb[:, :, 0] = peak_R - dR;  lb[:, :, 1] = peak_E - dEta
    lb[:, :, 2] = 0.1;  lb[:, :, 3] = 0.05
    lb[:, :, 4] = 0.1;  lb[:, :, 5] = 0.05

    ub = torch.zeros(B, max_peaks, 8, device=device)
    ub[:, :, 0] = peak_R + dR;  ub[:, :, 1] = peak_E + dEta
    ub[:, :, 2] = 3.0;  ub[:, :, 3] = 3.0
    ub[:, :, 4] = 3.0;  ub[:, :, 5] = 3.0
    ub[:, :, 6] = 1.0;  ub[:, :, 7] = peak_I + dIMax

    return init, n_peaks, lb, ub


# ─── Frame-level entry point (called from sr_process.py) ────────────────────

def gpu_fit_frame_patches(patches_to_fit, patches_Y00, patches_Z00,
                          patches_exp, nr_pixels_in_patch,
                          sr_params, sr_config, srfac,
                          omega, shiftYpx, shiftZpx,
                          torch_devs, n_steps=20, lr=0.15,
                          use_compile=True):
    """Fit all patches in one frame via GPU pseudo-Voigt fitting.

    Drop-in replacement for the per-patch fitting loop in sr_process.py.
    Performs batched GPU peak detection + Adam optimization, then
    extracts per-peak CSV row data on CPU.

    Args:
        patches_to_fit: (N, 1, H, W) SR-predicted patches
        patches_Y00, patches_Z00: lists of patch origins
        patches_exp: (N, 1, H_native, W_native) native-res patches
        nr_pixels_in_patch: list of non-zero pixel counts
        sr_params: dict with Ypx_BC, Zpx_BC, etc.
        sr_config: dict with lrsz, peak_find_args, etc.
        srfac: super-resolution factor
        omega: rotation angle for this frame
        shiftYpx, shiftZpx: sub-pixel shift corrections
        torch_devs: torch.device
        n_steps: Adam optimizer iterations
        lr: learning rate
        use_compile: use torch.compile

    Returns:
        df_rows: list of [spotID, IntegInt, omega, Y, Z, IMax, R, Eta, ...] rows
        n_peaks_list: list of peak counts per patch
        spotID: final spot counter value
    """
    from sr_midas.physics.peaks2d import pseudoVoigt2d_diffLGwidth

    n_patches = len(patches_to_fit)
    if n_patches == 0:
        return [], [], 0

    lrsz = sr_config["lrsz"]
    Ypx_BC = sr_params["Ypx_BC"]
    Zpx_BC = sr_params["Zpx_BC"]
    lr_int_thresh = sr_config["peak_find_args"]["pvfit_int_thresh"][f"SRx{srfac}"]
    min_d = sr_config["peak_find_args"]["min_d"][f"SRx{srfac}"]
    thresh_rel = sr_config["peak_find_args"]["thresh_rel"][f"SRx{srfac}"]

    device = torch_devs
    patch_intensities = patches_to_fit[:, 0]  # (N, H, W)

    Y00_t = torch.tensor(np.array(patches_Y00, dtype=np.float32), device=device)
    Z00_t = torch.tensor(np.array(patches_Z00, dtype=np.float32), device=device)
    patches_t = torch.from_numpy(patch_intensities.astype(np.float32)).to(device)

    grid_RR, grid_EE = build_RE_grids(Y00_t, Z00_t, lrsz, srfac, Ypx_BC, Zpx_BC, device)

    # exclude_border=True matches peak_local_max default
    init_params, n_peaks_detected, lb, ub = detect_peaks_and_init(
        patches_t, grid_RR, grid_EE, srfac,
        min_distance=min_d, threshold_rel=thresh_rel,
        edge_bound_cutoff_fac=0.0, exclude_border=True)

    threshold = lr_int_thresh / (srfac * srfac)

    best_params, costs = batched_adam_fit(
        grid_RR, grid_EE, patches_t, init_params,
        n_peaks_detected, lb, ub,
        n_steps=n_steps, lr=lr, threshold=threshold,
        use_compile=use_compile)

    # Transfer to CPU for CSV row extraction
    best_params_np = best_params.detach().cpu().numpy()
    n_peaks_np = n_peaks_detected.cpu().numpy()
    grid_RR_np = grid_RR.cpu().numpy()
    grid_EE_np = grid_EE.cpu().numpy()

    df_rows = []
    n_peaks_list = []
    spotID = 0
    downscale_fac = max(int(srfac), 1)

    for patch_i in range(n_patches):
        n_pk = int(n_peaks_np[patch_i])
        n_peaks_list.append(n_pk)
        Y00_val = patches_Y00[patch_i]
        Z00_val = patches_Z00[patch_i]

        for peak_j in range(n_pk):
            spotID += 1
            p = best_params_np[patch_i, peak_j]

            R, Eta_val = float(p[0]), float(p[1])
            SigGR, SigGEta = float(p[2]), float(p[3])
            SigLR, SigLEta = float(p[4]), float(p[5])
            LGmix_val, IMax_fit = float(p[6]), float(p[7])

            YCen_px = Ypx_BC + R * np.sin(np.deg2rad(Eta_val)) + float(shiftYpx)
            ZCen_px = Zpx_BC + R * np.cos(np.deg2rad(Eta_val)) + float(shiftZpx)

            gRR_i, gEE_i = grid_RR_np[patch_i], grid_EE_np[patch_i]
            peak_fit_patch = pseudoVoigt2d_diffLGwidth(
                gRR_i, gEE_i, y0=R, z0=Eta_val,
                ySigG=SigGR, zSigG=SigGEta,
                ySigL=SigLR, zSigL=SigLEta,
                LGmix=LGmix_val, IMax=IMax_fit)

            if downscale_fac > 1:
                h, w = peak_fit_patch.shape
                nh, nw = h // downscale_fac, w // downscale_fac
                pSRx1 = peak_fit_patch[:nh*downscale_fac, :nw*downscale_fac].reshape(
                    nh, downscale_fac, nw, downscale_fac).sum(axis=(1, 3))
            else:
                pSRx1 = peak_fit_patch

            r_max, c_max = np.unravel_index(np.argmax(pSRx1), pSRx1.shape)
            maxY = Y00_val + c_max
            maxZ = Z00_val + r_max

            IntegratedIntensity = float(np.sum(peak_fit_patch))
            IMax_out = float(np.max(pSRx1))
            SigmaR = max(SigGR, SigLR)
            SigmaEta = max(SigGEta, SigLEta)
            NrPixels = int(np.count_nonzero(pSRx1 * patches_exp[patch_i]))
            TotalNrPixels = nr_pixels_in_patch[patch_i]

            df_rows.append([spotID, IntegratedIntensity, omega, YCen_px, ZCen_px,
                           IMax_out, R, Eta_val, SigmaR, SigmaEta, NrPixels,
                           TotalNrPixels, n_pk, maxY, maxZ,
                           maxY - YCen_px, maxZ - ZCen_px])

    return df_rows, n_peaks_list, spotID


def warmup_gpu_compile(sr_config, sr_params, srfac, torch_devs,
                       n_steps=20, lr=0.15):
    """One-time torch.compile warmup. Call before processing frames."""
    dummy_n = 50
    lrsz = sr_config["lrsz"]
    dummy_patches = torch.randn(dummy_n, lrsz * srfac, lrsz * srfac, device=torch_devs)
    dummy_y = torch.zeros(dummy_n, device=torch_devs)
    dummy_z = torch.zeros(dummy_n, device=torch_devs)
    dummy_RR, dummy_EE = build_RE_grids(dummy_y, dummy_z, lrsz, srfac,
                                         sr_params["Ypx_BC"], sr_params["Zpx_BC"],
                                         torch_devs)
    dummy_init, dummy_np, dummy_lb, dummy_ub = detect_peaks_and_init(
        dummy_patches, dummy_RR, dummy_EE, srfac)
    lr_int_thresh = sr_config["peak_find_args"]["pvfit_int_thresh"][f"SRx{srfac}"]
    batched_adam_fit(dummy_RR, dummy_EE, dummy_patches, dummy_init,
                    dummy_np, dummy_lb, dummy_ub,
                    n_steps=n_steps, lr=lr,
                    threshold=lr_int_thresh / (srfac * srfac),
                    use_compile=True)
    torch.cuda.synchronize()
