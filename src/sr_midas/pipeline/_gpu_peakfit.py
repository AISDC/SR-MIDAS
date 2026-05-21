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
from torch.amp import autocast


# ─── Coordinate transforms (batched) ────────────────────────────────────────

def build_RE_grids(Y00, Z00, lrsz, srfac, Ypx_BC, Zpx_BC, device):
    """Build per-patch R-Eta coordinate grids.

    Each patch is a (lrsz*srfac, lrsz*srfac) sub-window of the detector
    starting at (Z00, Y00). This function returns the radial distance
    from the beam center (R, in low-res pixel units) and the azimuthal
    angle (Eta, degrees) at every super-resolved pixel of every patch.

    Args:
        Y00:     (B,) torch float, Y coord of each patch origin (low-res px).
        Z00:     (B,) torch float, Z coord of each patch origin (low-res px).
        lrsz:    int, patch size in low-res pixels (e.g. 20).
        srfac:   int, super-resolution upscale factor (e.g. 8).
        Ypx_BC:  float, Y of beam center on the detector (low-res px).
        Zpx_BC:  float, Z of beam center on the detector (low-res px).
        device:  torch.device on which the output grids live.

    Returns:
        grid_RR: (B, lrsz*srfac, lrsz*srfac) torch float, radial distance.
        grid_EE: (B, lrsz*srfac, lrsz*srfac) torch float, azimuthal angle
            in degrees with sign indicating side of beam (sign(Y - Ypx_BC)).
    """
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
    """Evaluate the summed multi-peak 2D pseudo-Voigt forward model.

    Each peak is `PV = LG*L + (1-LG)*G` with separate Gaussian and
    Lorentzian widths in R and Eta. Peak slots beyond
    `n_peaks_per_patch[i]` are masked to zero (they contribute nothing
    to the summed output, but are still computed — `max_peaks` is fixed
    at 5 for batched processing).

    Args:
        grid_RR: (B, H, W) torch float, R coords (from `build_RE_grids`).
        grid_EE: (B, H, W) torch float, Eta coords (degrees).
        params:  (B, max_peaks, 8) torch float — per peak
            (R, Eta, sigGR, sigGE, sigLR, sigLE, LGmix, IMax).
        n_peaks_per_patch: (B,) torch long, active peak count per patch.
        threshold: float; if > 0, output pixels below this value are
            clamped to 0 (post-sum thresholding to suppress tails).

    Returns:
        patches: (B, H, W) torch float, sum of per-peak pseudo-Voigts.
    """
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
    """Map unconstrained `raw` into [lower, upper] via shifted sigmoid.

    Args:
        raw:   torch tensor of any shape; the unconstrained optimizer variable.
        lower: torch tensor broadcast-compatible with `raw`; per-element lower bound.
        upper: torch tensor broadcast-compatible with `raw`; per-element upper bound.

    Returns:
        torch tensor same shape as `raw`, in [lower, upper] element-wise.
    """
    return lower + (upper - lower) * torch.sigmoid(raw)

def _inv_project(params, lower, upper):
    """Inverse of `_project`: map bounded `params` back to unconstrained space.

    Used once to convert initial bounded guesses into the optimizer's
    `raw` representation. Clamped slightly inside the open interval to
    avoid `log(0)` / `log(inf)`.

    Args:
        params: bounded torch tensor in (lower, upper).
        lower:  per-element lower bound (broadcast-compatible).
        upper:  per-element upper bound (broadcast-compatible).

    Returns:
        torch tensor same shape as `params`, finite real values suitable
        as input to the optimizer; passing the result back through
        `_project` reproduces `params` up to the clamp epsilon.
    """
    eps = 1e-6
    t = ((params - lower) / (upper - lower + 1e-12)).clamp(eps, 1.0 - eps)
    return torch.log(t / (1.0 - t))


# ─── Compiled optimization step ─────────────────────────────────────────────

_compiled_step_fn = None

def _get_compiled_step():
    """Lazily build (and cache) the JIT-compiled optimizer step.

    The returned closure takes the optimizer state plus targets and
    returns the scalar loss for one step of `batched_adam_fit`. It is
    compiled with `torch.compile(dynamic=True)` so that variable batch
    sizes across frames don't trigger recompilation. The compiled
    function is cached in module-global `_compiled_step_fn` so the
    compile cost is paid once per process.

    Returns:
        A callable `step(raw, target_flat, lb, ub, n_pk, grid_RR, grid_EE)
        -> scalar loss tensor`.
    """
    global _compiled_step_fn
    if _compiled_step_fn is None:
        @torch.compile(mode='default', dynamic=True)
        def _step(raw, target_flat, lb, ub, n_pk, grid_RR, grid_EE):
            """One JIT-compiled Adam step body: project, forward, residuals.

            The forward (pseudo-Voigt rendering) runs under `autocast(fp16)`
            on CUDA to halve memory bandwidth. The squared-residual sum
            is computed in fp32 to avoid accumulator drift across the
            millions of pixels in a frame.

            Args:
                raw:        (B, max_peaks, 8) torch float, unbounded
                    optimizer variable (autograd-tracked).
                target_flat: (B, H*W) torch float, observed intensities
                    flattened for residual subtraction.
                lb, ub:     (B, max_peaks, 8) torch float bounds for
                    `_project`.
                n_pk:       (B,) torch long, active peak count per patch
                    (passed to `pseudo_voigt_2d_batch` for masking).
                grid_RR, grid_EE: (B, H, W) torch float coordinate grids.

            Returns:
                Scalar torch float (loss = sum of squared residuals
                across all patches and all pixels).
            """
            with autocast(device_type='cuda', dtype=torch.float16):
                params = _project(raw, lb, ub)
                pred = pseudo_voigt_2d_batch(grid_RR, grid_EE, params, n_pk)
            # promote back to fp32 for the residual + reduction so the
            # sum across millions of squared residuals doesn't lose
            # precision in the accumulator.
            residuals = pred.float().reshape(pred.shape[0], -1) - target_flat
            return (residuals * residuals).sum()
        _compiled_step_fn = _step
    return _compiled_step_fn


# ─── Core optimizer ──────────────────────────────────────────────────────────

def batched_adam_fit(grid_RR, grid_EE, target, init_params,
                     n_peaks, lower, upper,
                     n_steps=20, lr=0.15, threshold=0.0,
                     use_compile=True):
    """Bounded Adam minimization of sum-sq residuals over all patches.

    Bounds are enforced by an inverse-sigmoid reparameterization: the
    optimizer variable is `raw`, and `_project(raw, lower, upper)` is
    the bounded parameter passed to the forward model each step. The
    per-step closure is JIT-compiled on CUDA when `use_compile=True`.

    Args:
        grid_RR, grid_EE: (B, H, W) torch float coordinate grids.
        target:    (B, H, W) torch float, observed intensities to fit.
        init_params: (B, max_peaks, 8) torch float, starting bounded params.
        n_peaks:   (B,) torch long, active peak count per patch.
        lower, upper: (B, max_peaks, 8) torch float, per-element bounds.
        n_steps:   int, number of Adam iterations (default 20).
        lr:        float, Adam learning rate (default 0.15).
        threshold: float, post-sum forward-model threshold (see
            `pseudo_voigt_2d_batch`); only used in the fallback path.
        use_compile: bool, use `torch.compile` step on CUDA when True.

    Returns:
        params: (B, max_peaks, 8) torch float, fitted bounded params.
        costs:  (B,) torch float, per-patch mean-squared residual at the
            final params (sqrt of this is reported as `FitRMSE`).
    """
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
    """Detect up to 5 peaks per patch on GPU and build initial bounds.

    Matches scikit-image `peak_local_max` semantics: local 3x3 maxima
    above `threshold_rel * patch_max`, with plateau suppression so each
    flat region contributes one pixel, and border exclusion within
    `min_distance` (plus an optional `edge_bound_cutoff_fac * srfac`
    margin). Up to 5 peaks per patch are kept (ordered by intensity);
    patches with no detected peak fall back to the brightest pixel.

    Initial bounded parameters and sigmoid bounds are then constructed
    for `batched_adam_fit`.

    Args:
        patches:   (B, H, W) torch float, SR-predicted patch intensities.
        grid_RR, grid_EE: (B, H, W) torch float, R-Eta coordinate grids.
        srfac:     int, super-resolution factor (used to scale `dIMax`
            and the optional edge cutoff).
        min_distance: int, half-kernel size for max-pool peak detection.
        threshold_rel: float in [0, 1], threshold relative to per-patch max.
        edge_bound_cutoff_fac: float, extra border to exclude expressed
            in low-res pixels (multiplied by `srfac` for SR pixels).
        exclude_border: bool, exclude peaks within `min_distance` of edge.

    Returns:
        init:    (B, max_peaks=5, 8) torch float, initial bounded params.
        n_peaks: (B,) torch long, count of active peaks per patch (>=1).
        lb, ub:  (B, max_peaks=5, 8) torch float, per-element bounds for
            sigmoid reparameterization in `batched_adam_fit`.
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



# ─── Per-peak forward model (no sum over peaks) ─────────────────────────────

def _eval_pv_per_peak(grid_RR, grid_EE, params):
    """Evaluate 2D pseudo-Voigt for each peak slot separately.

    Same math as `pseudo_voigt_2d_batch` but without the per-patch sum
    over the peak axis and without the `n_peaks` mask. Used by the
    post-fit GPU reductions to render each peak's intensity into its
    own (H, W) tile so per-peak quantities (IntegratedIntensity, IMax,
    argmax position, etc.) can be computed via batched reductions.

    Args:
        grid_RR, grid_EE: (B, H, W) torch float coordinate grids.
        params: (B, max_peaks, 8) torch float — per peak
            (R, Eta, sigGR, sigGE, sigLR, sigLE, LGmix, IMax).

    Returns:
        (B, max_peaks, H, W) torch float; inactive peak slots still
        compute real values — callers are responsible for masking with
        an active-peak mask if those slots must be ignored.
    """
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

    return LG * L + (1.0 - LG) * G


# ─── Frame-level entry point (all-GPU pipeline) ─────────────────────────────

def gpu_fit_frame_patches(patches_to_fit_t, patches_Y00, patches_Z00,
                          patches_exp_t, nr_pixels_in_patch,
                          patches_Isum,
                          sr_params, sr_config, srfac,
                          omega, shiftYpx, shiftZpx,
                          torch_devs, n_steps=20, lr=0.15,
                          use_compile=True):
    """Fit all patches in one frame on GPU and return MIDAS-schema rows.

    The forward model is a sum of 2D pseudo-Voigts (up to 5 peaks per
    patch). Fitting uses bounded Adam (20 steps, lr=0.15) with a
    `torch.compile`-JIT'd step (see `_get_compiled_step`). The per-peak
    post-fit reductions (per-peak forward render, sum-pool to native
    pixels, IntegratedIntensity / IMax / argmax position / NrPixels /
    YCen / ZCen / SigmaR / SigmaEta / FitRMSE / rawIMax) all run as
    batched ops on device; a single D2H transfer at the end produces
    the (total_peaks, 29) MIDAS-schema CSV rows.

    Args:
        patches_to_fit_t: (N, 1, lrsz*srfac, lrsz*srfac) torch float32
            on `torch_devs`. Output of the SR cascade.
        patches_Y00, patches_Z00: per-patch origin pixel coords (N,).
            Accepted as torch tensor (any numeric dtype), numpy array,
            or list[int]; coerced to float32 on `torch_devs` internally.
        patches_exp_t: (N, 1, lrsz, lrsz) torch float32 on `torch_devs`,
            native-resolution patch intensities. Used for `NrPixels`
            (mask of nonzero native pixels) and `rawIMax`.
        nr_pixels_in_patch: per-patch nonzero pixel count (N,). Same
            type flexibility as `patches_Y00`. Written verbatim as
            `TotalNrPixelsInPeakRegion`.
        patches_Isum: per-patch raw intensity sum (N,). Same type
            flexibility as `patches_Y00`. Written as `RawSumIntensity`.
        sr_params: dict; needs `Ypx_BC`, `Zpx_BC`.
        sr_config: dict; needs `lrsz`, and
            `peak_find_args.{pvfit_int_thresh,min_d,thresh_rel}` keyed
            by `f"SRx{srfac}"`.
        srfac: int, SR factor (2, 4, or 8).
        omega: float, rotation angle for this frame (degrees).
        shiftYpx, shiftZpx: float, sub-pixel Y/Z corrections added to
            the fitted (YCen, ZCen).
        torch_devs: torch.device, the CUDA device for all compute.
        n_steps: int, Adam iterations (default 20).
        lr: float, Adam learning rate (default 0.15).
        use_compile: bool, JIT-compile the Adam step on CUDA (default True).

    Returns:
        df_rows: list[list[float]] of length `total_peaks`, each row
            29 columns in MIDAS CSV order (see `sr_process.col_names`).
        n_peaks_list: list[int] of length N, active peak count per patch.
        spotID: int, total peaks across all patches (== len(df_rows)).
    """
    n_patches = patches_to_fit_t.shape[0]
    if n_patches == 0:
        return [], [], 0

    lrsz = sr_config["lrsz"]
    Ypx_BC = sr_params["Ypx_BC"]
    Zpx_BC = sr_params["Zpx_BC"]
    lr_int_thresh = sr_config["peak_find_args"]["pvfit_int_thresh"][f"SRx{srfac}"]
    min_d = sr_config["peak_find_args"]["min_d"][f"SRx{srfac}"]
    thresh_rel = sr_config["peak_find_args"]["thresh_rel"][f"SRx{srfac}"]

    device = torch_devs
    patches_t = patches_to_fit_t[:, 0].contiguous()  # (N, H, W)

    def _to_float_tensor(x):
        """Coerce list / numpy array / torch tensor to float32 on `device`.

        Lets the function accept either Python lists (from the legacy
        CPU path) or pre-existing torch tensors (from the all-GPU
        pipeline) without an explicit branch at each call site.

        Args:
            x: list, numpy array, or torch tensor of numeric values.

        Returns:
            torch float32 tensor on `device` with the same elements as `x`.
        """
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=torch.float32)
        return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)

    Y00_t = _to_float_tensor(patches_Y00)
    Z00_t = _to_float_tensor(patches_Z00)
    isum_t = _to_float_tensor(patches_Isum)
    nrpx_t = _to_float_tensor(nr_pixels_in_patch)

    grid_RR, grid_EE = build_RE_grids(Y00_t, Z00_t, lrsz, srfac, Ypx_BC, Zpx_BC, device)

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

    B = n_patches
    K = best_params.shape[1]

    with torch.no_grad():
        pv_per_peak = _eval_pv_per_peak(grid_RR, grid_EE, best_params)
        H, W = pv_per_peak.shape[-2], pv_per_peak.shape[-1]
        if srfac > 1:
            pooled = F.avg_pool2d(pv_per_peak.reshape(B * K, 1, H, W),
                                  kernel_size=srfac, stride=srfac,
                                  divisor_override=1)
            pv_srx1 = pooled.reshape(B, K, H // srfac, W // srfac)
        else:
            pv_srx1 = pv_per_peak

        Ws = pv_srx1.shape[-1]
        integrated = pv_per_peak.sum(dim=(-2, -1))
        imax_out   = pv_srx1.amax(dim=(-2, -1))
        argmax_flat = pv_srx1.flatten(start_dim=-2).argmax(dim=-1)
        r_max = (argmax_flat // Ws).to(torch.float32)
        c_max = (argmax_flat %  Ws).to(torch.float32)
        maxY = Y00_t.unsqueeze(1) + c_max
        maxZ = Z00_t.unsqueeze(1) + r_max

        exp_b = patches_exp_t.squeeze(1).unsqueeze(1)
        nr_pixels = ((pv_srx1 * exp_b) != 0).sum(dim=(-2, -1)).to(torch.float32)

        R     = best_params[..., 0]
        Eta   = best_params[..., 1]
        sGR   = best_params[..., 2]
        sGE   = best_params[..., 3]
        sLR   = best_params[..., 4]
        sLE   = best_params[..., 5]
        LGmix = best_params[..., 6]

        SigmaR   = torch.maximum(sGR, sLR)
        SigmaEta = torch.maximum(sGE, sLE)

        eta_rad = torch.deg2rad(Eta)
        YCen = Ypx_BC + R * torch.sin(eta_rad) + float(shiftYpx)
        ZCen = Zpx_BC + R * torch.cos(eta_rad) + float(shiftZpx)
        diffY = maxY - YCen
        diffZ = maxZ - ZCen

        fit_rmse   = torch.sqrt(costs).unsqueeze(1).expand(B, K)
        omega_t    = torch.full((B, K), float(omega), device=device)
        nPeaks_t   = n_peaks_detected.to(torch.float32).unsqueeze(1).expand(B, K)
        total_nrpx = nrpx_t.unsqueeze(1).expand(B, K)
        rawIMax    = patches_exp_t.amax(dim=(-2, -1)).squeeze(1).unsqueeze(1).expand(B, K)
        raw_isum   = isum_t.unsqueeze(1).expand(B, K)
        zeros_bk   = torch.zeros((B, K), device=device)

        active = (torch.arange(K, device=device).unsqueeze(0)
                  < n_peaks_detected.unsqueeze(1))

        cols = [
            zeros_bk, integrated, omega_t, YCen, ZCen, imax_out,
            R, Eta, SigmaR, SigmaEta, nr_pixels,
            total_nrpx, nPeaks_t, maxY, maxZ, diffY, diffZ,
            rawIMax, zeros_bk, fit_rmse, zeros_bk,
            sGR, sLR, sGE, sLE,
            LGmix, raw_isum, zeros_bk, fit_rmse,
        ]
        rows_bk = torch.stack(cols, dim=-1)
        rows_flat = rows_bk.reshape(B * K, 29)
        active_flat = active.reshape(B * K)
        valid_rows = rows_flat[active_flat]
        total_peaks = valid_rows.shape[0]

        valid_rows[:, 0] = torch.arange(1, total_peaks + 1,
                                        device=device, dtype=valid_rows.dtype)

    rows_np = valid_rows.cpu().numpy()
    n_peaks_np = n_peaks_detected.cpu().numpy()

    df_rows = [list(row) for row in rows_np]
    n_peaks_list = [int(n) for n in n_peaks_np]
    spotID = int(total_peaks)
    return df_rows, n_peaks_list, spotID


def warmup_gpu_compile(sr_config, sr_params, srfac, torch_devs,
                       n_steps=20, lr=0.15):
    """Trigger torch.compile of the Adam step on dummy data.

    The first call to `batched_adam_fit` with `use_compile=True` pays a
    multi-second compile cost. Calling this once at pipeline startup
    moves that cost out of the first real frame.

    Args:
        sr_config: dict; needs `lrsz` and `peak_find_args.pvfit_int_thresh`.
        sr_params: dict; needs `Ypx_BC`, `Zpx_BC`.
        srfac:     int, SR factor used to size the dummy patches.
        torch_devs: torch.device, CUDA device on which to compile.
        n_steps:   int, must match the value `batched_adam_fit` will be
            called with later (default 20).
        lr:        float, Adam learning rate (default 0.15; matches
            `batched_adam_fit` default).

    Returns:
        None. Side effect: populates the module-global
        `_compiled_step_fn` cache so subsequent calls reuse the same
        compiled artifact.
    """
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
