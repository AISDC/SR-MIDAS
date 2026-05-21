"""GPU-only cascaded SR-CNN inference (SRx2 -> SRx4 -> SRx8).

Eliminates the per-stage CPU round-trip in `sr_process.py:457-555`:
  - keeps patches as a torch tensor on device for the entire cascade,
  - replaces `np.repeat(np.repeat(...))` upsampling with
    `torch.repeat_interleave` on device (bit-equivalent for nearest-
    neighbor upsampling),
  - replaces per-stage max-normalization with `tensor.amax(dim=...,
    keepdim=True)` on device,
  - applies the SRx8 intensity rescaling on device (`patches_Isum` is
    already a tensor here),
  - returns the final SR prediction as a torch tensor — the caller can
    hand it straight to `gpu_fit_frame_patches_v3` without a numpy round
    trip.

Numerics: bit-for-bit identical to the v1 CPU-bouncing cascade aside from
floating-point ordering differences inside the CNN (PyTorch is allowed
to choose different reduction orders on GPU vs CPU). With CUDA
deterministic mode disabled (the default in `sr_process.py`), small
last-decimal differences are expected.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.amp import autocast


def _upsample_normalize_gpu(patches_t, upscale_fac):
    """GPU equivalent of the per-stage CPU upsample + max-normalize.

    Replaces `np.repeat(np.repeat(x, k, 2), k, 3) / k**2` followed by
    per-patch max-normalize. `repeat_interleave` is bit-equivalent to
    `np.repeat` for nearest-neighbor upsampling.

    Args:
        patches_t:    (N, C, H, W) torch float on GPU.
        upscale_fac:  int, nearest-neighbor upsample factor in H and W.

    Returns:
        (N, C, H*upscale_fac, W*upscale_fac) torch float on the same
        device. Each (N, C) tile is normalized so its max == 1 (or
        unchanged if its max was 0).
    """
    up = patches_t.repeat_interleave(upscale_fac, dim=2) \
                  .repeat_interleave(upscale_fac, dim=3)
    up = up / float(upscale_fac * upscale_fac)
    max_vals = up.amax(dim=(2, 3), keepdim=True)
    max_vals = torch.where(max_vals == 0, torch.ones_like(max_vals), max_vals)
    return up / max_vals


def cascade_sr_gpu(patches_exp_np, patches_Isum,
                   x2mod, x4mod, x8mod,
                   x2mod_ch, x4mod_ch, x8mod_ch,
                   srfac, batch_size, torch_devs,
                   use_autocast=True):
    """Run the cascaded SR CNN (SRx2 -> SRx4 -> SRx8) entirely on GPU.

    Each stage upsamples by 2x via `repeat_interleave`, max-normalizes,
    runs the CNN in `batch_size` chunks, and passes the prediction
    directly into the next stage as a torch tensor. The SRx8 stage uses
    `torch.amp.autocast` on CUDA (matching the original behavior at
    `sr_process.py:521-548`) and rescales each predicted patch so its
    sum matches the corresponding native patch's intensity sum.

    Args:
        patches_exp_np: (N, 1, lrsz, lrsz) — either numpy or torch
            float; native-resolution patches from `extract_patches_gpu`
            (torch) or `patches_from_detector_frame` (numpy).
        patches_Isum:   (N,) sequence — list, numpy, or torch tensor
            of per-patch native intensity sums (used to rescale SRx8).
        x2mod, x4mod, x8mod: trained CNN modules from
            `load_trained_CNNSR`. Already moved to `torch_devs`.
        x2mod_ch, x4mod_ch, x8mod_ch: list[int] of channel indices each
            CNN consumes (typically `[0]`).
        srfac:        int in {2, 4, 8}; controls how many stages run.
        batch_size:   int, mini-batch size for CNN forward (from
            `sr_config["batch_size"]`).
        torch_devs:   torch.device, the CUDA device.
        use_autocast: bool, enable mixed-precision autocast on the SRx8
            stage (default True; ignored on non-CUDA devices).

    Returns:
        (N, 1, lrsz*srfac, lrsz*srfac) torch float32 on `torch_devs` —
        the final SR prediction. Callers can pass it straight into
        `gpu_fit_frame_patches` without a CPU round-trip.
    """
    n_patches = patches_exp_np.shape[0]
    if isinstance(patches_exp_np, torch.Tensor):
        patches_exp_t = patches_exp_np.to(device=torch_devs, dtype=torch.float32)
    else:
        patches_exp_t = torch.from_numpy(patches_exp_np.astype(np.float32)).to(torch_devs)

    def _run_stage(input_t, channels, mod, upscale, use_amp=False):
        """One cascade stage: channel-select -> upsample -> CNN inference.

        Args:
            input_t:    (N, C_in, H, W) torch float on `torch_devs`,
                input to this stage (either patches_exp_t for the SRx2
                stage, or the previous stage's output).
            channels:   list[int] or int, channel indices the CNN
                consumes (kept-dim selection from `input_t`).
            mod:        the CNN module to call.
            upscale:    int, nearest-neighbor upscale before the CNN.
            use_amp:    bool, if True (and on CUDA) wrap the forward
                in autocast fp16.

        Returns:
            (N, C_out, H*upscale, W*upscale) torch float on `torch_devs`.
        """
        # mimic `patches[:, channels, :, :]` keeping the channel dim
        sub = input_t[:, channels, :, :] if isinstance(channels, (list, tuple)) \
              else input_t[:, [channels], :, :]
        Xin = _upsample_normalize_gpu(sub, upscale)
        out_chunks = []
        n_batches = n_patches // batch_size
        with torch.no_grad():
            ctx = autocast(device_type=torch_devs.type) if (use_amp and torch_devs.type == "cuda") \
                  else torch.amp.autocast(device_type=torch_devs.type, enabled=False)
            with ctx:
                for i in range(n_batches + 1):
                    s = i * batch_size
                    f = min((i + 1) * batch_size, n_patches)
                    if s >= n_patches:
                        break
                    batch = Xin[s:f]
                    pred = mod.forward(batch)
                    out_chunks.append(pred)
        return torch.cat(out_chunks, dim=0)

    # All three stages run in autocast when use_autocast=True (matches
    # sr_autocast_bench V7: ~1.20x over today's SRx8-only autocast with
    # fidelity preserved). Inter-stage tensors are cast back to fp32
    # so `_upsample_normalize_gpu` (divide-by-max) stays in fp32.
    SRx2_pred = _run_stage(patches_exp_t, x2mod_ch, x2mod, 2, use_amp=use_autocast)
    if srfac <= 2:
        return SRx2_pred.float()
    SRx2_pred = SRx2_pred.float()

    SRx4_pred = _run_stage(SRx2_pred, x4mod_ch, x4mod, 2, use_amp=use_autocast)
    if srfac <= 4:
        return SRx4_pred.float()
    SRx4_pred = SRx4_pred.float()

    SRx8_pred = _run_stage(SRx4_pred, x8mod_ch, x8mod, 2, use_amp=use_autocast)
    SRx8_pred = SRx8_pred.float()  # ensure float32 even if autocast used fp16

    # Per-patch intensity rescaling so sum(SRx8_pred) matches the native
    # patch's sum. Mirrors sr_process.py:528-532 / 542-546.
    isum_t = torch.as_tensor(patches_Isum, device=torch_devs, dtype=torch.float32)
    current_sums = SRx8_pred.sum(dim=(1, 2, 3))
    # Avoid div-by-zero on degenerate patches
    safe = torch.where(current_sums == 0, torch.ones_like(current_sums), current_sums)
    scaling = (isum_t / safe).view(-1, 1, 1, 1)
    SRx8_pred = SRx8_pred * scaling
    return SRx8_pred
