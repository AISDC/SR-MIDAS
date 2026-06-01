"""CLI wrapper for run_sr_process."""

import argparse

from sr_midas.pipeline.sr_process import run_sr_process


def main():
    parser = argparse.ArgumentParser(
        description="Run the MIDAS super-resolution pipeline."
    )
    parser.add_argument("-midasZarrDir", type=str, required=True,
                        help="Path to MIDAS .zip zarr directory")
    parser.add_argument("-srfac", type=int, default=8,
                        help="Super-resolution factor (2, 4, or 8)")
    parser.add_argument("-SRconfig", type=str, default=None,
                        help="Path to SR config .json or .txt file (default: bundled cnnsr_sr_config.json)")
    parser.add_argument("-saveSRpatches", type=int, default=1,
                        help="Save SR patches (1=yes, 0=no)")
    parser.add_argument("-saveFrameGoodCoords", type=int, default=1,
                        help="Save frame good coordinates (1=yes, 0=no)")
    parser.add_argument("-maxFrames", type=int, default=0,
                        help="Cap the number of frames processed (0 = no cap). "
                             "Useful when running midas_style on real data.")
    # The peak-fit routine is selected via the `peak_fit_method` field in
    # the sr_config JSON (bundled cnnsr_sr_config.json or any custom one
    # passed via -SRconfig). Valid values: 'gpu_adam', 'gpu_midas_style',
    # 'midas_style'. Edit the JSON to switch.

    args = parser.parse_args()

    run_sr_process(
        midasZarrDir=args.midasZarrDir,
        srfac=args.srfac,
        SRconfig_path=args.SRconfig,
        saveSRpatches=args.saveSRpatches,
        saveFrameGoodCoords=args.saveFrameGoodCoords,
        max_frames=(args.maxFrames if args.maxFrames > 0 else None),
    )


if __name__ == "__main__":
    main()
