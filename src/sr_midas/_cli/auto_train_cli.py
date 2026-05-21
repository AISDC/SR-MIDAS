"""CLI wrapper for the automated CNNSR training workflow.

Usage:
    sr-midas-auto-train -config train_config.json
"""

import argparse

from sr_midas.workflows.auto_train import run_auto_train


def main():
    parser = argparse.ArgumentParser(
        description="Automated end-to-end CNNSR model training. "
                    "Takes a MIDAS zip directory and produces trained cascaded "
                    "models (x2, x4, x8) plus a ready-to-use sr_config.json.",
        epilog="Example: sr-midas-auto-train -config my_train_config.json"
    )
    parser.add_argument(
        "-config", type=str, required=True,
        help="Path to JSON config file. Required keys: 'midas_dir', 'output_dir'. "
             "All other parameters have sensible defaults."
    )
    args = parser.parse_args()

    sr_config_path = run_auto_train(args.config)

    print(f"\nGenerated SR config: {sr_config_path}")
    print("Use it with: sr-midas-process -midasZarrDir <data> "
          f"-SRconfig {sr_config_path}")


if __name__ == "__main__":
    main()
