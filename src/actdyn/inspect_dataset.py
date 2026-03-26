from __future__ import annotations

import argparse
import json

from actdyn.data.robomimic_lowdim import auto_detect_obs_keys, summarize_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a RoboMimic HDF5 dataset.")
    parser.add_argument("--dataset", required=True, help="Path to HDF5 dataset.")
    args = parser.parse_args()

    summary = summarize_dataset(args.dataset)
    summary["auto_detected_lowdim_obs_keys"] = auto_detect_obs_keys(args.dataset)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
