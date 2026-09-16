"""Download a Satlas-Pretrain backbone from the torchgeo/satlas Hub mirror.

The aerial Swin-V2-B is the one this project uses (see src/models/satlas.py):

    python tools/fetch_satlas.py                     # aerial_swinb_si -> pretrain_weight/
    python tools/fetch_satlas.py --list              # show every file in the repo

Apache-2.0, so the file can sit alongside the other checkpoints in
pretrain_weight/ without a licence caveat.
"""

import argparse
import os
import urllib.request

REPO = "https://huggingface.co/torchgeo/satlas/resolve/main"
DEFAULT = "aerial_swinb_si-e4169eb1.pth"

# The variants worth considering here. Aerial is the only one whose corpus
# matches CEI's sub-metre nadir RGB; the Sentinel-2 and Landsat backbones are
# pretrained at 10 m and 30 m and are listed only so the choice is visible.
KNOWN = {
    "aerial_swinb_si-e4169eb1.pth": "Swin-V2-B, NAIP aerial RGB, single-image (recommended)",
    "aerial_swinb_mi-326d69e1.pth": "Swin-V2-B, NAIP aerial RGB, multi-image",
    "sentinel2_swinb_si_rgb-156a98d5.pth": "Swin-V2-B, Sentinel-2 RGB at 10 m",
    "landsat_swinb_si-4af978f6.pth": "Swin-V2-B, Landsat at 30 m",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=DEFAULT, help=f"File to fetch (default: {DEFAULT}).")
    parser.add_argument("--out-dir", default="pretrain_weight")
    parser.add_argument("--list", action="store_true", help="List known variants and exit.")
    args = parser.parse_args()

    if args.list:
        for name, description in KNOWN.items():
            print(f"{name:38} {description}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    destination = os.path.join(args.out_dir, args.file)
    if os.path.exists(destination):
        print(f"Already present: {destination}")
        return

    url = f"{REPO}/{args.file}"
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, destination)
    print(f"Saved to {destination} ({os.path.getsize(destination) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
