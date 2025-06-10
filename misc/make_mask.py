import os
import glob
from typing import List
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from PIL import Image


def _gather_files(patterns: List[str]) -> List[str]:
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    return sorted(files)

CH_PATTERNS = {
    0: ["data/raw_data/2022_Sv03s_LL/*.dat"],
    1: ["data/raw_data/2022_Sv05s_LL/*.dat"],
    2: ["data/raw_data/2022_Sv07s_LL/*.dat"],
    3: ["data/raw_data/2022_Sv10s_LL/*.dat"],
}
PAD_N, PAD_S, PAD_W, PAD_E = 4, 21, 6, 6  # (N,S,W,E)
TARGET_H, TARGET_W = 400, 400
OUT_DIR  = "data/exp_data"
PNG_PATH = os.path.join(OUT_DIR, "sea_sea.png")


def main() -> None:
    ch_files = {ch: _gather_files(pats) for ch, pats in CH_PATTERNS.items()}
    counts = [len(v) for v in ch_files.values()]
    if len(set(counts)) != 1:
        raise RuntimeError("Channel file counts differ!", counts)

    num_samples = counts[0]
    mask = np.zeros((TARGET_H, TARGET_W), dtype=bool)

    for idx in range(num_samples):
        dfs = [pd.read_csv(ch_files[ch][idx], names=["lon", "lat", "amp", "sid"]) for ch in range(4)]
        lats = np.unique(dfs[0]["lat"].values)[::-1]
        lons = np.unique(dfs[0]["lon"].values)
        h, w = len(lats), len(lons)

        sample = np.zeros((4, h, w), dtype=np.float32)
        for ch, df in enumerate(dfs):
            for row in df.itertuples(index=False):
                lon_idx = np.where(lons == row.lon)[0]
                lat_idx = np.where(lats == row.lat)[0]
                if lon_idx.size and lat_idx.size:
                    sample[ch, lat_idx[0], lon_idx[0]] = row.amp

        sample = np.pad(sample, ((0, 0), (PAD_N, PAD_S), (PAD_W, PAD_E)), mode="constant")
        zoom_factors = (1.0, TARGET_H / sample.shape[1], TARGET_W / sample.shape[2])
        sample = zoom(sample, zoom=zoom_factors, order=1)

        mask |= (sample != 0).any(axis=0)
        print(f"[{idx + 1}/{num_samples}] mask updated")

    mask_img = (mask.astype(np.uint8) * 255)

    os.makedirs(OUT_DIR, exist_ok=True)

    Image.fromarray(mask_img).save(PNG_PATH)
    print(f"Mask image saved → {PNG_PATH}")


if __name__ == "__main__":
    main()
