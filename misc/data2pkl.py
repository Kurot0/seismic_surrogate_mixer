import os
import glob
import csv
import pickle
import numpy as np
import pandas as pd
from scipy.ndimage import zoom


def _gather_files(patterns):
    files: list[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    return sorted(files)


def process_dat_files(
    dat_paths_channel0: list[str],
    dat_paths_channel1: list[str],
    dat_paths_channel2: list[str],
    dat_paths_channel3: list[str],
    output_dir: str,
    output_filename: str,
):
    
    ch0_files = _gather_files(dat_paths_channel0)
    ch1_files = _gather_files(dat_paths_channel1)
    ch2_files = _gather_files(dat_paths_channel2)
    ch3_files = _gather_files(dat_paths_channel3)

    if not (len(ch0_files) == len(ch1_files) == len(ch2_files) == len(ch3_files)):
        raise ValueError("Number of .dat files mismatch among channels.")

    num_samples   = len(ch0_files)
    num_channels  = 4
    target_h, target_w = 400, 400

    images, labels = [], []

    for i in range(num_samples):
        paths = [ch0_files[i], ch1_files[i], ch2_files[i], ch3_files[i]]
        dfs   = [pd.read_csv(p, names=["lon", "lat", "amp", "sid"]) for p in paths]

        lats = np.unique(dfs[0]["lat"].values)[::-1]
        lons = np.unique(dfs[0]["lon"].values)
        h, w = len(lats), len(lons)

        sample = np.zeros((num_channels, h, w), dtype=np.float32)
        for ch, df in enumerate(dfs):
            for row in df.itertuples(index=False):
                lon_idx = np.where(lons == row.lon)[0]
                lat_idx = np.where(lats == row.lat)[0]
                if lon_idx.size and lat_idx.size:
                    sample[ch, lat_idx[0], lon_idx[0]] = row.amp

        if "2022" in paths[0]:
            sample = np.pad(sample, ((0, 0), (4, 21), (6, 6)), mode="constant")
        else:
            sample = sample[:, 67:-51, 92:-16]

        zoom_factors = (1.0, target_h / sample.shape[1], target_w / sample.shape[2])
        sample = zoom(sample, zoom=zoom_factors, order=1)

        images.append(sample)
        labels.append(os.path.basename(paths[0]).split(".")[0])
        print(f"Processing DAT file {i+1}/{num_samples}: {paths[0]}")

    data_4d = np.stack(images, axis=0)

    data_4d[data_4d > 8] = 8

    ch_min = data_4d.min(axis=(0, 2, 3), keepdims=True)
    ch_max = data_4d.max(axis=(0, 2, 3), keepdims=True)
    data_4d = (data_4d - ch_min) / (ch_max - ch_min + 1e-8)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_filename)
    with open(out_path, "wb") as f:
        pickle.dump({"images": data_4d, "labels": labels}, f)
    print(f"Saved .dat data to: {out_path}")


def process_csv_files(
    special_12_files: set[str],
    csv_paths: list[str],
    output_dir: str,
    output_filename: str,
):

    all_csv_files = _gather_files(csv_paths)
    if not all_csv_files:
        print("No .csv files found.")
        return

    def load_baseline_data_for_csv(csv_file_path):
        base_dir = os.path.dirname(csv_file_path)
        base_name = os.path.basename(csv_file_path)

        if "2013_calc_scenario" in base_dir:
            max_data = np.genfromtxt(
                "data/raw_data/2013_calc_scenario/TF111_h01-add-ase_nm-vr2_fm1_sd1.csv",
                delimiter=","
            )
            padding_type = 1
        else:
            if base_name in special_12_files:
                sf010 = np.genfromtxt(
                    "data/raw_data/source_R04_done/SF010_h03_adm_nm_vr1_fm1_sd1_rrrr.csv",
                    delimiter=","
                )
                ts010 = np.genfromtxt(
                    "data/raw_data/source_R04_done/TS010_h03_adm_nm_vr1_fm1_sd1_rrrr.csv",
                    delimiter=","
                )
                max_data = np.concatenate([sf010, ts010], axis=0)
                padding_type = 3
            else:
                max_data = np.genfromtxt(
                    "data/raw_data/source_R04_done/TF111_h03_adm_nm_vr1_as1_nm_vr1_fm1_sd1_rrrr.csv",
                    delimiter=","
                )
                padding_type = 2

            max_data = max_data[:, :-1]

        return max_data, padding_type

    def get_padding_sizes(padding_type):
        if padding_type == 1:
            return (488, 13, 48, 477)
        elif padding_type == 2:
            return (488, 14, 47, 473)
        elif padding_type == 3:
            return (501, 59, 88, 480)
        
        # if padding_type == 1:
        #     return (191, 725, 485, 560)
        # elif padding_type == 2:
        #     return (192, 725, 481, 556)
        # elif padding_type == 3:
        #     return (237, 738, 525, 563)

    csv_images_list = []
    csv_filenames = []

    for file_index, csv_file in enumerate(all_csv_files):
        print(f"Processing CSV file {file_index + 1}/{len(all_csv_files)}: {csv_file}")
        base_name = os.path.basename(csv_file).split('.')[0]
        base_dir = os.path.dirname(csv_file)

        max_data, padding_type = load_baseline_data_for_csv(csv_file)
        
        csv_lat = np.unique(max_data[:, 0])[::-1]
        csv_lon = np.unique(max_data[:, 1])

        num_channels = max_data.shape[1] - 3

        tmp_4d = np.zeros((1, num_channels, len(csv_lat), len(csv_lon)), dtype=np.float32)

        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            data = [row for row in reader]

        if ("source_R03" in base_dir) or ("source_R04_done" in base_dir):
            new_data = []
            for row in data:
                row = row[:-1]
                new_data.append(row)
            data = new_data

        for i, row in enumerate(data):
            if len(row) < max_data.shape[1]:
                row += ['0.0'] * (max_data.shape[1] - len(row))
            row = row[:max_data.shape[1]]
            data[i] = row

        data = np.array(data, dtype=float)

        if data.shape[1] > 7:
            data[:, 6] = data[:, 6] * 10**data[:, 7]
            data = np.delete(data, 7, axis=1)

        for row_i in range(data.shape[0]):
            lat_val = data[row_i, 0]
            lon_val = data[row_i, 1]
            vals = data[row_i, 2:2+num_channels]

            lat_idx = np.where(csv_lat == lat_val)[0]
            lon_idx = np.where(csv_lon == lon_val)[0]
            if len(lat_idx) > 0 and len(lon_idx) > 0:
                tmp_4d[0, :, lat_idx[0], lon_idx[0]] = vals

        channels_to_remove = [6, 7, 8, 9, 10, 11]
        tmp_4d = np.delete(tmp_4d, channels_to_remove, axis=1)

        top, bottom, left, right = get_padding_sizes(padding_type)
        tmp_4d = np.pad(
            tmp_4d,
            pad_width=((0, 0), (0, 0), (top, bottom), (left, right)),
            mode='constant'
        )

        new_h, new_w = 720, 896
        # new_h, new_w = 944, 1184
        oh, ow = tmp_4d.shape[2], tmp_4d.shape[3]
        zoom_factors = (1.0, 1.0, float(new_h) / oh, float(new_w) / ow)
        tmp_4d = zoom(tmp_4d, zoom=zoom_factors, order=1)

        csv_images_list.append(tmp_4d[0]) 
        csv_filenames.append(base_name)

    csv_data_4d = np.stack(csv_images_list, axis=0)
    channel_min = csv_data_4d.min(axis=(0, 2, 3), keepdims=True)
    channel_max = csv_data_4d.max(axis=(0, 2, 3), keepdims=True)
    csv_data_4d = (csv_data_4d - channel_min) / (channel_max - channel_min + 1e-8)

    csv_output_file = os.path.join(output_dir, output_filename)
    csv_data_dict = {
        'images': csv_data_4d,
        'labels': csv_filenames
    }
    with open(csv_output_file, 'wb') as f:
        pickle.dump(csv_data_dict, f)
    print(f"Saved .csv data to: {csv_output_file}")


def main():
    dat_paths_channel0 = [
        'data/raw_data/2013_Sv03s_LL/*.dat',
        'data/raw_data/2021_Sv03s_LL/*.dat',
        'data/raw_data/2022_Sv03s_LL/*.dat',
    ]
    dat_paths_channel1 = [
        'data/raw_data/2013_Sv05s_LL/*.dat',
        'data/raw_data/2021_Sv05s_LL/*.dat',
        'data/raw_data/2022_Sv05s_LL/*.dat',
    ]
    dat_paths_channel2 = [
        'data/raw_data/2013_Sv07s_LL/*.dat',
        'data/raw_data/2021_Sv07s_LL/*.dat',
        'data/raw_data/2022_Sv07s_LL/*.dat',
    ]
    dat_paths_channel3 = [
        'data/raw_data/2013_Sv10s_LL/*.dat',
        'data/raw_data/2021_Sv10s_LL/*.dat',
        'data/raw_data/2022_Sv10s_LL/*.dat',
    ]

    csv_paths = [
        'data/raw_data/2013_calc_scenario/*.csv',
        'data/raw_data/source_R03/*.csv',
        'data/raw_data/source_R04_done/*.csv'
    ]
    special_12_files = {
        "AS010_h01_adm_nm_vr1_fm1_sd1_rrrr.csv",
        "AS010_h02_adm_nm_vr1_fm1_sd1_rrrr.csv",
        "AS010_h03_adm_nm_vr1_fm1_sd1_rrrr.csv",
        "SF010_h03_adm_nm_vr1_fm1_sd1_rrrr.csv",
        "SF010_h15_adm_nm_vr1_fm1_sd1_rrrr.csv",
        "SF010_h16_adm_nm_vr1_fm1_sd1_rrrr.csv",
        "SO010_h03_adm_nm_vr1_fm1_sd1_rrrr.csv",
        "SO010_h04_adm_nm_vr1_fm1_sd1_rrrr.csv",
        "SO010_h05_adm_nm_vr1_fm1_sd1_rrrr.csv",
        "TS010_h03_adm_nm_vr1_fm1_sd1_rrrr.csv",
        "TS010_h11_adm_nm_vr1_fm1_sd1_rrrr.csv",
        "TS010_h12_adm_nm_vr1_fm1_sd1_rrrr.csv",
    }

    output_dir = 'data/prep_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_dat_files(
        dat_paths_channel0=dat_paths_channel0,
        dat_paths_channel1=dat_paths_channel1,
        dat_paths_channel2=dat_paths_channel2,
        dat_paths_channel3=dat_paths_channel3,
        output_dir=output_dir,
        output_filename='Sv_LL.pkl'  
    )

    process_csv_files(
        special_12_files=special_12_files,
        csv_paths=csv_paths,
        output_dir=output_dir,
        output_filename='source.pkl'
    )


if __name__ == "__main__":
    main()
