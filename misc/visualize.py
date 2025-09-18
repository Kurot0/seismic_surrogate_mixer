import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from pdf2image import convert_from_path
import pickle


PERIOD_MAP = {0: 3, 1: 5, 2: 7, 3: 10}
PERIOD_DIR = {ch: f"{sec}second" for ch, sec in PERIOD_MAP.items()}

def load_labels(labels_path: str):
    with open(labels_path, "rb") as f:
        return pickle.load(f)

def infer_mode_and_model_from_pred_path(pred_data_path: str):
    m = re.search(r"exp_[^/\\]+_([A-Za-z0-9]+)_([A-Za-z0-9]+)", pred_data_path)
    if not m:
        raise ValueError(f"Cannot extract mode/model from pred_data_path: {pred_data_path}")
    mode, model = m.group(1), m.group(2)
    return mode, model

def infer_cv_index_from_pred_path(pred_data_path: str):
    m = re.search(r"cv(\d+)_pred\.pt$", pred_data_path)
    if not m:
        raise ValueError(f"Cannot extract fold index from pred_data_path: {pred_data_path}")
    return int(m.group(1))

def default_true_data_path(mode: str, fold_idx: int):
    return f"data/cv_data_{mode}/crossValidation_all{fold_idx}/y_test.pt"

def default_labels_path(mode: str):
    return f"data/labels_dictionary_{mode}.pkl"

def get_scenario_names_for_fold(labels_dict: dict, fold_idx: int):
    key = f"quakeData-all-crossValidation{fold_idx}"
    if key not in labels_dict:
        raise KeyError(f"Missing key in labels dictionary: {key}")
    fold_entry = labels_dict[key]
    if "test" not in fold_entry:
        raise KeyError(f"Key {key} does not contain 'test'")
    return list(fold_entry["test"])

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_tensor(path: str):
    return torch.load(path, weights_only=False).numpy()

def save_images(data_path, save_dir, mask_path, apply_mask, show_colorbar, model_name, scenario_names):
    ensure_dir(save_dir)
    data_4d = load_tensor(data_path)
    batch_size, num_channels, _, _ = data_4d.shape
    mask = None
    if apply_mask == 0 and mask_path is not None:
        mask = torch.from_numpy(np.array(Image.open(mask_path))).float() / 255.0
    for i in range(batch_size):
        images_for_concat = []
        scenario = scenario_names[i]
        for channel in range(num_channels):
            period = PERIOD_MAP.get(channel, f"ch{channel}")
            channel_dir = os.path.join(save_dir, PERIOD_DIR[channel])
            ensure_dir(channel_dir)
            data = data_4d[i, channel]
            plt.figure()
            if apply_mask == 0 and mask is not None:
                masked_data = np.where(mask == 1, data, np.nan)
                cmap = plt.get_cmap('Reds')
                cmap.set_bad(color='gray')
                plt.imshow(masked_data, cmap=cmap, interpolation='nearest', vmin=0, vmax=1.0)
            else:
                plt.imshow(data, cmap='Reds', interpolation='nearest', vmin=0, vmax=1.0)
            if show_colorbar != 0:
                cbar = plt.colorbar()
                cbar.set_label('velocity response spectra [m/s]', rotation=90, labelpad=20, fontsize=14)
                cbar.ax.yaxis.set_label_position('right')
                cbar.ax.invert_yaxis()
                cbar.ax.tick_params(labelsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            img_fname = f"{model_name}_{scenario}_{period}.png"
            img_path = os.path.join(channel_dir, img_fname)
            plt.savefig(img_path, bbox_inches='tight')
            plt.close()
            ext = os.path.splitext(img_path)[1].lower()
            if ext == '.pdf':
                images = convert_from_path(img_path)
                img = images[0]
            else:
                img = Image.open(img_path)
            images_for_concat.append(img)
        widths, heights = zip(*(img.size for img in images_for_concat))
        total_width = sum(widths)
        max_height = max(heights)
        concat_img = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in images_for_concat:
            concat_img.paste(img, (x_offset, 0))
            x_offset += img.width
        concat_img.save(os.path.join(save_dir, f"{model_name}_{scenario}_concat.png"))

def save_scatter_plots(true_data_path, pred_data_path, mask_path, save_dir, model_name, scenario_names):
    ensure_dir(save_dir)
    true_data = load_tensor(true_data_path)
    pred_data = load_tensor(pred_data_path)
    batch_size, num_channels, _, _ = true_data.shape
    mask = torch.from_numpy(np.array(Image.open(mask_path))).float() / 255.0
    for i in range(batch_size):
        scatter_imgs = []
        scenario = scenario_names[i]
        for channel in range(num_channels):
            period = PERIOD_MAP.get(channel, f"ch{channel}")
            channel_dir = os.path.join(save_dir, PERIOD_DIR[channel])
            ensure_dir(channel_dir)
            true_batch = true_data[i, channel]
            pred_batch = pred_data[i, channel]
            true_values = true_batch[mask == 1]
            pred_values = pred_batch[mask == 1]
            plt.figure(figsize=(6, 6))
            plt.scatter(true_values, pred_values, s=1)
            plt.xlabel('true', fontsize=14)
            plt.ylabel('predicted', fontsize=14)
            plt.xlim(0, 1.05)
            plt.ylim(0, 1.05)
            plt.xticks(np.arange(0, 1.2, 0.2), fontsize=14)
            plt.yticks(np.arange(0, 1.2, 0.2), fontsize=14)
            scatter_fname = f"{model_name}_{scenario}_{period}_sp.png"
            scatter_path = os.path.join(channel_dir, scatter_fname)
            plt.savefig(scatter_path)
            plt.close()
            scatter_imgs.append(Image.open(scatter_path))
        widths, heights = zip(*(img.size for img in scatter_imgs))
        total_width = sum(widths)
        max_height = max(heights)
        concat_scatter = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in scatter_imgs:
            concat_scatter.paste(img, (x_offset, 0))
            x_offset += img.width
        concat_scatter.save(os.path.join(save_dir, f"{model_name}_{scenario}_concat_sp.png"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_data_path", type=str, help="Path to predicted data")
    parser.add_argument("--true_data_path", type=str, default=None, help="Path to true data. If not specified, inferred automatically")
    parser.add_argument("--labels_path", type=str, default=None, help="Path to labels dictionary. If not specified, inferred automatically")
    parser.add_argument("--save_dir", type=str, default="data/images", help="Output directory for images")
    parser.add_argument("--mask_path", type=str, default="data/exp_data/sea_400.png", help="Path to mask image")
    parser.add_argument("--apply_mask", type=int, default=0, help="0: apply mask, non-zero: do not apply")
    parser.add_argument("--show_colorbar", type=int, default=0, help="0: no colorbar, non-zero: show colorbar")
    args = parser.parse_args()

    pred_data_path = args.pred_data_path
    mode, model = infer_mode_and_model_from_pred_path(pred_data_path)
    fold_idx = infer_cv_index_from_pred_path(pred_data_path)
    true_data_path = args.true_data_path or default_true_data_path(mode, fold_idx)
    labels_path = args.labels_path or default_labels_path(mode)
    labels_dict = load_labels(labels_path)
    scenario_names = get_scenario_names_for_fold(labels_dict, fold_idx)
    
    save_images(pred_data_path, args.save_dir, args.mask_path, args.apply_mask, args.show_colorbar, model, scenario_names)
    save_scatter_plots(true_data_path, pred_data_path, args.mask_path, args.save_dir, model, scenario_names)

if __name__ == "__main__":
    main()
