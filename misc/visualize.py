import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path
import pickle


PERIOD_MAP = {0: 3, 1: 5, 2: 7, 3: 10}
PERIOD_DIR = {ch: f"{sec}second" for ch, sec in PERIOD_MAP.items()}
LABEL_TRUE = "Ground truth"
LABEL_PRED = "Prediction"
FIGSIZE_INCH = (6, 6)
FIG_DPI = 200

def load_labels(labels_path: str):
    with open(labels_path, "rb") as f:
        return pickle.load(f)

def load_tensor(path: str):
    return torch.load(path, weights_only=False).numpy()

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

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

def infer_experiment_root_from_pred_path(pred_data_path: str):
    pred_dir = os.path.dirname(pred_data_path)
    exp_root = os.path.dirname(pred_dir)
    if exp_root == "" or exp_root == os.path.sep:
        return os.path.join("data")
    return exp_root

def infer_default_save_dir_from_pred_path(pred_data_path: str):
    exp_root = infer_experiment_root_from_pred_path(pred_data_path)
    fold_idx = infer_cv_index_from_pred_path(pred_data_path)
    return os.path.join(exp_root, f"images_cv{fold_idx}")

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

def pil_open_first_page(img_path: str) -> Image.Image:
    ext = os.path.splitext(img_path)[1].lower()
    if ext == '.pdf':
        images = convert_from_path(img_path)
        return images[0]
    else:
        return Image.open(img_path)

def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    if hasattr(draw, "textbbox"):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return (r - l), (b - t)
    elif hasattr(draw, "textsize"):
        return draw.textsize(text, font=font)
    else:
        if hasattr(font, "getbbox"):
            l, t, r, b = font.getbbox(text)
            return (r - l), (b - t)
        return (max(8, 9 * len(text)), 18)

def add_header_label(img: Image.Image, text: str, font_size: int | None = None, pad_y: int = 8) -> Image.Image:
    W, H = img.size
    if font_size is None:
        font_size = max(18, int(W * 0.055))
    font = None
    for name in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
        try:
            font = ImageFont.truetype(name, font_size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    dummy = Image.new("RGB", (1, 1), "white")
    d0 = ImageDraw.Draw(dummy)
    text_w, text_h = _measure_text(d0, text, font)
    header_h = text_h + pad_y * 2
    canvas = Image.new("RGB", (W, H + header_h), color="white")
    canvas.paste(img, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    x = (W - text_w) // 2
    y = (header_h - text_h) // 2
    draw.text((x, y), text, fill="black", font=font)
    return canvas

def concat_grid(two_rows: list[list[Image.Image]]) -> Image.Image:
    n_cols = max(len(two_rows[0]), len(two_rows[1]))
    for row in two_rows:
        while len(row) < n_cols:
            row.append(Image.new("RGB", (1, 1), color="white"))
    col_widths = [max(two_rows[0][c].size[0], two_rows[1][c].size[0]) for c in range(n_cols)]
    row_heights = [max(img.size[1] for img in row) for row in two_rows]
    total_w = sum(col_widths)
    total_h = sum(row_heights)
    canvas = Image.new("RGB", (total_w, total_h), "white")
    y = 0
    for r, row in enumerate(two_rows):
        x = 0
        for c, img in enumerate(row):
            w, h = img.size
            x_offset = x + (col_widths[c] - w) // 2
            y_offset = y + (row_heights[r] - h) // 2
            canvas.paste(img, (x_offset, y_offset))
            x += col_widths[c]
        y += row_heights[r]
    return canvas

def render_map_and_save(data_2d: np.ndarray, out_path: str, mask: np.ndarray | None,
                        apply_mask: int, show_colorbar: int):
    plt.figure(figsize=FIGSIZE_INCH, dpi=FIG_DPI)
    if apply_mask == 0 and mask is not None:
        masked = np.where(mask == 1, data_2d, np.nan)
        cmap = plt.get_cmap('Reds')
        cmap.set_bad(color='gray')
        plt.imshow(masked, cmap=cmap, interpolation='nearest', vmin=0, vmax=1.0)
    else:
        plt.imshow(data_2d, cmap='Reds', interpolation='nearest', vmin=0, vmax=1.0)
    if show_colorbar != 0:
        cbar = plt.colorbar()
        cbar.set_label('velocity response spectra [m/s]', rotation=90, labelpad=20, fontsize=14)
        cbar.ax.yaxis.set_label_position('right')
        cbar.ax.invert_yaxis()
        cbar.ax.tick_params(labelsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close()

def render_scatter_and_save(x_vals: np.ndarray, y_vals: np.ndarray,
                            out_path: str, xlabel: str, ylabel: str):
    plt.figure(figsize=FIGSIZE_INCH, dpi=FIG_DPI)
    plt.scatter(x_vals, y_vals, s=1)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.xticks(np.arange(0, 1.2, 0.2), fontsize=14)
    plt.yticks(np.arange(0, 1.2, 0.2), fontsize=14)
    plt.savefig(out_path, dpi=FIG_DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close()

def save_single_maps(data_4d: np.ndarray, save_dir_base: str, kind_name: str,
                     mask_path: str | None, apply_mask: int, show_colorbar: int,
                     model_name_for_filename: str, scenario_names: list[str]):
    ensure_dir(save_dir_base)
    mask = None
    if mask_path is not None and os.path.exists(mask_path):
        mask_img = Image.open(mask_path)
        mask = np.array(mask_img).astype(np.float32) / 255.0
    batch_size, num_channels, _, _ = data_4d.shape
    for i in range(batch_size):
        scenario = scenario_names[i]
        for ch in range(num_channels):
            sec = PERIOD_MAP.get(ch, f"ch{ch}")
            ch_dir = os.path.join(save_dir_base, kind_name, PERIOD_DIR[ch])
            ensure_dir(ch_dir)
            fname = f"{model_name_for_filename}_{scenario}_{sec}.png"
            out_path = os.path.join(ch_dir, fname)
            render_map_and_save(data_4d[i, ch], out_path, mask, apply_mask, show_colorbar)

def build_concat_grid_for_maps(true_imgs: list[Image.Image], pred_imgs: list[Image.Image], secs: list[int]) -> Image.Image:
    row_true = [add_header_label(img, f"{LABEL_TRUE} - {sec} s") for img, sec in zip(true_imgs, secs)]
    row_pred = [add_header_label(img, f"{LABEL_PRED} - {sec} s") for img, sec in zip(pred_imgs, secs)]
    return concat_grid([row_true, row_pred])

def build_concat_grid_for_scatters(true_row_imgs: list[Image.Image],
                                   pred_row_imgs: list[Image.Image],
                                   secs: list[int]) -> Image.Image:
    row_true = [add_header_label(img, f"{LABEL_TRUE} - {sec} s")
                for img, sec in zip(true_row_imgs, secs)]
    row_pred = [add_header_label(img, f"{LABEL_PRED} - {sec} s")
                for img, sec in zip(pred_row_imgs, secs)]
    return concat_grid([row_true, row_pred])

def save_maps_and_concat(true_data_path: str, pred_data_path: str, save_dir: str,
                         mask_path: str | None, apply_mask: int, show_colorbar: int,
                         model_name: str, scenario_names: list[str]):
    ensure_dir(save_dir)
    concat_dir = os.path.join(save_dir, "concat")
    ensure_dir(concat_dir)
    true_4d = load_tensor(true_data_path)
    pred_4d = load_tensor(pred_data_path)
    batch_size, num_channels, _, _ = true_4d.shape
    secs = [PERIOD_MAP.get(ch, f"ch{ch}") for ch in range(num_channels)]
    save_single_maps(true_4d, save_dir, "true", mask_path, apply_mask, show_colorbar, "true", scenario_names)
    save_single_maps(pred_4d, save_dir, "pred", mask_path, apply_mask, show_colorbar, model_name, scenario_names)
    for i in range(batch_size):
        scenario = scenario_names[i]
        true_imgs = []
        pred_imgs = []
        for ch in range(num_channels):
            sec = PERIOD_MAP.get(ch, f"ch{ch}")
            true_path = os.path.join(save_dir, "true", PERIOD_DIR[ch], f"true_{scenario}_{sec}.png")
            pred_path = os.path.join(save_dir, "pred", PERIOD_DIR[ch], f"{model_name}_{scenario}_{sec}.png")
            true_imgs.append(pil_open_first_page(true_path))
            pred_imgs.append(pil_open_first_page(pred_path))
        grid = build_concat_grid_for_maps(true_imgs, pred_imgs, secs)
        grid.save(os.path.join(concat_dir, f"{model_name}_{scenario}_concat.png"))

def save_scatters_and_concat(true_data_path: str, pred_data_path: str, save_dir: str,
                             mask_path: str, model_name: str, scenario_names: list[str]):
    ensure_dir(save_dir)
    concat_dir = os.path.join(save_dir, "concat")
    ensure_dir(concat_dir)
    true_4d = load_tensor(true_data_path)
    pred_4d = load_tensor(pred_data_path)
    batch_size, num_channels, H, W = true_4d.shape
    secs = [PERIOD_MAP.get(ch, f"ch{ch}") for ch in range(num_channels)]
    mask = np.array(Image.open(mask_path)).astype(np.float32) / 255.0
    valid = (mask == 1)
    for i in range(batch_size):
        scenario = scenario_names[i]
        row_true_imgs = []
        row_pred_imgs = []
        for ch in range(num_channels):
            sec = PERIOD_MAP.get(ch, f"ch{ch}")
            true_vals = true_4d[i, ch][valid]
            pred_vals = pred_4d[i, ch][valid]
            true_scatter_dir = os.path.join(save_dir, "true", PERIOD_DIR[ch])
            pred_scatter_dir = os.path.join(save_dir, "pred", PERIOD_DIR[ch])
            ensure_dir(true_scatter_dir)
            ensure_dir(pred_scatter_dir)
            sp_true_path = os.path.join(true_scatter_dir, f"true_{scenario}_{sec}_sp.png")
            render_scatter_and_save(true_vals, true_vals, sp_true_path, xlabel="true", ylabel="true")
            sp_pred_path = os.path.join(pred_scatter_dir, f"{model_name}_{scenario}_{sec}_sp.png")
            render_scatter_and_save(true_vals, pred_vals, sp_pred_path, xlabel="true", ylabel="predicted")
            row_true_imgs.append(pil_open_first_page(sp_true_path))
            row_pred_imgs.append(pil_open_first_page(sp_pred_path))
        grid = build_concat_grid_for_scatters(row_true_imgs, row_pred_imgs, secs)
        grid.save(os.path.join(concat_dir, f"{model_name}_{scenario}_concat_sp.png"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_data_path", type=str, help="Path to predicted data tensor")
    parser.add_argument("--true_data_path", type=str, default=None, help="Path to true data tensor. If not specified, inferred automatically from mode & fold.")
    parser.add_argument("--labels_path", type=str, default=None, help="Path to labels dictionary. If not specified, inferred automatically.")
    parser.add_argument("--save_dir", type=str, default=None, help="Root directory to save images. Defaults to a path inferred from pred_data_path and fold (images_cv{fold}/ under the experiment root).")
    parser.add_argument("--mask_path", type=str, default="data/sea_400.png", help="Path to mask image")
    parser.add_argument("--apply_mask", type=int, default=0, help="0: apply mask, non-zero: do not apply")
    parser.add_argument("--show_colorbar", type=int, default=0, help="0: no colorbar, non-zero: show colorbar")
    args = parser.parse_args()

    pred_data_path = args.pred_data_path
    mode, model = infer_mode_and_model_from_pred_path(pred_data_path)
    fold_idx = infer_cv_index_from_pred_path(pred_data_path)
    true_data_path = args.true_data_path or default_true_data_path(mode, fold_idx)
    labels_path = args.labels_path or default_labels_path(mode)
    save_dir = args.save_dir or infer_default_save_dir_from_pred_path(pred_data_path)
    ensure_dir(save_dir)
    labels_dict = load_labels(labels_path)
    scenario_names = get_scenario_names_for_fold(labels_dict, fold_idx)
    save_maps_and_concat(true_data_path, pred_data_path, save_dir, args.mask_path, args.apply_mask, args.show_colorbar, model, scenario_names)
    save_scatters_and_concat(true_data_path, pred_data_path, save_dir, args.mask_path, model, scenario_names)

if __name__ == "__main__":
    main()
