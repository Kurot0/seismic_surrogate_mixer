import os
import re
import yaml
import argparse
import time
import torch
import importlib
import numpy as np
from PIL import Image
from evaluate import Evaluator


class Inferencer:
    def __init__(self, params):
        dev_cfg = params.get("device", "cuda")
        self.device = torch.device(f"cuda:{dev_cfg}" if isinstance(dev_cfg, int) else dev_cfg) if torch.cuda.is_available() or dev_cfg=="cpu" else torch.device("cpu")

        data_path = params['data_path']
        underground_data_path = params['underground_data_path']

        self.x_test = torch.load(data_path + '/x_test.pt', weights_only=False).float().to(self.device)
        self.y_test = torch.load(data_path + '/y_test.pt', weights_only=False).float().to(self.device)
        self.y_pred_path = data_path + '/y_pred.pt'

        output_channels = params['output_channels']
        self.y_test = self.y_test[:, output_channels, :, :]

        underground_channels = params['underground_channels']
        self.underground_data = torch.load(underground_data_path, weights_only=False).float().to(self.device)[:, underground_channels, :, :]
        min_val = torch.min(self.underground_data)
        max_val = torch.max(self.underground_data)
        self.underground_data = (self.underground_data - min_val) / (max_val - min_val)

        self.batch_size = params['batch_size']
        test_dataset = torch.utils.data.TensorDataset(self.x_test)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        model_module = importlib.import_module(params['model_module'])
        model_class = getattr(model_module, params['model_class'])

        out_chans = len(output_channels)
        self.model = model_class(out_chans=out_chans, **params).to(self.device)

        self.psnr = None
        self.ssim = None
        self.psnr_list = None
        self.ssim_list = None
        self.time = 0

        self.mask = torch.from_numpy(np.array(Image.open(params['mask_path']))).float().to(self.device) / 255.0
        self.evaluator = Evaluator(self.mask)

    def inference(self):
        self.model.eval()
        all_outputs = []
        start_time = time.time()
        with torch.no_grad():
            for inputs in self.test_loader:
                inputs = inputs[0].to(self.device)
                underground_data = self.underground_data.to(self.device)
                outputs = self.model(inputs, underground_data)
                all_outputs.append(outputs)
        end_time = time.time()
        self.time = end_time - start_time

        y_pred = torch.cat(all_outputs, dim=0)
        torch.save(y_pred.cpu(), self.y_pred_path)

        self.psnr, self.ssim, self.psnr_list, self.ssim_list = self.evaluator.evaluate(y_pred, self.y_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=str)
    args = parser.parse_args()

    exp_dir = os.path.abspath(args.exp_dir)
    ckpt_dir = os.path.join(exp_dir, "checkpoint")
    cfg_path = os.path.join(exp_dir, "config.yaml")

    with open(cfg_path, "r") as f:
        params = yaml.safe_load(f)

    data_path_base = params["data_path"]

    channel_names = ["Sv03", "Sv05", "Sv07", "Sv10"]
    output_channels = params['output_channels']
    num_channels = len(output_channels)

    psnr_results = []
    ssim_results = []
    per_channel_psnr_results = [[] for _ in range(num_channels)]
    per_channel_ssim_results = [[] for _ in range(num_channels)]
    best_epochs = []

    def target_models():
        for fname in sorted(os.listdir(ckpt_dir)):
            if fname.endswith(".pth"):
                m = re.match(r"cv(\d+)_model\.pth", fname)
                fold = int(m.group(1)) if m else None
                yield fname, fold

    print(f"Evaluation results for: {args.exp_dir}")

    for model_fname, fold in target_models():
        model_path = os.path.join(ckpt_dir, model_fname)
        if fold is not None:
            params["data_path"] = os.path.join(
                data_path_base, f"crossValidation_all{fold}"
            )

        inf = Inferencer(params)
        state = torch.load(model_path, map_location=inf.device)
        inf.model.load_state_dict(state)

        inf.inference()

        psnr_results.append(inf.psnr)
        ssim_results.append(inf.ssim)

        for c in range(num_channels):
            per_channel_psnr_results[c].append(inf.psnr_list[c])
            per_channel_ssim_results[c].append(inf.ssim_list[c])

        del inf
        torch.cuda.empty_cache()

    print()
    formatted_results = ", ".join(
        f"({psnr:.6f}, {ssim:.6f})" for psnr, ssim in zip(psnr_results, ssim_results)
    )
    combined_psnr = float(np.mean(psnr_results))
    combined_ssim = float(np.mean(ssim_results))

    print(f"Individual Results: {formatted_results}")
    print(f"Combined PSNR: {combined_psnr}")
    print(f"Combined SSIM: {combined_ssim}")
    print()

    for c in range(num_channels):
        ch_index = output_channels[c]
        ch_name = channel_names[ch_index] if ch_index < len(channel_names) else f"Ch{ch_index}"

        fold_psnrs_for_ch = per_channel_psnr_results[c]
        fold_ssims_for_ch = per_channel_ssim_results[c]

        ch_psnr_mean = float(np.mean(fold_psnrs_for_ch))
        ch_ssim_mean = float(np.mean(fold_ssims_for_ch))

        print(f"{ch_name}:")
        individual_str = ", ".join(
            f"({ps:.6f}, {ss:.6f})" for ps, ss in zip(fold_psnrs_for_ch, fold_ssims_for_ch)
        )
        print(f"  Individual Results: {individual_str}")
        print(f"  Combined PSNR: {ch_psnr_mean}")
        print(f"  Combined SSIM: {ch_ssim_mean}")
        print()

if __name__ == "__main__":
    main()
