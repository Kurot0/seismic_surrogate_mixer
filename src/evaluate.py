import os
import torch
import numpy as np
from PIL import Image
from ssimloss import SSIMLoss


class Evaluator:
    def __init__(self, mask):
        self.mask = mask

    def evaluate(self, y_pred, y_test):
        extended_mask = self.mask.expand_as(y_test)
        
        _, C, _, _ = y_pred.shape

        psnr_list = []
        ssim_list = []

        ssim_loss_fn = SSIMLoss(mask=self.mask, multi_channel=False)  

        for c in range(C):
            y_test_c = y_test[:, c:c+1, :, :]
            y_pred_c = y_pred[:, c:c+1, :, :]
            y_test_c_unmasked = y_test_c[extended_mask[:, c:c+1, :, :] == 1]
            y_pred_c_unmasked = y_pred_c[extended_mask[:, c:c+1, :, :] == 1]

            mse_loss_c = torch.nn.functional.mse_loss(y_test_c_unmasked, y_pred_c_unmasked)
            pixel_range = 1.0
            psnr_c = 20 * torch.log10(pixel_range / torch.sqrt(mse_loss_c))
            psnr_list.append(psnr_c.item())

            ssim_val_c = -ssim_loss_fn(y_pred_c, y_test_c)
            ssim_list.append(ssim_val_c.item())

        psnr_avg = float(np.mean(psnr_list))
        ssim_avg = float(np.mean(ssim_list))

        return psnr_avg, ssim_avg, psnr_list, ssim_list

def load_data(pred_paths, truth_paths):
    all_y_pred = []
    all_y_test = []

    for pred_path, truth_path in zip(pred_paths, truth_paths):
        y_pred = torch.load(pred_path).float()
        y_test = torch.load(truth_path).float()
        all_y_pred.append(y_pred)
        all_y_test.append(y_test)
        
    return torch.cat(all_y_pred, dim=0), torch.cat(all_y_test, dim=0)

def main():
    mask_path = 'data/sea.png'
    base_truth_dir = 'data/cv_data/crossValidation_all{}/y_test.pt'
    base_pred_dir = 'data/cv_data/crossValidation_all{}/y_pred.pt'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mask = torch.from_numpy(np.array(Image.open(mask_path))).float().to(device) / 255.0

    pred_paths = [base_pred_dir.format(i) for i in range(10)]
    truth_paths = [base_truth_dir.format(i) for i in range(10)]

    y_pred, y_test = load_data(pred_paths, truth_paths)
    y_pred = y_pred.to(device)
    y_test = y_test.to(device)

    evaluator = Evaluator(mask)
    psnr, ssim = evaluator.evaluate(y_pred, y_test)

    print(f'Combined PSNR: {psnr}')
    print(f'Combined SSIM: {ssim}')

if __name__ == '__main__':
    main()
