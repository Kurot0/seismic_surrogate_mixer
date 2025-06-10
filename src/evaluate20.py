import torch
import numpy as np
from PIL import Image
from ssimloss import SSIMLoss

class Evaluator:
    def __init__(self, true_data_path, pred_data_path, mask_path, eval_region):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.y_test = torch.load(true_data_path).float().to(self.device)
        self.y_pred = torch.load(pred_data_path).float().to(self.device)

        full_mask = torch.from_numpy(np.array(Image.open(mask_path))).float().to(self.device) / 255.0
        self.mask = torch.zeros_like(full_mask)
        self.mask[eval_region[1]:eval_region[3], eval_region[0]:eval_region[2]] = full_mask[eval_region[1]:eval_region[3], eval_region[0]:eval_region[2]]

    def evaluate(self):
        psnrs = []
        ssims = []

        for i in range(self.y_test.size(0)):
            y_test_masked = (self.y_test[i] * self.mask).unsqueeze(0)
            y_pred_masked = (self.y_pred[i] * self.mask).unsqueeze(0)

            extended_mask = self.mask.unsqueeze(0).expand_as(self.y_test[i])
            y_test_unmasked = self.y_test[i][extended_mask == 1]
            y_pred_unmasked = self.y_pred[i][extended_mask == 1]

            mse_loss = torch.nn.functional.mse_loss(y_test_unmasked, y_pred_unmasked)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse_loss))
            psnrs.append(psnr.cpu().detach().numpy())

            ssim_loss = SSIMLoss(mask=self.mask)
            ssim = -ssim_loss(y_test_masked, y_pred_masked)
            ssims.append(ssim.cpu().detach().numpy())

        return psnrs, ssims


def main():
    mask_path = 'data/sea.png'
    true_data_path = 'data/cv_data/crossValidation_all0/y_test.pt'
    pred_data_path = 'data/result/exp_240927012750/pred_data/cv0_pred.pt'
    eval_region = [240, 230, 285, 280]  # [x_min, y_min, x_max, y_max]

    evaluator = Evaluator(true_data_path, pred_data_path, mask_path, eval_region)
    psnrs, ssims = evaluator.evaluate()

    for i, (psnr, ssim) in enumerate(zip(psnrs, ssims)):
        print(f"{i+1:03d} PSNR:{psnr:.6f} SSIM:{ssim:.8f}")

    print(f"Average PSNR: {np.mean(psnrs)}")
    print(f"Average SSIM: {np.mean(ssims)}")


if __name__ == '__main__':
    main()
