import os
import datetime
import shutil
import yaml
import random
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from train import Trainer
from inference import Inferencer


def torch_fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def save_loss_graph(trainer, file_path):
    plt.figure()
    plt.plot([abs(loss) for loss in trainer.losses], label='Training Loss')
    plt.plot([abs(val_loss) for val_loss in trainer.val_losses], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.xlim(0, 300)
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(file_path)
    plt.close()

def load_data(pred_paths, truth_paths):
    all_y_pred = []
    all_y_test = []

    for pred_path, truth_path in zip(pred_paths, truth_paths):
        y_pred = torch.load(pred_path, weights_only=False).float()
        y_test = torch.load(truth_path, weights_only=False).float()
        all_y_pred.append(y_pred)
        all_y_test.append(y_test)
        
    return torch.cat(all_y_pred, dim=0), torch.cat(all_y_test, dim=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        params = yaml.safe_load(stream)

    seed = params['seed']
    cross_validation_folds = params.get('cross_validation_folds', 10) 
    data_path_base = params['data_path']  
    base_dir = params.get('result_base_dir', "data/result")  
    output_channels = params['output_channels']

    torch_fix_seed(seed)

    current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    experiment_dir = os.path.join(base_dir, f"exp_{current_time}")
    os.makedirs(experiment_dir, exist_ok=True)

    loss_graph_dir = os.path.join(experiment_dir, 'loss_graph')
    pred_data_dir = os.path.join(experiment_dir, 'pred_data')
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoint')
    os.makedirs(loss_graph_dir, exist_ok=True)
    os.makedirs(pred_data_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    shutil.copy(args.config, os.path.join(experiment_dir, 'config.yaml'))

    pred_paths = []
    truth_paths = []
    training_times = []
    inference_times = []
    pred_image_counts = []
    best_epochs = []
    psnr_results = []
    ssim_results = []
    per_channel_psnr_results = []
    per_channel_ssim_results = []

    for i in range(cross_validation_folds):
        dir_name = f'crossValidation_all{i}'
        print(f"Starting cross validation {i}")
        
        params['data_path'] = os.path.join(data_path_base, dir_name)
        
        trainer = Trainer(params)
        trainer.train()
        training_times.append(trainer.time)

        inferencer = Inferencer(params)
        inferencer.model.load_state_dict(trainer.best_model)
        inferencer.model.eval()
        inferencer.inference()
        inference_times.append(inferencer.time)
        best_epochs.append(trainer.best_epoch)

        y_pred_path = os.path.join(params['data_path'], 'y_pred.pt')
        with torch.inference_mode():
            pred_tensor = torch.load(y_pred_path, map_location='cpu', weights_only=False)
        pred_image_counts.append(pred_tensor.shape[0])
        del pred_tensor

        psnr_results.append(inferencer.psnr)
        ssim_results.append(inferencer.ssim)
        per_channel_psnr_results.append(inferencer.psnr_list)
        per_channel_ssim_results.append(inferencer.ssim_list)

        loss_graph_path = os.path.join(loss_graph_dir, f'cv{i}_loss.png')
        save_loss_graph(trainer, loss_graph_path)
        
        pred_file = f'cv{i}_pred.pt'
        pred_paths.append(os.path.join(pred_data_dir, pred_file))
        shutil.copy(os.path.join(params['data_path'], 'y_pred.pt'), os.path.join(pred_data_dir, pred_file))

        truth_paths.append(os.path.join(params['data_path'], 'y_test.pt'))

        model_path = os.path.join(checkpoint_dir, f'cv{i}_model.pth')
        torch.save(trainer.best_model, model_path)

        del trainer
        del inferencer
        torch.cuda.empty_cache()

    channel_psnr_means = []
    channel_ssim_means = []
    num_channels = len(per_channel_psnr_results[0])

    for c in range(num_channels):
        fold_psnrs_for_ch = [per_channel_psnr_results[i][c] for i in range(cross_validation_folds)]
        fold_ssims_for_ch = [per_channel_ssim_results[i][c] for i in range(cross_validation_folds)]

        ch_psnr_mean = float(np.mean(fold_psnrs_for_ch))
        ch_ssim_mean = float(np.mean(fold_ssims_for_ch))

        channel_psnr_means.append(ch_psnr_mean)
        channel_ssim_means.append(ch_ssim_mean)

    combined_psnr = float(np.mean(channel_psnr_means))
    combined_ssim = float(np.mean(channel_ssim_means))

    formatted_results = ", ".join(f"({psnr_result:.6f}, {ssim_result:.6f})" for psnr_result, ssim_result in zip(psnr_results, ssim_results))
    print(f"Individual Results: {formatted_results}")
    print(f'Combined PSNR: {combined_psnr}')
    print(f'Combined SSIM: {combined_ssim}')

    with open(os.path.join(experiment_dir, 'result.txt'), 'w') as f:
        f.write(f"Individual Results: {formatted_results}\n")
        f.write(f"Combined PSNR: {combined_psnr}\n")
        f.write(f"Combined SSIM: {combined_ssim}\n\n")
        
        channel_names = ["Sv03", "Sv05", "Sv07", "Sv10"]
        for c in range(len(per_channel_psnr_results[0])):
            ch_index = output_channels[c]
            if ch_index < len(channel_names):
                ch_name = channel_names[ch_index]
            else:
                ch_name = f"Ch{ch_index}"

            fold_psnrs_for_ch = [per_channel_psnr_results[i][c] for i in range(cross_validation_folds)]
            fold_ssims_for_ch = [per_channel_ssim_results[i][c] for i in range(cross_validation_folds)]

            f.write(f"{ch_name}:\n")
            individual_str = ", ".join(f"({ps:.6f}, {ss:.6f})" 
                                       for ps, ss in zip(fold_psnrs_for_ch, fold_ssims_for_ch))
            f.write(f"  Individual Results: {individual_str}\n")
            ch_psnr_mean = channel_psnr_means[c]
            ch_ssim_mean = channel_ssim_means[c]
            f.write(f"  Combined PSNR: {ch_psnr_mean}\n")
            f.write(f"  Combined SSIM: {ch_ssim_mean}\n\n")

        f.write("Individual Results: " + ",".join(map(str, best_epochs)) + "\n")
        f.write(f"Average Best Epoch: {np.mean(best_epochs)}\n\n")
        
        f.write(f"Total Training time: {sum(training_times):.2f} seconds\n")
        f.write(f"Total Inference time: {sum(inference_times):.4f} seconds\n")
        f.write(f"Avarage Inference time: {(sum(inference_times)*1000)/sum(pred_image_counts):.4f} milliseconds\n")

if __name__ == '__main__':
    main()
