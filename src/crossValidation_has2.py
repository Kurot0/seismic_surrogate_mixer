import os
import datetime
import shutil
import yaml
import random
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from inference import Inferencer
from evaluate import Evaluator

import copy
import time
import importlib
from ssimloss import SSIMLoss


class EarlyStopping:
    def __init__(self, patience, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class Trainer:
    def __init__(self, params):
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        data_path = params['data_path']
        underground_data_path = params['underground_data_path']

        self.x_train = torch.load(data_path + '/x_train.pt', weights_only=False).float()
        self.y_train = torch.load(data_path + '/y_train.pt', weights_only=False).float()
        self.x_valid = torch.load(data_path + '/x_valid.pt', weights_only=False).float()
        self.y_valid = torch.load(data_path + '/y_valid.pt', weights_only=False).float()

        output_channels = params['output_channels']
        underground_channels = params['underground_channels']
        self.underground_data = torch.load(underground_data_path, weights_only=False).float()[:, underground_channels, :, :]
        min_val = torch.min(self.underground_data)
        max_val = torch.max(self.underground_data)
        self.underground_data = (self.underground_data - min_val) / (max_val - min_val)

        self.batch_size = params['batch_size']
        self.num_epochs = params['num_epochs']
        self.patience = params['patience']
        self.learning_rate = params['learning_rate']

        dataset = torch.utils.data.TensorDataset(self.x_train, self.y_train)
        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = torch.utils.data.TensorDataset(self.x_valid, self.y_valid)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        model_module = importlib.import_module(params['model_module'])
        model_class = getattr(model_module, params['model_class'])

        out_chans = len(output_channels)
        self.model = model_class(out_chans=out_chans, **params).to(self.device)
        
        self.mask = torch.from_numpy(np.array(Image.open(params['mask_path']))).float().to(self.device) / 255.0

        self.criterion = SSIMLoss(mask=self.mask, multi_channel=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.early_stop_model = None
        self.best_model = None
        self.best_loss = float('inf')
        self.best_epoch = 0

        self.losses = []
        self.val_losses = [] 

        self.time = 0

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        start_time = time.time()
        for inputs, targets in self.data_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            underground_data = self.underground_data.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs, underground_data, epoch)  # Pass epoch number
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        end_time = time.time()
        self.time += end_time - start_time

        epoch_loss = running_loss / len(self.data_loader.dataset)
        return epoch_loss

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                underground_data = self.underground_data.to(self.device)
                outputs = self.model(inputs, underground_data, epoch=None)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
        
        val_loss = running_loss / len(self.val_loader.dataset)
        return val_loss       

    def train(self):
        early_stop = EarlyStopping(patience=self.patience)
        stop_triggered = False
        self.best_epoch = None
        early_stop_epoch = None
        for epoch in range(self.num_epochs):
            loss = self.train_one_epoch(epoch)
            val_loss = self.validate()
            print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {-loss}, Validation Loss: {-val_loss}')

            if val_loss < self.best_loss:
                self.best_model = copy.deepcopy(self.model.state_dict())
                self.best_loss = val_loss
                self.best_epoch = epoch + 1

            early_stop(val_loss)
            if early_stop.early_stop and not stop_triggered:
                print("Early stopping triggered")
                stop_triggered = True
                self.early_stop_model = copy.deepcopy(self.model.state_dict())
                early_stop_epoch = epoch + 1
                break

            self.losses.append(loss)
            self.val_losses.append(val_loss)

        if self.early_stop_model is not None:
            print(f"The best model : epoch {self.best_epoch}, Early stopping : epoch {early_stop_epoch}")
        else:
            print(f"The best model : epoch {self.best_epoch}")


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
        f.write(f"Avarage Inference time: {sum(inference_times)/0.36:.4f} milliseconds\n")

if __name__ == '__main__':
    main()
