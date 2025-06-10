import copy
import time
import torch
import importlib
import numpy as np
from PIL import Image
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
        dev_cfg = params.get("device", "cuda")
        self.device = torch.device(f"cuda:{dev_cfg}" if isinstance(dev_cfg, int) else dev_cfg) if torch.cuda.is_available() or dev_cfg=="cpu" else torch.device("cpu")

        data_path = params['data_path']
        underground_data_path = params['underground_data_path']

        self.x_train = torch.load(data_path + '/x_train.pt', weights_only=False).float()
        self.y_train = torch.load(data_path + '/y_train.pt', weights_only=False).float()
        self.x_valid = torch.load(data_path + '/x_valid.pt', weights_only=False).float()
        self.y_valid = torch.load(data_path + '/y_valid.pt', weights_only=False).float()

        output_channels = params['output_channels']
        self.y_train = self.y_train[:, output_channels, :, :]
        self.y_valid = self.y_valid[:, output_channels, :, :]

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

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        start_time = time.time()
        for inputs, targets in self.data_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device) 
            underground_data = self.underground_data.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs, underground_data, train_flag=True)
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
                outputs = self.model(inputs, underground_data, train_flag=True)
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
            loss = self.train_one_epoch()
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
