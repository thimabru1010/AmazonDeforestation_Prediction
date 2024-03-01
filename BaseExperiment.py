import torch
from tqdm import tqdm
from openstl.models.simvp_model import SimVP_Model
from segmentation_models_pytorch.losses import FocalLoss
import torch.nn as nn
import torch.optim as optm
import os
import json
from metrics import confusion_matrix, f1_score
import numpy as np
import torch.nn.functional as F
from torchsummary import summary
from preprocess import load_tif_image, preprocess_patches, divide_pred_windows

from sklearn.metrics import f1_score as skf1_score
from CustomLosses import WMSELoss, WMAELoss

class BaseExperiment():
    def __init__(self, trainloader, valloader, custom_model_config, custom_training_config, seed=42):
        # TODO: wrap into a function to create work dir
        # Create work dir
        self.work_dir_path = os.path.join('work_dirs', custom_training_config['ex_name'])
        if not os.path.exists(self.work_dir_path):
            os.makedirs(self.work_dir_path)
            
        self._save_json(custom_model_config, 'model_config.json')
        self._save_json(custom_training_config, 'model_training.json')
            
        self.epochs = custom_training_config['epoch']
        self.patience = custom_training_config['patience']
        self.delta = custom_training_config['delta']
        self.device = "cuda:0"
        in_shape = custom_training_config['in_shape']
        # img_full_shape = custom_training_config['img_shape']
        torch.manual_seed(seed)
        
        print('Input shape:', in_shape)
        self.model = self._build_model(in_shape, None, custom_model_config)
        
        print(summary(self.model, tuple(in_shape)))
        
        self.optm = optm.Adam(self.model.parameters(), lr=custom_training_config['lr'])
        
        if custom_training_config['amazon_mask']:
            if custom_training_config['pixel_size'] == '25K':
                mask = load_tif_image('data/IBAMA_INPE/25K/INPE/tiff/mask.tif')
            elif custom_training_config['pixel_size'] == '1K':
                pass
                # mask = mask_patches
                # mask = load_tif_image('data/IBAMA_INPE/1K/tiff_filled/mask.tif')
                # xcut = (mask.shape[0] // custom_training_config['patch_size']) * custom_training_config['patch_size']
                # ycut = (mask.shape[1] // custom_training_config['patch_size']) * custom_training_config['patch_size']
                # mask = mask[:img_full_shape[1], :img_full_shape[2]]
            # mask = None
            self.loss = WMSELoss(weight=1)
        else:
            self.loss = nn.MSELoss()
            # mask = None
            
        self.mae = nn.L1Loss()
        
        self.trainloader = trainloader
        self.valloader = valloader
        
    def _build_model(self, in_shape, nclasses, custom_model_config):
        return SimVP_Model(in_shape=in_shape, nclasses=nclasses, **custom_model_config).to(self.device)
    
    def _save_json(self, data, filename):
        with open(os.path.join(self.work_dir_path, filename), 'w') as f:
            json.dump(data, f)
            
    def train_one_epoch(self):
        train_loss = 0
        self.model.train(True)
        for inputs, labels in tqdm(self.trainloader):
            # Zero your gradients for every batch!
            self.optm.zero_grad()
            
            y_pred = self.model(inputs.to(self.device))
            # Get only the first temporal channel
            y_pred = y_pred[:, 0].contiguous().unsqueeze(1)
            
            loss = self.loss(y_pred, labels.to(self.device))
            loss.backward()
            
            # Adjust learning weights
            self.optm.step()
            
            train_loss += loss.detach()
        train_loss = train_loss / len(self.trainloader)
        
        return train_loss

    def validate_one_epoch(self):
        val_loss = 0
        val_mae = 0
        self.model.eval()
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for inputs, labels in tqdm(self.valloader):
                y_pred = self.model(inputs.to(self.device))
                # Get only the first temporal channel
                y_pred = y_pred[:, 0].contiguous().unsqueeze(1)
                loss = self.loss(y_pred, labels.to(self.device))
                mae = self.mae(y_pred, labels.to(self.device))
                
                val_loss += loss.detach()
                val_mae += mae.detach()
            
        val_loss = val_loss / len(self.valloader)
        val_mae = val_mae / len(self.valloader)
        
        return val_loss, val_mae
    
    def train(self):
        min_val_loss = float('inf')
        early_stop_counter = 0
        for epoch in range(self.epochs):
            
            train_loss = self.train_one_epoch()
            
            val_loss, val_mae = self.validate_one_epoch()
            
            if val_loss + self.delta < min_val_loss:
                min_val_loss = val_loss
                early_stop_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.work_dir_path, 'checkpoint.pth'))
            else:
                early_stop_counter += 1
                print(f"Val loss didn't improve! Early Stopping counter: {early_stop_counter}")
            
            if early_stop_counter >= self.patience:
                print(f'Early Stopping! Early Stopping counter: {early_stop_counter}')
                break
            
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f} | Validation Loss = {val_loss:.6f} | Validation MAE = {val_mae:.6f}")

def _build_model(in_shape, nclasses, custom_model_config, device):
    return SimVP_Model(in_shape=in_shape, nclasses=nclasses, **custom_model_config).to(device)

def test_model(testloader, custom_training_config, custom_model_config):
    work_dir_path = os.path.join('work_dirs', custom_training_config['ex_name'])
    device = "cuda:0"
    in_shape = custom_training_config['in_shape']
        
    model = _build_model(in_shape, None, custom_model_config, device)
    model.load_state_dict(torch.load(os.path.join(work_dir_path, 'checkpoint.pth')))
    # model.load_state_dict(os.path.join(work_dir_path, 'checkpoint.pth')).to(device)
    model.eval()
    
    # mse = nn.MSELoss()
    mse = WMSELoss(weight=1)
    mae = WMAELoss(weight=1)
    # mae = nn.L1Loss()
    
    test_loss = 0.0
    test_mae = 0.0
    # cm = np.zeros((2, 2), dtype=int)
    # Disable gradient computation and reduce memory consumption.
    skip_cont = 0
    with torch.no_grad():
        for inputs, labels in tqdm(testloader):
            # Check if all pixels are -1
            if torch.all(labels == -1):
                skip_cont += 1
                continue
            y_pred = model(inputs.to(device))
            # Get only the first temporal channel
            y_pred = y_pred[:, 0].contiguous().unsqueeze(1)
            
            loss = mse(y_pred, labels.to(device))
            _mae = mae(y_pred, labels.to(device))
                
            test_loss += loss.detach()
            test_mae += _mae.detach()

        test_loss = test_loss / (len(testloader) - skip_cont)
        test_mae = test_mae / (len(testloader) - skip_cont)
    
    print("======== Metrics ========")
    print(f'MSE: {test_loss:.6f} | MAE: {test_mae:.6f}')
    
    #! Baseline test
    # Check if the model outputed zero por all pixels
    test_loss = 0.0
    test_mae = 0.0
    # Disable gradient computation and reduce memory consumption.
    for inputs, labels in tqdm(testloader):
        # y_pred = model(inputs.to(device))
        if torch.all(labels == -1):
            skip_cont += 1
            continue
        y_pred = torch.zeros_like(labels)
        
        loss = mse(y_pred, labels)
        _mae = mae(y_pred, labels)
            
        test_loss += loss.detach()
        test_mae += _mae.detach()

    test_loss = test_loss / (len(testloader) - skip_cont)
    test_mae = test_mae / (len(testloader) - skip_cont)
    
    print("======== Zero Pred Baseline Metrics ========")
    print(f'MSE: {test_loss:.6f} | MAE: {test_mae:.6f}')