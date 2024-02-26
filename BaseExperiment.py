import torch
from tqdm import tqdm
from openstl.models.simvp_model import SimVP_Model
from segmentation_models_pytorch.losses import FocalLoss
import torch.nn as nn
import torch.optim as optm
import os
from metrics import confusion_matrix, f1_score
import numpy as np
import torch.nn.functional as F
from torchsummary import summary
from preprocess import load_tif_image

from sklearn.metrics import f1_score as skf1_score
from weighted_mse_loss import WMSELoss

class BaseExperiment():
    def __init__(self, trainloader, valloader, custom_model_config, custom_training_config, seed=42):
        # TODO: wrap into a function to create work dir
        # Create work dir
        self.work_dir_path = os.path.join('work_dirs', custom_training_config['ex_name'])
        if not os.path.exists(self.work_dir_path):
            os.makedirs(self.work_dir_path)
            
        self.epochs = custom_training_config['epoch']
        self.patience = custom_training_config['patience']
        self.delta = custom_training_config['delta']
        self.device = "cuda:0"
        in_shape = custom_training_config['in_shape']
        torch.manual_seed(seed)
        self.model = self._build_model(in_shape, None, custom_model_config)
        
        print(summary(self.model, tuple(in_shape)))
        
        self.optm = optm.Adam(self.model.parameters(), lr=custom_training_config['lr'])
        
        if custom_training_config['amazon_mask']:
            mask = load_tif_image('data/IBAMA_INPE/25K/INPE/tiff/mask.tif')
            self.loss = WMSELoss(weight=0.8, mask=mask)
        else:
            self.loss = nn.MSELoss()
            mask = None
            
        self.mae = nn.L1Loss()
        
        self.trainloader = trainloader
        self.valloader = valloader
        
    def _build_model(self, in_shape, nclasses, custom_model_config):
        return SimVP_Model(in_shape=in_shape, nclasses=nclasses, **custom_model_config).to(self.device)
    
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
        
def old_test_model(testloader, custom_training_config, custom_model_config):
    work_dir_path = os.path.join('work_dirs', custom_training_config['ex_name'])
    device = "cuda:0"
    in_shape = custom_training_config['in_shape']
        
    model = _build_model(in_shape, None, custom_model_config)
    model.load_state_dict(os.path.join(work_dir_path, 'checkpoint.pth')).to(device)
    model.eval()
    
    cm = np.zeros((2, 2), dtype=int)
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for inputs, labels in tqdm(testloader):
            y_pred = model(inputs.to(device))
            # Get only the first temporal channel
            y_pred = y_pred[:, 0].contiguous().unsqueeze(1)
            
            #TODO: compute other classification metrics
            _y_pred = torch.argmax(F.softmax(y_pred, dim=2), dim=2).cpu().numpy()[:, 0]
            # _labels = torch.argmax(labels).detach().numpy()
            _labels = labels.detach().numpy()[:, 0, 1]
            # print(_labels.shape)
            # print(_y_pred.shape)
            # 1/0
            _f1_clss0, _f1_clss1 = f1_score(_y_pred, _labels)
            _cm, _, _, _, _ = confusion_matrix(_labels, _y_pred)
            
            f1_clss0 += _f1_clss0.item()
            f1_clss1 += _f1_clss1.item()
            cm[0, 0] += _cm[0, 0]
            cm[0, 1] += _cm[0, 1]
            cm[1, 0] += _cm[1, 0]
            cm[1, 1] += _cm[1, 1]
        
    f1_clss0 = f1_clss0 / len(testloader)
    f1_clss1 = f1_clss1 / len(testloader)
    
    print("====== Confusion Matrix ======")
    print(cm)
    print(f'F1 Score No def: {f1_clss0:.4f} - F1 Score Def: {f1_clss1:.4f}')
    

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
    
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    
    test_loss = 0.0
    test_mae = 0.0
    # cm = np.zeros((2, 2), dtype=int)
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for inputs, labels in tqdm(testloader):
            y_pred = model(inputs.to(device))
            # Get only the first temporal channel
            y_pred = y_pred[:, 0].contiguous().unsqueeze(1)
            
            loss = mse(y_pred, labels.to(device))
            _mae = mae(y_pred, labels.to(device))
                
            test_loss += loss.detach()
            test_mae += _mae.detach()

        test_loss = test_loss / len(testloader)
        test_mae = test_mae / len(testloader)
    
    print("======== Metrics ========")
    print(f'MSE: {test_loss:.6f} | MAE: {test_mae:.6f}')
    
    #! Baseline test
    # Check if the model outputed zero por all pixels
    test_loss = 0.0
    test_mae = 0.0
    # cm = np.zeros((2, 2), dtype=int)
    # Disable gradient computation and reduce memory consumption.
    for inputs, labels in tqdm(testloader):
        # y_pred = model(inputs.to(device))
        y_pred = torch.zeros_like(labels)
        # Get only the first temporal channel
        # y_pred = y_pred[:, 0].contiguous().unsqueeze(1)
        
        loss = mse(y_pred, labels)
        _mae = mae(y_pred, labels)
            
        test_loss += loss.detach()
        test_mae += _mae.detach()

        test_loss = test_loss / len(testloader)
        test_mae = test_mae / len(testloader)
    
    print("======== Zero Pred Baseline Metrics ========")
    print(f'MSE: {test_loss:.6f} | MAE: {test_mae:.6f}')