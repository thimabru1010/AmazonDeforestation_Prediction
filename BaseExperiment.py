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

from sklearn.metrics import f1_score as skf1_score

def weighted_mse_loss(y_pred, y_true, weights):
    return torch.mean(weights * (y_pred - y_true) ** 2)

class BaseExperiment():
    def __init__(self, trainloader, valloader, custom_model_config, custom_training_config):
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
        self.model = self._build_model(in_shape, None, custom_model_config)
        
        self.optm = optm.Adam(self.model.parameters(), lr=custom_training_config['lr'])
        
        # TODO: try to weight MSE loss
        self.loss = nn.MSELoss()
        
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
        f1_clss0 = 0
        f1_clss1 = 0
        skf1 = 0
        cm = np.zeros((2, 2), dtype=int)
        self.model.eval()
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for inputs, labels in tqdm(self.valloader):
                y_pred = self.model(inputs.to(self.device))
                # Get only the first temporal channel
                y_pred = y_pred[:, 0].contiguous().unsqueeze(1)
                loss = self.loss(y_pred, labels.to(self.device))
                
                #TODO: compute other classification metrics
    
                # _skf1 = skf1_score(_y_pred.reshape(-1), _labels.reshape(-1))
                # _f1_clss0, _f1_clss1 = f1_score(_y_pred, _labels)
                # _cm, _, _, _, _ = confusion_matrix(_labels, _y_pred)
                
                val_loss += loss.detach()
                # f1_clss0 += _f1_clss0.item()
                # f1_clss1 += _f1_clss1.item()
                # skf1 += _skf1.item()
                # cm[0, 0] += _cm[0, 0]
                # cm[0, 1] += _cm[0, 1]
                # cm[1, 0] += _cm[1, 0]
                # cm[1, 1] += _cm[1, 1]
            
        val_loss = val_loss / len(self.valloader)
        # f1_clss0 = f1_clss0 / len(self.valloader)
        # f1_clss1 = f1_clss1 / len(self.valloader)
        # skf1 = skf1 / len(self.valloader)
        
        # print("====== Confusion Matrix ======")
        # print(cm)
        # print(skf1)
        # print(f'F1 Score No def: {f1_clss0:.4f} - F1 Score Def: {f1_clss1:.4f}')
        
        return val_loss
    
    def train(self):
        min_val_loss = float('inf')
        early_stop_counter = 0
        for epoch in range(self.epochs):
            
            train_loss = self.train_one_epoch()
            
            val_loss = self.validate_one_epoch()
            
            if val_loss < min_val_loss + self.delta:
                min_val_loss = val_loss
                early_stop_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.work_dir_path, 'checkpoint.pth'))
            else:
                early_stop_counter += 1
                print(f"Val loss didn't improve! Early Stopping counter: {early_stop_counter}")
            
            if early_stop_counter >= self.patience:
                print(f'Early Stopping! Early Stopping counter: {early_stop_counter}')
                break
            
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f} | Validation Loss = {val_loss:.6f}")

def _build_model(in_shape, nclasses, custom_model_config):
    return SimVP_Model(in_shape=in_shape, nclasses=nclasses, **custom_model_config).to(self.device)
        
def test_model(testloader, custom_training_config, custom_model_config):
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