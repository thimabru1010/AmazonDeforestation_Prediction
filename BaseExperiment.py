import torch
from tqdm import tqdm
from openstl.models.simvp_model import SimVP_Model
from segmentation_models_pytorch.losses import FocalLoss
import torch.nn as nn
import torch.optim as optm
import os
# from metrics import 

class BaseExperiment():
    def __init__(self, trainloader, valloader, custom_model_config, custom_training_config):
        self.epochs = custom_training_config['epoch']
        self.patience = custom_training_config['patience']
        self.device = "cuda:0"
        in_shape = (4, 1, 64, 64)
        self.model = self._build_model(in_shape, 2, custom_model_config)
        # self.model = SimVP_Model(in_shape=in_shape, nclasses=2).to(self.device)
        
        self.optm = optm.Adam(self.model.parameters(), lr=custom_training_config['lr'])
        
        # self.loss = nn.CrossEntropyLoss(weight=torch.Tensor(oss_weights).to(device), ignore_index=50)
        # self.loss = nn.CrossEntropyLoss()
        self.loss = FocalLoss("binary", gamma=3).to(self.device)
        
        self.trainloader = trainloader
        self.valloader = valloader
        
        # Create work dir
        self.work_dir_path = os.path.join('work_dirs', custom_training_config['ex_name'])
        if not os.path.exists(self.work_dir_path):
            os.makedirs(self.work_dir_path)
        
    def _build_model(self, in_shape, nclasses, custom_model_config):
        return SimVP_Model(in_shape=in_shape, nclasses=nclasses, **custom_model_config).to(self.device)
    
    def train_one_epoch(self):
        train_loss = 0
        for inputs, labels in tqdm(self.trainloader):
            # print(inputs.shape)
            y_pred = self.model(inputs.to(self.device))
            # Get only the first temporal channel
            y_pred = y_pred[:, 0].contiguous().unsqueeze(1)
            
            loss = self.loss(y_pred=y_pred, y_true=labels.to(self.device))
            
            self.optm.zero_grad()
            loss.backward()
            self.optm.step()
            
            train_loss += loss.detach()
        train_loss = train_loss / len(self.trainloader)
        
        return train_loss

    def validate_one_epoch(self):
        val_loss = 0
        for inputs, labels in tqdm(self.valloader):
            y_pred = self.model(inputs.to(self.device))
            # Get only the first temporal channel
            y_pred = y_pred[:, 0].contiguous().unsqueeze(1)
            loss = self.loss(y_pred=y_pred, y_true=labels.to(self.device))
            
            #TODO: compute other classification metrics
            
            val_loss += loss.detach()
        val_loss = val_loss / len(self.valloader)
        
        return val_loss
    
    def train(self):
        min_val_loss = float('inf')
        early_stop_counter = 0
        for epoch in range(self.epochs):
            
            train_loss = self.train_one_epoch()
          
            val_loss = self.validate_one_epoch()
            
            if val_loss < min_val_loss:
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
    
    def test(self, testloader):
        for inputs, labels in tqdm(testloader):
            pass