import torch
from tqdm import tqdm
from openstl.models.simvp_model import SimVP_Model
from segmentation_models_pytorch.losses import FocalLoss
import torch.nn as nn
import torch.optim as optm
import os
import json
from metrics_amazon import CM, f1_score0, f1_score1, Recall, Precision, ACC
import numpy as np
import torch.nn.functional as F
from torchsummary import summary
from preprocess import load_tif_image, preprocess_patches, divide_pred_windows, reconstruct_sorted_patches, reconstruct_time_patches
from torch.optim.lr_scheduler import StepLR
from CustomLosses import WMSELoss, WMAELoss

class BaseExperiment():
    def __init__(self, trainloader, valloader, custom_model_config, training_config, seed=42):
        # TODO: wrap into a function to create work dir
        # Create work dir
        self.work_dir_path = os.path.join('work_dirs', training_config['ex_name'])
        if not os.path.exists(self.work_dir_path):
            os.makedirs(self.work_dir_path)
            
        self._save_json(custom_model_config, 'model_config.json')
        self._save_json(training_config, 'model_training.json')
            
        self.epochs = training_config['epoch']
        self.patience = training_config['patience']
        self.delta = training_config['delta']
        self.device = "cuda:0"
        in_shape = training_config['in_shape']
        
        print('Input shape:', in_shape)
        torch.manual_seed(seed)
        # self.model = TimeSformer(img_size=in_shape[2], num_classes=1, num_frames=in_shape[0], attention_type='divided_space_time')
        
        self.model = self._build_model(in_shape, custom_model_config['num_classes'], custom_model_config, training_config['batch_size'])
        
        self.aux_metrics = {}
        if custom_model_config['num_classes']:
            self.classification = True
            if training_config['loss'] == 'focal':
                # self.loss = FocalLoss(mode='multiclass', alpha=training_config['focal_alpha'],\
                #     gamma=training_config['focal_gamma'], ignore_index=-1)
                self.loss = FocalLoss(mode='binary', alpha=training_config['focal_alpha'],\
                    gamma=training_config['focal_gamma'], ignore_index=-1)
            elif training_config['loss'] == 'ce':
                class_weights = torch.tensor([1, 1], dtype=torch.float32).to(self.device)
                self.loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
            for metric in training_config['aux_metrics']:
                if metric == 'f1_score0':
                    self.aux_metrics['f1_score0'] = f1_score0
                elif metric == 'f1_score1':
                    self.aux_metrics['f1_score1'] = f1_score1
                elif metric == 'Recall':
                    self.aux_metrics['Recall'] = Recall
                elif metric == 'Precision':
                    self.aux_metrics['Precision'] = Precision
                elif metric == 'CM':
                    self.aux_metrics['CM'] = CM
                # elif metric == 'MSE':
                #     self.aux_metrics['MSE'] = nn.MSELoss()
        else:
            self.classification = False
            self.loss = WMSELoss(weight=1)
            # TODO: add auxiliary metrics
            self.mae = WMAELoss(weight=1)
        
        # print(summary(self.model, tuple(in_shape)))
        print(self.model)
        
        if training_config['optmizer'] == 'adam':
            self.optm = optm.Adam(self.model.parameters(), lr=training_config['lr'])
        elif training_config['optmizer'] == 'sgd':
            self.optm = optm.SGD(self.model.parameters(), lr=training_config['lr'], momentum=training_config['sgd_momentum'])
        
        self.scheduler = StepLR(self.optm, step_size=training_config['scheduler_step_size'],\
            gamma=training_config['scheduler_decay_factor'])
        
        # if training_config['amazon_mask'] and training_config['pixel_size'] == '25K':
        #     mask = load_tif_image('data/IBAMA_INPE/25K/INPE/tiff/mask.tif')
            
        # self.loss = WMSELoss(weight=1)
        # self.mae = WMAELoss(weight=1)
        self.mse = nn.MSELoss()
        
        self.trainloader = trainloader
        self.valloader = valloader
        
    def _build_model(self, in_shape, num_classes, custom_model_config, batch_size):
        return SimVP_Model(in_shape=in_shape, nclasses=num_classes, batch_size=batch_size, **custom_model_config).to(self.device)
    
    def _save_json(self, data, filename):
        with open(os.path.join(self.work_dir_path, filename), 'w') as f:
            json.dump(data, f)
            
    def train_one_epoch(self):
        train_loss = 0
        train_area_reg = 0
        self.model.train(True)
        for inputs, labels, def_area in tqdm(self.trainloader):
            # Zero your gradients for every batch!
            self.optm.zero_grad()
            
            y_pred, mid_pred = self.model(inputs.to(self.device))
            # Get only the first temporal channel
            y_pred = y_pred[:, :2].contiguous()#.unsqueeze(1)
            y_pred = torch.transpose(y_pred, 1, 2)
            labels = labels.type(torch.LongTensor)
            
            print(mid_pred.shape, def_area.shape)
            
            area_reg = self.mse(mid_pred, def_area.to(self.device))
            # area_reg.backward()
            # print(y_pred.shape, labels.shape, labels.squeeze(2).shape)
            # print(y_pred.dtype, labels.squeeze(2).dtype)
            loss = self.loss(y_pred, labels.squeeze(2).to(self.device))
            total_loss = loss + area_reg
            total_loss.backward()
            # loss.backward()
            
            # Adjust learning weights
            self.optm.step()
            
            train_loss += loss.detach()
            train_area_reg += area_reg.detach()
            
        train_loss = train_loss / len(self.trainloader)
        train_area_reg = train_area_reg / len(self.trainloader)
        
        return train_loss, train_area_reg

    def validate_one_epoch(self):
        val_loss = 0
        val_area_reg = 0
        val_aux_metrics = dict.fromkeys(self.aux_metrics.keys(), 0)
        self.model.eval()
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for inputs, labels, def_area in tqdm(self.valloader):
                y_pred, mid_pred = self.model(inputs.to(self.device))
                # Get only the first temporal channel
                y_pred = y_pred[:, :2].contiguous()#.unsqueeze(1)
                # Change B, T, C to B, C, T
                y_pred = torch.transpose(y_pred, 1, 2)
                labels = labels.type(torch.LongTensor)
                
                area_reg = self.mse(mid_pred, def_area.to(self.device))
                
                loss = self.loss(y_pred, labels.squeeze(2).to(self.device))
                # mae = self.mae(y_pred, labels.to(self.device))
                # y_pred = F.softmax(y_pred, dim=1)
                # y_pred = torch.argmax(y_pred, dim=1)
                y_pred = F.sigmoid(y_pred)
                y_pred[y_pred >= 0.5] = 1
                y_pred[y_pred < 0.5] = 0
                y_pred = y_pred.squeeze(1)
                labels = labels.squeeze(2)
                # print(y_pred.shape, labels.shape)
                y_pred = y_pred[labels != -1].cpu().numpy()
                labels = labels[labels != -1].cpu().numpy()
                for metric_name in self.aux_metrics.keys():
                    
                    val_aux_metrics[metric_name] += self.aux_metrics[metric_name](y_pred, labels)
                
                val_loss += loss.detach()
                val_area_reg += area_reg.detach()
            
        val_loss = val_loss / len(self.valloader)
        val_area_reg = val_area_reg / len(self.valloader)
        for metric_name in self.aux_metrics.keys():
            if metric_name != 'CM':
                val_aux_metrics[metric_name] = val_aux_metrics[metric_name] / len(self.valloader)

        return val_loss, val_area_reg, val_aux_metrics
    
    def train(self):
        min_val_loss = float('inf')
        early_stop_counter = 0
        for epoch in range(self.epochs):
            
            train_loss, train_area_reg = self.train_one_epoch()
            
            val_loss, val_area_reg, val_aux_metrics = self.validate_one_epoch()
            
            last_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()
            
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
            terminal_str = f"Epoch {epoch}: LR = {last_lr:.8f} | Train Loss = {train_loss:.6f} | Val Loss = {val_loss:.6f} | Train MSE = {train_area_reg:.6f} | Val MSE = {val_area_reg:.6f}"
            
            cm = val_aux_metrics['CM']
            TP, FP, FN, TN = cm[0, 0], cm[1, 0], cm[0, 1], cm[1, 1]
            prec = TP/(TP+FP)
            rec = TP/(TP+FN)
            _f1_score = 2 * prec * rec / (prec + rec)
            val_aux_metrics['f1_score0'] = _f1_score
            
            prec = TN/(TN+FN)
            rec = TN/(TN+FP)
            _f1_score = 2 * prec * rec / (prec + rec)
            val_aux_metrics['f1_score1'] = _f1_score
            
            for metric_name in val_aux_metrics.keys():
                if metric_name != 'CM':
                    terminal_str += f" | Val {metric_name} = {val_aux_metrics[metric_name]:.6f}"
            print(terminal_str)
            print(val_aux_metrics['CM'])
            # print(f"Epoch {epoch}: Train Loss = {train_loss:.6f} | Validation Loss = {val_loss:.6f}")
            # print(f"Epoch {epoch}: Train Loss = {train_loss:.6f} | Validation Loss = {val_loss:.6f} | Validation MAE = {val_mae:.6f}")

def _build_model(in_shape, nclasses, custom_model_config, batch_size, device):
    return SimVP_Model(in_shape=in_shape, nclasses=nclasses, batch_size=batch_size, **custom_model_config).to(device)

def test_model(testloader, training_config, custom_model_config):
    work_dir_path = os.path.join('work_dirs', training_config['ex_name'])
    device = "cuda:0"
    in_shape = training_config['in_shape']
    
    model = _build_model(in_shape, custom_model_config['num_classes'], custom_model_config,\
        training_config['batch_size'], device)
    # model = _build_model(in_shape, None, custom_model_config, device)
    model.load_state_dict(torch.load(os.path.join(work_dir_path, 'checkpoint.pth')))
    # model.load_state_dict(os.path.join(work_dir_path, 'checkpoint.pth')).to(device)
    model.eval()
    
    aux_metrics = {}
    if custom_model_config['num_classes']:
        classification = True
        #TODO: add auxiliary metrics
        for metric in training_config['aux_metrics']:
            if metric == 'f1_score0':
                aux_metrics['f1_score0'] = f1_score0
            elif metric == 'f1_score1':
                aux_metrics['f1_score1'] = f1_score1
            elif metric == 'Recall':
                aux_metrics['Recall'] = Recall
            elif metric == 'Precision':
                aux_metrics['Precision'] = Precision
            elif metric == 'CM':
                aux_metrics['CM'] = CM
    else:
        classification = False
        mse = WMSELoss(weight=1)
        # TODO: add auxiliary metrics
        mae = WMAELoss(weight=1)
        test_loss = 0.0
        test_mae = 0.0
        
    # cm = np.zeros((2, 2), dtype=int)
    # Disable gradient computation and reduce memory consumption.
    skip_cont = 0
    preds = []
    preds_def = []
    labels_stack = []
    val_aux_metrics = {metric_name: 0 for metric_name in aux_metrics.keys()}
    with torch.no_grad():
        for inputs, labels in tqdm(testloader):
            y_pred, _ = model(inputs.to(device))
            # Get only the first temporal channel
            y_pred = y_pred[:, :2].contiguous()#.unsqueeze(1)
            # Change B, T, C to B, C, T
            y_pred = torch.transpose(y_pred, 1, 2)
            labels = labels.type(torch.LongTensor)
            
            # if torch.all(labels == -1):
            #     skip_cont += 1
            #     # continue
            
            if classification:    
                # print(y_pred.shape, labels.shape)
                y_pred = y_pred.cpu()
                labels = labels.cpu()
                
                # y_pred = F.softmax(y_pred, dim=1)
                y_pred = F.sigmoid(y_pred)
                # print(y_pred.shape)
                preds.append(y_pred[0].numpy())
                labels_stack.append(labels.numpy())
                # preds_def.append(y_pred[0, 1].numpy())
                
                # y_pred = torch.argmax(y_pred, dim=1).squeeze(1)
                y_pred[y_pred >= 0.5] = 1
                y_pred[y_pred < 0.5] = 0
                #TODO: verificar se vai dar problema com sigmoid
                y_pred = y_pred.squeeze(1)
                labels = labels.squeeze(2)

                # preds.append(y_pred[0].numpy())
                
                # If all labels are outside the mask, don't calculate the error
                # But they are inferred by the model and appended to the list due to reconstruction
                if torch.all(labels == -1):
                    skip_cont += 1
                    continue
                
                y_pred = y_pred[labels != -1].numpy()
                labels = labels[labels != -1].numpy()
                
                for metric_name in aux_metrics.keys():
                    val_aux_metrics[metric_name] += aux_metrics[metric_name](y_pred, labels)
            else:
                preds.append(y_pred.cpu().numpy()[0, 0, 0])  
                if torch.all(labels == -1):
                    skip_cont += 1
                    continue
                loss = mse(y_pred, labels.to(device))
                _mae = mae(y_pred, labels.to(device))
                test_loss += loss.detach()
                test_mae += _mae.detach()
            
                # print(y_pred.cpu().numpy()[0, 0, 0].shape)
                # preds.append(y_pred.cpu().numpy()[0, 0, 0])
            

        preds = np.stack(preds, axis=0)
        print(preds.shape)
        np.save(os.path.join(work_dir_path, 'preds.npy'), preds)
        labels_stack = np.stack(labels_stack, axis=0)
        print(labels_stack.shape)
        np.save(os.path.join(work_dir_path, 'labels_patches.npy'), labels_stack)
        del labels_stack
        # preds_def = np.stack(preds_def, axis=0)
        # print(preds_def.shape)
        print("======== Test Metrics ========")
        if classification:
            for metric_name in aux_metrics.keys():
                if metric_name != 'CM':
                    val_aux_metrics[metric_name] = val_aux_metrics[metric_name] / (len(testloader) - skip_cont)
            cm = val_aux_metrics['CM']
            TP, FP, FN, TN = cm[0, 0], cm[1, 0], cm[0, 1], cm[1, 1]
            prec = TP/(TP+FP)
            rec = TP/(TP+FN)
            _f1_score = 2 * prec * rec / (prec + rec)
            val_aux_metrics['f1_score0'] = _f1_score
            
            prec = TN/(TN+FN)
            rec = TN/(TN+FP)
            _f1_score = 2 * prec * rec / (prec + rec)
            val_aux_metrics['f1_score1'] = _f1_score
            
            terminal_str = f""
            for metric_name in val_aux_metrics.keys():
                if metric_name != 'CM':
                    terminal_str += f"{metric_name} = {val_aux_metrics[metric_name]:.6f} | "
            print(terminal_str)
            print(val_aux_metrics['CM'])
        else:
            test_loss = test_loss / (len(testloader) - skip_cont)
            test_mae = test_mae / (len(testloader) - skip_cont)
            
            print(f'MSE: {test_loss:.6f} | MAE: {test_mae:.6f}')
            # preds = np.stack(preds, axis=0)
            # print(preds.shape)
            test_loss = 0.0
    return preds

    # #! Classification Baseline Metrics
    # skip_cont = 0
    # val_aux_metrics = {metric_name: 0 for metric_name in aux_metrics.keys()}
    # for inputs, labels in tqdm(testloader):
    #     if torch.all(labels == -1):
    #         skip_cont += 1
    #         continue
        
    #     # y_pred = y_pred[labels != -1].numpy()
    #     labels = labels[labels != -1].numpy()
        
    #     for metric_name in aux_metrics.keys():
    #         val_aux_metrics[metric_name] += aux_metrics[metric_name](labels, labels)
    # print('Skip cont:', skip_cont)
    # print("======== Classification Baseline Metrics ========")
    # for metric_name in aux_metrics.keys():
    #     val_aux_metrics[metric_name] = val_aux_metrics[metric_name] / (len(testloader) - skip_cont)
    #     terminal_str = f""
    #     for metric_name in val_aux_metrics.keys():
    #         if metric_name != 'CM':
    #             terminal_str += f"{metric_name} = {val_aux_metrics[metric_name]:.6f} | "
    # print(terminal_str)
    # print(val_aux_metrics['CM'])
    # cm = val_aux_metrics['CM']
    # TP, FP, FN, TN = cm[0, 0], cm[1, 0], cm[0, 1], cm[1, 1]
    # print(FP, FN)
    # prec = TN/(TN+FN)
    # rec = TN/(TN+FP)
    # _f1_score1 = 2 * prec * rec / (prec + rec)
    # print(f'F1 Score Class 1: {_f1_score1:.6f}')