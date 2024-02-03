import torch
from tqdm import tqdm
from openstl.models.simvp_model import SimVP_Model
from segmentation_models_pytorch.losses import FocalLoss
import torch.optim as optm

class BaseExperiment():
    def __init__(self, args):
        self.epochs = 0
        self.device = "cuda:0"
        in_shape = (4, 12, 64, 64)
        self.model = SimVP_Model(in_shape=in_shape, nclasses=2).to(self.device)
        self.trainloader = args.trainloader
        
        self.optm = optm.Adam(self.model.parameters(), lr=1e-4)
        self.loss = FocalLoss("binary", gamma=3).to(self.device)
    
    def train_one_epoch(self):
        self.epoch += 1
        print(f"\nEpoch {self.epoch}")
        for inputs, labels in tqdm(self.trainloader):
            y_pred = self.model(inputs.to(self.device))
            # Get only the first temporal channel
            y_pred = y_pred[:, 0].contiguous().unsqueeze(1)
            # y_pred = torch.argmax(y_pred, axis=2)
            # print(y_pred.shape)
            loss = self.loss(y_pred=y_pred, y_true=labels.to(self.device))
            
            self.optm.zero_grad()
            self.loss.backward()
            self.optm.step()
            
            train_loss += loss.detach()
        train_loss = train_loss / len(self.trainloader)
        
        return train_loss

    def validate_one_epoch(self):
        pass
    
    def train(self, n_epochs):
        for epoch in range(self.epochs):
            
            # train for 1 epoch and compute error
            train_loss = self.train_one_epoch()

            # compute validation error and save history            
            val_err = self.validate_one_epoch()
            
            # model.errs.append([train_err, val_err])

            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f} | Validation Loss = {val_err:.6f}")
    
        