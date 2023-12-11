import numpy as np
import torch


class Recorder:
    def __init__(self, verbose=False, delta=0, early_stop_time=10):
        self.verbose = verbose
        self.best_score = None
        self.val_loss_min = np.Inf
        self.delta = delta
        self.decrease_time = 0
        self.early_stop_time = early_stop_time

    def __call__(self, val_loss, model, path, early_stop=False):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score >= self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.decrease_time = 0
            print(f'\nEarly Stopping counter: {self.decrease_time}')
        else:
            self.decrease_time += 1
            print(f'\nEarly Stopping counter: {self.decrease_time}')
        return self.decrease_time >= self.early_stop_time if early_stop else False
        # return True if early_stop else False

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        print(path+'/'+'checkpoint.pth')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss
