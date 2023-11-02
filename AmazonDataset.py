import torch
import torchvision
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import numpy as np
from pathlib import Path

class CustomDataset(Dataset):
    def read_sorted_paths(self, root_dir, num_windows=7):
        filenames = []
        for i in range(len(os.listdir(root_dir)) // num_windows):
            for j in range(num_windows):
                filename = Path(f'patch={i}_trimester_window={j}.npy')
                filenames.append(filename)
        return filenames
    
    def __init__(self, root_dir: Path, normalize: bool=False, transform: torchvision.transforms=None,
                 Debug: bool=False, Test: bool=False):
        super(CustomDataset, self).__init__()
        self.root_dir = root_dir
        if Test:
            self.data_files = self.read_sorted_paths(self.root_dir)
        else:
            self.data_files = os.listdir(root_dir)
        if Debug:
            self.data_files = self.data_files[:20]
        self.mean = None
        self.std = None
        self.normalize = normalize
        
        self.transform = transform

        if normalize:
            # # get the mean/std values along the channel dimension
            # mean = data.mean(axis=(0, 1, 2, 3)).reshape(1, 1, -1, 1, 1)
            # std = data.std(axis=(0, 1, 2, 3)).reshape(1, 1, -1, 1, 1)
            # data = (data - mean) / std
            # self.mean = mean
            # self.std = std
            pass

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        # print(self.root_dir / self.data_files[index])
        patch_window = np.load(self.root_dir / self.data_files[index])
        
        data = torch.tensor(patch_window[:-1]).float().transpose(1, 0)
        labels = torch.tensor(patch_window[-1]).float().unsqueeze(1).transpose(1, 0)
        
        # Apply min-max normalization
        #! Since Data is categorical 0, 1 and 2 min-max normalization in simply dividing by 2.
        #! x_scaled = (x - xmin) / (xmax - min). xmax=2, xmin=0
        # if self.normalize:
        #     data = data / 2
        
        if self.transform:
            data = self.transform(data)
            labels = self.transform(labels)
            
        # return F.one_hot(data, num_classes=3), F.one_hot(labels, num_classes=3)
        return data, labels