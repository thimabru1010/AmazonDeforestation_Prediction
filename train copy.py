
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, X, Y, normalize=False):
        super(CustomDataset, self).__init__()
        # Pensar em normalizar sem usar mean e std
        self.X = X
        # self.Y = Y
        self.mean = None
        self.std = None

        if normalize:
            # get the mean/std values along the channel dimension
            mean = X.mean(axis=(0, 1, 2, 3)).reshape(1, 1, -1, 1, 1)
            std = X.std(axis=(0, 1, 2, 3)).reshape(1, 1, -1, 1, 1)
            data = (data - mean) / std
            self.mean = mean
            self.std = std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index]).float()
        # labels = data[]
        # labels = torch.tensor(self.Y[index]).float()
        return data
    

# batch_size = 1

# train_set = CustomDataset(X=X_train, Y=Y_train)
# val_set = CustomDataset(X=X_val, Y=Y_val)
# test_set = CustomDataset(X=X_test, Y=Y_test)

# dataloader_train = torch.utils.data.DataLoader(
#     train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
# dataloader_val = torch.utils.data.DataLoader(
#     val_set, batch_size=batch_size, shuffle=True, pin_memory=True)
# dataloader_test = torch.utils.data.DataLoader(
#     test_set, batch_size=batch_size, shuffle=True, pin_memory=True)