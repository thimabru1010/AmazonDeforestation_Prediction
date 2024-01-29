import numpy as np
from torch.utils.data import Dataset
import torch

class GiovanniDataset(Dataset):
    def __init__(
        self, 
        X, 
        patches, 
        frames_idx, 
        county_data=None, 
        county_defor=None,
        precip_data=None,
        tpi_data=None,
        landcover_data=None,
        scores_data=None,
        night_data=None,
        device=None,
        normalize=True
    ):
        super(GiovanniDataset, self).__init__()

        self.patches = patches
        self.frames_idx = frames_idx
        self.X = X
        self.county_data = county_data
        self.county_defor = county_defor
        self.precip_data = precip_data
        self.tpi_data = tpi_data
        self.landcover_data = landcover_data
        self.scores_data = scores_data
        self.night_data = night_data

        self.autor_window = 4
        self.ix = frames_idx["x"].min()
        self.iy = frames_idx["y"].min()
        
        self.device = device
        
        self.mean = None
        self.std = None
        
        # self.patches = self.patches[:20]
        
        if normalize:
            pass

    def __len__(self):
        return len(self.patches) * (self.X.shape[0]-self.autor_window)

    def __getitem__(self, index):

        # get index info
        idx_patch = index // (self.X.shape[0] - self.autor_window)
        idx_time   = index % (self.X.shape[0] - self.autor_window)
        idx_frames = self.frames_idx.loc[self.patches[idx_patch]]

        # get input
        input_matrix = self.X[
            idx_time:idx_time+self.autor_window, 
            idx_frames["x"].min()-self.ix:idx_frames["x"].max()-self.ix+1, 
            idx_frames["y"].min()-self.iy:idx_frames["y"].max()-self.iy+1
        ]
        
        # def_deter = input_matrix.copy()
        # print('DEBUG GIOVANI DATASET')
        input_matrix[input_matrix > 1e-7] += 0.5
        # input_matrix[input_matrix <= 1e-7] = 0
        
        input_matrix = np.expand_dims(input_matrix, axis=1)
        # print(input_matrix.shape)

        if self.county_data is not None:
            # Repete os valores para cada trimestre (1, 2, 64, 64) --> (4, 2, 64, 64)
            county_data = self.county_data[np.newaxis,
                    :,
                    idx_frames["x"].min()-self.ix:idx_frames["x"].max()-self.ix+1, 
                    idx_frames["y"].min()-self.iy:idx_frames["y"].max()-self.iy+1]
            # county_data = np.expand_dims(county_data, axis=1)
            # print(county_data.shape)
            input_matrix = np.concatenate((input_matrix, np.tile(county_data, (4, 1, 1, 1))), axis=1)
            # print(input_matrix.shape)
        
        if self.county_defor is not None:
            county_defor = self.county_defor[idx_time:idx_time+self.autor_window,
                                             np.newaxis,
                                             idx_frames["x"].min()-self.ix:idx_frames["x"].max()-self.ix+1, 
                                             idx_frames["y"].min()-self.iy:idx_frames["y"].max()-self.iy+1]
            input_matrix = np.concatenate((input_matrix, county_defor), axis=1)

        
        if self.precip_data is not None:
            precip_data = self.precip_data[
                    idx_time:idx_time+self.autor_window,
                    np.newaxis,
                    idx_frames["x"].min()-self.ix:idx_frames["x"].max()-self.ix+1, 
                    idx_frames["y"].min()-self.iy:idx_frames["y"].max()-self.iy+1]
            input_matrix = np.concatenate([input_matrix, precip_data], axis=1)
        
        if self.tpi_data is not None:
            tpi_data = self.tpi_data[
                    np.newaxis,
                    :,
                    idx_frames["x"].min()-self.ix:idx_frames["x"].max()-self.ix+1, 
                    idx_frames["y"].min()-self.iy:idx_frames["y"].max()-self.iy+1]
            # print(tpi_data.shape, np.tile(tpi_data, (4, 1, 1, 1)).shape)
            # 1/0
            input_matrix = np.concatenate([input_matrix, np.tile(tpi_data, (4, 1, 1, 1))], axis=1)
        
        if self.landcover_data is not None:
            input_matrix = np.concatenate([
                input_matrix,
                self.landcover_data[
                    :,
                    idx_frames["x"].min()-self.ix:idx_frames["x"].max()-self.ix+1, 
                    idx_frames["y"].min()-self.iy:idx_frames["y"].max()-self.iy+1
                ]
            ])
        
        if self.scores_data is not None:
            scores_data = self.scores_data[
                    np.newaxis,
                    [idx_time+self.autor_window],
                    idx_frames["x"].min()-self.ix:idx_frames["x"].max()-self.ix+1, 
                    idx_frames["y"].min()-self.iy:idx_frames["y"].max()-self.iy+1]
            input_matrix = np.concatenate([input_matrix, np.tile(scores_data, (4, 1, 1, 1))], axis=1)
        
        if self.night_data is not None:
            night_data = np.tile(self.night_data[
                    np.newaxis,
                    :,
                    idx_time+self.autor_window-1,
                    idx_frames["x"].min()-self.ix:idx_frames["x"].max()-self.ix+1, 
                    idx_frames["y"].min()-self.iy:idx_frames["y"].max()-self.iy+1], (4, 1, 1, 1))
            input_matrix = np.concatenate([input_matrix, night_data], axis=1)
            
        # input_matrix = np.tile(scores_data, (4, 1, 1, 1))
        data = torch.tensor(input_matrix).float() #.to(self.device)

        # get output
        labels = np.zeros(
            (   2,
                idx_frames["x"].max()-idx_frames["x"].min() + 1, 
                idx_frames["y"].max()-idx_frames["y"].min() + 1
            )
        )
        # print(idx_frames["x"].max()-idx_frames["x"].min() + 1, idx_frames["y"].max()-idx_frames["y"].min() + 1)
        # 1/0
        target_idx = np.where(
            self.X[
                idx_time+self.autor_window, 
                idx_frames["x"].min()-self.ix:idx_frames["x"].max()-self.ix+1, 
                idx_frames["y"].min()-self.iy:idx_frames["y"].max()-self.iy+1
            ] > 1e-7
        )
        labels[0, :, :] = 1
        labels[0, :, :][target_idx] = 0
        labels[1, :, :][target_idx] = 1
        labels = torch.tensor(labels).float()# .to(self.device)
        # print('DEBUG')
        # print(labels.shape)
        # print(labels.unsqueeze(0).shape)
        # print(data.shape)
        # 1/0
        # print('DEBUG GIOVANNI DATASET 2')
        # print(data.shape, labels.shape)
        return data, labels[1].unsqueeze(0).unsqueeze(0)