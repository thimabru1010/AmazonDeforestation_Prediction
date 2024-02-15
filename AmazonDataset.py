import torch
import torchvision
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
from pathlib import Path
from preprocess import load_tif_image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, root_dir: Path, normalize: bool=False, transform: torchvision.transforms=None,
                 Debug: bool=False):
        super(CustomDataset, self).__init__()
        self.root_dir = root_dir

        self.data_files = os.listdir(root_dir / 'Input')
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
        # data = torch.tensor(np.load(self.root_dir / 'Input' / self.data_files[index])).unsqueeze(1).float()
        # labels = torch.tensor(np.load(self.root_dir / 'Labels' / self.data_files[index])).unsqueeze(0).float()
        data = np.load(self.root_dir / 'Input' / self.data_files[index]).astype(np.float32)
        labels = np.load(self.root_dir / 'Labels' / self.data_files[index]).astype(np.float32)
        
        if self.transform:
            transformed = self.transform(image=data.transpose(1, 2, 0), mask=labels)
            data = transformed['image'].unsqueeze(1).float()
            labels = transformed['mask'].unsqueeze(0).float()
            # data = self.transform(data)
            # labels = self.transform(labels)
        else:
            data = torch.tensor(data).unsqueeze(1).float()
            labels = torch.tensor(labels).unsqueeze(0).float()
            
        # if self.normalize:
        # data = data / 3
        
        # print(labels.shape)
        # data[data == 0] = -1
        # labels[labels == 50] = 0
        # print('DEBUG AmazonDataset')
        # print(data.shape)
        # data = F.one_hot(data.long(), num_classes=2).view(data.shape[0], -1, data.shape[2], data.shape[3]).float()
        # print(data.shape)
        # 1/0
        return data, labels
#! ------------------------------------------------------------------------------------------------------------------------    
    
class CustomDataset_Test(Dataset):
    def __init__(self, img_path: Path, normalize: bool=False, transform: torchvision.transforms=None,
                Debug: bool=False, patch_size: int=64, val_fill: int=4):
        super(CustomDataset_Test, self).__init__()
        # self.root_dir = 
        self.patch_size = patch_size
        # Remember number of trimesters of test set to complete validation set
        print(img_path)
        img = load_tif_image(img_path)
        # img = img.reshape((3, -1, img.shape[1], img.shape[2])).max(axis=0)[val_fill:]
        img = img.reshape((3, -1, img.shape[1], img.shape[2])).sum(axis=0)[val_fill:]
        self.original_shape = img.shape
        print(img.shape)
        
        print('Img Test Class Percentages')
        def_count = np.sum(img == 1)
        no_def_count = np.sum(img == 0)
        total_classes = def_count + no_def_count
        print(f'Deforestation: {def_count/total_classes} ')
        print(f'No Deforestation: {no_def_count/total_classes} ')

        expanded_img = np.ones((img.shape[0], (img.shape[1] // 64 + 1) * 64, (img.shape[2] // 64 + 1) * 64)) * -1
        expanded_img[:, :img.shape[1], :img.shape[2]] = img
        print(expanded_img.shape)
        self.expanded_shape = expanded_img.shape
        
        self.data_files = self.extract_sorted_patches(expanded_img, patch_size)
        self.patches_original_shape = self.data_files.shape
        print(self.data_files.shape)
        self.data_files = self.data_files.reshape(-1, self.data_files.shape[2], self.data_files.shape[3], self.data_files.shape[4])

        if Debug:
            self.data_files = self.data_files[:20]
            
        self.mean = None
        self.std = None
        self.normalize = normalize
        
        self.transform = transform

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        # print(self.root_dir / self.data_files[index])
        # patch_window = np.load(self.root_dir / self.data_files[index])
        patch_window = self.data_files[index]
        # print(patch_window.shape)
        
        data = torch.tensor(patch_window[:-1]).unsqueeze(1).float()
        labels = torch.tensor(patch_window[-1]).unsqueeze(0).float()
        
        # data[data == 0] = -1
        labels[labels == 50] = 0
        print(data.shape, labels.shape)
        1/0
        
        # Apply min-max normalization
        # if self.normalize:
        #     data = data / 2
        # data = F.one_hot(data.long(), num_classes=2).view(data.shape[0], -1, data.shape[2], data.shape[3]).float()
        return data, labels
    
    def extract_sorted_patches(self, img, patch_size, window_size=5):
        '''Extract sorted patches windowed with no overlap, i.g, stride equal to patch_size'''
        stride = patch_size

        temp, height, width = img.shape

        num_patches_h = int(height / stride)
        num_patches_w = int(width / stride)
    
        patches = []
        total_iterations = (num_patches_h*num_patches_w) * (temp - window_size + 1)
        with tqdm(total=total_iterations, desc='Test:Extracting Patches') as pbar:
            for h in range(num_patches_h):
                for w in range(num_patches_w):
                    patch = img[:, h*stride:(h+1)*stride, w*stride:(w+1)*stride]
                    windowed_patches = []
                    for j in range(patch.shape[0] - window_size + 1):
                        windowed_patch = patch[j:j+window_size]
                        windowed_patches.append(windowed_patch)
                        pbar.update(1)
                    patches.append(np.stack(windowed_patches, axis=0))
                    
        return np.stack(patches, axis=0)
    
    def patches_reconstruction(self, patches):
        '''Reconstruct an image from a set of windowed patches systematically extracted and no overlap'''
        stride = self.patch_size

        temp, height, width = self.expanded_shape
        
        num_patches_h = self.expanded_shape[1] // stride
        num_patches_w = self.expanded_shape[2] // stride

        print(num_patches_h * num_patches_w, patches.shape[0])
        assert(num_patches_h * num_patches_w) == patches.shape[0], "Number of patches didn't match"
        
        img_reconstructed = np.zeros((patches.shape[1], height, width))
        cont = 0
        for h in range(num_patches_h):
            for w in range(num_patches_w):
                for j in range(patches.shape[1]):
                    img_reconstructed[j, h*stride:(h+1)*stride, w*stride:(w+1)*stride] = patches[cont, j, -1]
                cont += 1
        print('Reconstruction Done!')
        # Cut expanded image in original shape
        return img_reconstructed[:, :self.original_shape[1], :self.original_shape[2]]
#! ------------------------------------------------------------------------------------------------------------------------

class IbamaInpe25km_Dataset(Dataset):
    def __init__(self, root_dir: Path, normalize: bool=True, transform: torchvision.transforms=None,
                Debug: bool=False, mode: str='train', val_data=None, mean=None, std=None):
        super(IbamaInpe25km_Dataset, self).__init__()
        self.root_dir = root_dir
        if mode == 'train':
            # self.ibama_files = os.listdir(root_dir / 'Geotiff-IBAMA_resampled')
            self.inpe_files = os.listdir(root_dir / 'Geotiff-INPE/tiffs')
            # print(len(self.ibama_files), len(self.inpe_files))
            print(len(self.inpe_files))
            # Filter string to select files with 'ArCS' in the name
            self.arcs_files = list(filter(lambda x: 'ArCS' in x, self.inpe_files))
            self.arcs_files.remove('ArCS.tif')
            self.dear_files = list(filter(lambda x: 'DeAr' in x, self.inpe_files))
            self.dear_files.remove('DeAr.tif')
            print(len(self.arcs_files), len(self.dear_files))
            self.data_files = self.arcs_files
            # TODO create temporal windows sliding over the months
            self.data_files.sort()
            # print(self.data_files)
            func = lambda x: x.split('.tif')[0][4:]
            date_files = [func(file) for file in self.data_files]
            df_date = pd.DataFrame(date_files, columns=['date_str']) #0101
            df_date['date'] = pd.to_datetime(df_date['date_str'], format="%d%m%y")
            df_date = df_date.sort_values(by='date')
            print(df_date.head(10))
            df_date_train = df_date[df_date['date'] < '2019-01-01']
            df_date_val = df_date[(df_date['date'] >= '2019-01-01') & (df_date['date'] < '2020-01-01')]
            df_date_test = df_date[df_date['date'] >= '2020-01-01']
            
            self.data_files = self.sliding_window(df_date_train.date, 3)
            
            self.val_files = self.sliding_window(df_date_val.date, 3)
            
            self.test_files = self.sliding_window(df_date_test.date, 3)
            
            # self.data_files, self.val_files = train_test_split(self.data_files, test_size=0.2, random_state=42)
            self.mean, self.std = self._get_mean_std()
            
            # print(len(self.data_files))
            # 1/0
        else:
            self.data_files = val_data
            self.mean = mean
            self.std = std
            self.val_files = self.data_files
        
        if Debug:
            self.data_files = self.data_files[:20]
            
        self.normalize = normalize
        self.transform = transform

    def get_validation_set(self):
        return self.val_files
    
    def get_test_set(self):
        return self.test_files
    
    def _get_mean_std(self):
        '''Get mean and std of the dataset'''
        # print(self.data_files)
        df = pd.concat(self.data_files).to_frame(name='date')
        df.drop_duplicates(inplace=True)
        df['date_str'] = df['date'].dt.strftime('%d%m%y')
        for i, file in enumerate(df.date_str):
            img = load_tif_image(self.root_dir / 'Geotiff-INPE/tiffs' / ('ArCS' + file + '.tif'))
            if i == 0:
                data = np.expand_dims(img, axis=0)
            else:
                data = np.concatenate((data, np.expand_dims(img, axis=0)), axis=0)
        # Calculates the mean and std only for amazon pixels (Excluding the background)
        data[data < -1e38] = -1
        data = data[data != -1]
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        print('Mean:', mean, 'Std:', std)
        return mean, std
    
    def sliding_window(self, input_list, window_size):
        result = []
        for i in range(len(input_list) - window_size + 1):
            window = input_list[i:i + window_size]
            result.append(window)
        return result
    
    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        # print(self.root_dir / self.data_files[index])
        patch_window = self.data_files[index]
        # print(patch_window)
        df = patch_window.to_frame(name='date')
        df['date_str'] = df['date'].dt.strftime('%d%m%y')
        # print(patch_window.shape)
        filenames = df['date_str'].to_list()
        for i, file in enumerate(filenames[:-1]):
            # print(file)
            img = load_tif_image(self.root_dir / 'Geotiff-INPE/tiffs' / ('ArCS' + file + '.tif'))
            if i == 0:
                data = np.expand_dims(img, axis=0)
            else:
                data = np.concatenate((data, np.expand_dims(img, axis=0)), axis=0)
        
        data[data < -1e38] = 0
        # data_cp = np.zeros((2, 100, data.shape[2]))
        # data_cp[:, :data.shape[1], :] = data
        
        # data = torch.tensor(data).float()

        labels = load_tif_image(self.root_dir / 'Geotiff-INPE/tiffs' / ('ArCS' + filenames[-1] + '.tif'))
        labels[labels < -1e38] = 0
        
        # print(data.shape, labels.shape)
        # print(self.mean.shape, self.std.shape)
        
        if self.normalize:
            data = data - self.mean / self.std
            
        if self.transform:
            # For albumentations the image needs to be in shape (H, W, C)
            transformed = self.transform(image=data.reshape(data.shape[1], data.shape[2], data.shape[0]), mask=labels)
            data = transformed['image']
            labels = transformed['mask']
        else:
            data = torch.tensor(data)
            labels = torch.tensor(labels)
            
        return data.unsqueeze(1).float(), labels.unsqueeze(0).unsqueeze(0).float()