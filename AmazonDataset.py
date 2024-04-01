import torch
import torchvision
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
from pathlib import Path
from preprocess import load_npy_image, load_tif_image, preprocess_patches, divide_pred_windows, extract_sorted_patches, extract_temporal_sorted_patches
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from datetime import datetime
import cv2
from skimage.measure import block_reduce

try:
    from osgeo import gdal
    gdal.PushErrorHandler('CPLQuietErrorHandler')
except ImportError:
    print("osgeo module is not installed. Please install it with pip install GDAL")

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
                Debug: bool=False, mode: str='train', val_data=None, means=None, stds=None):
        super(IbamaInpe25km_Dataset, self).__init__()
        self.root_dir = root_dir
        inpe_folder_path = root_dir / 'INPE/tiff'
        self.inpe_folder_path = inpe_folder_path
        if mode == 'train':
            # self.ibama_files = os.listdir(root_dir / 'Geotiff-IBAMA_resampled')
            # inpe_folder_path = root_dir / 'INPE/tiff'
            # self.inpe_folder_path = inpe_folder_path
            self.inpe_files = os.listdir(inpe_folder_path)
            # print(len(self.ibama_files), len(self.inpe_files))
            print(len(self.inpe_files))
            # Filter string to select files with 'ArCS' in the name
            self.arcs_files = list(filter(lambda x: 'ArCS' in x, self.inpe_files))
            self.arcs_files.remove('ArCS.tif')
            self.clouds_files = list(filter(lambda x: 'nv' in x, self.inpe_files))
            self.clouds_files.remove('nv.tif')
            self.for_files = list(filter(lambda x: 'flor' in x, self.inpe_files))
            self.for_files.remove('flor.tif')
            
            print(len(self.arcs_files))
            self.data_files = self.arcs_files
            df_date = self.date2datetime(self.data_files)
            df_date_for = self.date2datetime(self.for_files)
            df_date_clouds = self.date2datetime(self.clouds_files, prefix=2, data_type='clouds')
            print(df_date_clouds.head(10))
            print(df_date_for.head(10))
            print(df_date.head(10))
            # df_date_train = df_date[df_date['date'] < '2019-01-01']
            # df_date_val = df_date[(df_date['date'] >= '2019-01-01') & (df_date['date'] < '2020-01-01')]
            # df_date_test = df_date[df_date['date'] >= '2020-01-01']
            df_date_train, df_date_val, df_date_test = self.split_train_val_test(df_date)
            df_date_train_for, _, _ = self.split_train_val_test(df_date_for)
            df_date_train_clouds, _, _ = self.split_train_val_test(df_date_clouds)
            
            self.data_files = self.sliding_window(df_date_train.date, 3)
            self.val_files = self.sliding_window(df_date_val.date, 3)
            self.test_files = self.sliding_window(df_date_test.date, 3)
            
            # self.data_files, self.val_files = train_test_split(self.data_files, test_size=0.2, random_state=42)
            self.mean, self.std = self._get_mean_std(self.data_files)
            self.mean_for, self.std_for = self._get_mean_std(df_date_train_for, 'flor')
            self.mean_clouds, self.std_clouds = self._get_mean_std(df_date_train_clouds, 'nv')
            
            
            # print(len(self.data_files))
            # 1/0
        else:
            self.data_files = val_data
            self.mean = means[0]
            self.std = stds[0]
            self.mean_for = means[1]
            self.std_for = stds[1]
            self.mean_clouds = means[2]
            self.std_clouds = stds[2]
            self.val_files = self.data_files
        
        if Debug:
            self.data_files = self.data_files[:20]
            
        # Hidrografia
        self.hidr_files = load_tif_image(inpe_folder_path / 'hidr.tif')
        self.hidr_files = self.normalize_non_temporal(self.hidr_files)
        # Não Floresta (Ex: Banco de Areia)
        self.no_for_files = load_tif_image(inpe_folder_path / 'nf.tif')
        self.no_for_files = self.normalize_non_temporal(self.no_for_files)
        # Categorias Fundiarias
        self.rodnofic = load_tif_image(inpe_folder_path / 'rodnofic.tif')
        self.rodofic = load_tif_image(inpe_folder_path / 'rodofic.tif')
        self.disturb = load_tif_image(inpe_folder_path / 'distUrb.tif')
        self.distrios = load_tif_image(inpe_folder_path / 'distrios.tif')
        self.distport = load_tif_image(inpe_folder_path / 'distport.tif')
        self.efams_apa = load_tif_image(inpe_folder_path / 'EFAMS_APA.tif')
        self.efams_ass = load_tif_image(inpe_folder_path / 'EFAMS_ASS.tif')
        self.efams_car = load_tif_image(inpe_folder_path / 'EFAMS_CAR.tif')
        self.efams_fpnd = load_tif_image(inpe_folder_path / 'EFAMS_FPND.tif')
        self.efams_ti = load_tif_image(inpe_folder_path / 'EFAMS_TI.tif')
        self.efams_uc = load_tif_image(inpe_folder_path / 'EFAMS_UC.tif')
        
        self.rodnofic = self.normalize_non_temporal(self.rodnofic)
        self.rodofic = self.normalize_non_temporal(self.rodofic)
        self.disturb = self.normalize_non_temporal(self.disturb)
        self.distrios = self.normalize_non_temporal(self.distrios)
        self.distport = self.normalize_non_temporal(self.distport)
        self.efams_apa = self.normalize_non_temporal(self.efams_apa)
        self.efams_ass = self.normalize_non_temporal(self.efams_ass)
        self.efams_car = self.normalize_non_temporal(self.efams_car)
        self.efams_fpnd = self.normalize_non_temporal(self.efams_fpnd)
        self.efams_ti = self.normalize_non_temporal(self.efams_ti)
        self.efams_uc = self.normalize_non_temporal(self.efams_uc)
            
        self.normalize = normalize
        self.transform = transform
    
    def split_train_val_test(self, df_date):
        df_date_train = df_date[df_date['date'] < '2019-01-01']
        df_date_val = df_date[(df_date['date'] >= '2019-01-01') & (df_date['date'] < '2020-01-01')]
        df_date_test = df_date[df_date['date'] >= '2020-01-01']
        return df_date_train, df_date_val, df_date_test
    
    def date2datetime(self, data_files, prefix=4, data_type='ArCS'):
        data_files.sort()
        func = lambda x: x.split('.tif')[0][prefix:]
        date_files = [func(file) for file in data_files]
        df_date = pd.DataFrame(date_files, columns=['date_str']) #0101
        if data_type == 'clouds':
            df_date['date'] = pd.to_datetime(df_date['date_str'], format="%Y%m%d")
            # df_date['date'] = df_date['date'].dt.strftime("%d%m%y")
        else:
            df_date['date'] = pd.to_datetime(df_date['date_str'], format="%d%m%y")
        df_date = df_date.sort_values(by='date')
        return df_date
    
    def normalize_non_temporal(self, data):
        '''Normalize the non-temporal channels'''
        data[data < -1e38] = -1
        _data = data[data != -1]
        mean = _data.mean(axis=0)
        std = _data.std(axis=0)
        print('Mean:', mean, 'Std:', std)
        return (data - mean) / std
    
    def get_validation_set(self):
        return self.val_files
    
    def get_test_set(self):
        return self.test_files
    
    def _get_mean_std(self, data_files, data_type='ArCS'):
        '''Get mean and std of the dataset'''
        # print(self.data_files)
        # print(type(data_files))
        if type(data_files) == list:
            df = pd.concat(data_files).to_frame(name='date')
        else:
            df = data_files.copy()
        df.drop_duplicates(inplace=True)
        if data_type == 'nv':
            df['date_str'] = df['date'].dt.strftime('%Y%m%d')
        else:
            df['date_str'] = df['date'].dt.strftime('%d%m%y')
            
        for i, file in enumerate(df.date_str):
            # img = load_tif_image(self.inpe_folder_path / ('ArCS' + file + '.tif'))
            # print(data_type + file + '.tif')
            img = load_tif_image(self.inpe_folder_path / (data_type + file + '.tif'))
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
            img = load_tif_image(self.inpe_folder_path / ('ArCS' + file + '.tif'))
            img_flor = load_tif_image(self.inpe_folder_path / ('flor' + file + '.tif'))
            cloud_date = datetime.strptime(file, '%d%m%y').strftime('%Y%m%d')
            try:
                img_clouds = load_tif_image(self.inpe_folder_path / ('nv' + cloud_date + '.tif'))
            except:
                cloud_date = cloud_date.replace('16', '15')
                img_clouds = load_tif_image(self.inpe_folder_path / ('nv' + cloud_date + '.tif'))
 
            if i == 0:
                data = np.expand_dims(img, axis=0)
                data_flor = np.expand_dims(img_flor, axis=0)
                data_clouds = np.expand_dims(img_clouds, axis=0)
            else:
                data = np.concatenate((data, np.expand_dims(img, axis=0)), axis=0)
                data_flor = np.concatenate((data_flor, np.expand_dims(img_flor, axis=0)), axis=0)
                data_clouds = np.concatenate((data_clouds, np.expand_dims(img_clouds, axis=0)), axis=0)
        
        data[data < -1e38] = 0
        data_flor[data_flor < -1e38] = 0
        data_clouds[data_clouds < -1e38] = 0
        data[data == -1] = 0
        data_flor[data_flor == -1] = 0
        data_clouds[data_clouds == -1] = 0

        labels = load_tif_image(self.inpe_folder_path / ('ArCS' + filenames[-1] + '.tif'))
        labels[labels < -1e38] = 0
        labels[labels == -1] = 0
        
        # print(data.shape, labels.shape)
        # print(self.mean.shape, self.std.shape)
        
                    
        # print('DEBUG')
        # print(data[data < 0])
        # print(labels[labels < 0])
        # 1/0
        
        if self.normalize:
            data = data - self.mean / self.std
            data_flor = data_flor - self.mean_for / self.std_for
            data_clouds = data_clouds - self.mean_clouds / self.std_clouds
            
        # data = np.stack((data, data_flor, data_clouds), axis=1)
        catg_fundi = np.stack((self.rodnofic, self.rodofic, self.disturb, self.distrios, self.distport,\
            self.efams_apa, self.efams_ass, self.efams_car, self.efams_fpnd, self.efams_ti, self.efams_uc), axis=0)
        catg_fundi = np.stack((catg_fundi, catg_fundi), axis=0)
        # data = np.concatenate((data, catg_fundi), axis=1)
        data = np.expand_dims(data, axis=1)
            
        if self.transform:
            # For albumentations the image needs to be in shape (H, W, C)
            transformed = self.transform(image=data.reshape(data.shape[2], data.shape[3], -1), mask=labels)
            data = transformed['image'].reshape(data.shape[0], data.shape[1], data.shape[2], data.shape[3])
            labels = transformed['mask']
        else:
            data = torch.tensor(data)
            labels = torch.tensor(labels)
            
        # print('DEBUG')
        # print(data[data < 0])
        # print(labels[labels < 0])
        # 1/0
            
        return data.float(), labels.unsqueeze(0).unsqueeze(0).float()
    
# ----------------------------------------------------------------------------------------------

class IbamaDETER1km_Dataset(Dataset):
    def __init__(self, root_dir: Path, normalize: bool=True, transform: torchvision.transforms=None,
                Debug: bool=False, mode: str='train', patch_size=64, overlap=0.1, min_def=0.02, window_size=6,\
                    val_data=None, mask_val_data=None, means=None, stds=None, dilation_size=-1):
        super(IbamaDETER1km_Dataset, self).__init__()
        self.root_dir = root_dir
        ibama_folder_path = root_dir / 'tiff_filled'
        self.ibama_folder_path = ibama_folder_path
        if mode == 'train':
            # deter_img = load_tif_image('data/DETER/deter_increments_1km_1week.tif')
            deter_img = load_npy_image('data/DETER/deter_increments_1km_1week.npy')
            
            # mask = load_tif_image('data/IBAMA_INPE/1K/tiff_filled/mask.tif')
            mask = load_npy_image('data/IBAMA_INPE/1K/tiff_filled/mask.npy')
            mask = mask[:deter_img.shape[1], :deter_img.shape[2]]
            
            # new_deter_img = []
            # for i in range(deter_img.shape[0]):
            #     new_deter_img.append(block_reduce(deter_img[i], (4, 4), np.sum))
            # deter_img = np.stack(new_deter_img, axis=0)
            # del new_deter_img
            # mask = block_reduce(mask, (2, 2), np.sum)                
            
            deter_img[:, mask == 0] = -1
            deter_img[deter_img > 0] = 1
            # xcut = (deter_img.shape[1] // patch_size) * patch_size
            # ycut = (deter_img.shape[2] // patch_size) * patch_size
            # deter_img = deter_img[:, :xcut, :ycut]
    
            self.img_shape = deter_img.shape
            # Each year has 48 weeks
            # 2020, 2021 = 2 * 48 = 96
            deter_img_train = deter_img[:96]
            # 2022 = 1 * 48 = 48
            deter_img_val = deter_img[96:(96 + 48)]
            # 2023 = 1 * 48 = 48
            # deter_img_test = deter_img[(96 + 48):(96 + 48 + 48)]
            deter_img_test = deter_img[(96 + 48):(96 + 48 + 48)]
            del deter_img
            
            self.mean = deter_img_train[deter_img_train != -1].mean()
            self.std = deter_img_train[deter_img_train != -1].std()
            
            train_patches = preprocess_patches(deter_img_train, patch_size=patch_size, overlap=overlap)
            print('Train Patches:', train_patches.shape)
            del deter_img_train
            val_patches = preprocess_patches(deter_img_val, patch_size=patch_size, overlap=0)
            print('Validation Patches:', val_patches.shape)
            del deter_img_val
            # test_patches = preprocess_patches(deter_img_test, patch_size=patch_size, overlap=0)
            test_patches = extract_temporal_sorted_patches(deter_img_test, patch_size)
            test_patches = test_patches.transpose(1, 0, 2, 3)
            print('Test Patches:', test_patches.shape)
            del deter_img_test
            
            # mask_train_patches = preprocess_patches(mask, patch_size=patch_size, overlap=overlap)
            # print('Mask Train Patches:', mask_train_patches.shape)
            # mask_val_patches = preprocess_patches(mask, patch_size=patch_size, overlap=0)
            # print('Mask Validation Patches:', mask_val_patches.shape)
            # # mask_test_patches = preprocess_patches(mask, patch_size=patch_size, overlap=0)
            # mask_test_patches = extract_sorted_patches(mask, patch_size)
            # print('Mask Test Patches:', mask_test_patches.shape)
            
            
            # self.data_files = train_patches
            # self.val_files = val_patches
            self.data_files, self.mask_files, _ = divide_pred_windows(train_patches, min_def=min_def, window_size=window_size)
            # print(f'Training shape: {self.data_files.shape} - {self.mask_files.shape}')
            print(f'Training shape: {self.data_files.shape}')
            
            # Only using min_def != 0 to speed training. Validation should not use this
            self.val_files, self.mask_val_files, _ = divide_pred_windows(val_patches, min_def=min_def/100,\
                window_size=window_size)
            # print(f'Validation shape: {self.val_files.shape} - {self.mask_val_files.shape}')
            print(f'Validation shape: {self.val_files.shape}')
            
            self.test_files, self.mask_test_files, _ = divide_pred_windows(test_patches, min_def=0,\
                window_size=window_size)
            # print(f'Test shape: {self.test_files.shape} - {self.mask_test_files.shape}')
            print(f'Test shape: {self.test_files.shape}')
            
            
        else:
            self.data_files = val_data
            self.mask_files = mask_val_data
            self.mean = means[0]
            self.std = stds[0]
        
        if Debug:
            self.data_files = self.data_files[:20]
            
        self.normalize = normalize
        self.transform = transform
        if dilation_size == -1:
            self.kernel = None
        else:
            self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
        self.mode = mode
    
    def get_validation_set(self):
        return self.val_files, self.mask_val_files
    
    def get_test_set(self):
        return self.test_files, self.mask_test_files
    
    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        # print(self.root_dir / self.data_files[index])
        patch_window = self.data_files[index]
        # print(patch_window.shape)
        # mask = self.mask_files[index]
        
        # data = patch_window[:-2]
        # labels = patch_window[-2:]
        
        # Apply Legal Amazon Mask
        # No negative values should be present in input data
        # data[:, mask == 0] = 0
        # data[data > 0] = 1 # Make every deforestation area to 1
        # data[data == -1] = 0
        # Negative values will be filtered in the cost function
        # labels[:, mask == 0] = -1
        # labels[labels > 0] = 1
        
        if self.mode == 'train' and self.kernel is not None:
            for i in range(patch_window.shape[0]):
                patches_sum = patch_window[i]
                patches_sum = patches_sum[patches_sum != -1]
                if np.sum(patches_sum) > 0 and i not in [0, 1, 2, 3]:
                    patch_window[i] = cv2.dilate(patch_window[i], self.kernel, iterations=1)
                    # labels = cv2.dilate(labels, self.kernel, iterations=1)
        # labels[:, mask == 0] = -1          
        
        data = patch_window[:-2]
        labels = patch_window[-2:]
        
        #! Label smoothing
        # labels[labels == 1] = labels[labels == 1] - 0.1
        # labels[labels == 0] = labels[labels == 0] + 0.1
        
        # Avoid negative values for the input
        data[data < 0] = 0
        
        if self.normalize:
            data = data - self.mean / self.std
            
        if self.transform:
            # For albumentations the image needs to be in shape (H, W, C)
            # print(data.shape, labels.shape)
            transformed = self.transform(image=data.reshape(data.shape[1], data.shape[2], -1), mask=labels.reshape(data.shape[1], data.shape[2], -1))
            data = transformed['image'].reshape(data.shape[0], data.shape[1], data.shape[2])
            labels = transformed['mask'].reshape(labels.shape[0], labels.shape[1], labels.shape[2])
        else:
            data = torch.tensor(data)
            labels = torch.tensor(labels)
        # print(data.shape, labels.shape)
        return data.unsqueeze(1).float(), labels.unsqueeze(1).float()