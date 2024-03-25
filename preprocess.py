import argparse
import numpy as np
from skimage.util.shape import view_as_windows
try:
    from osgeo import gdal
except ImportError:
    print("osgeo module is not installed. Please install it with pip install GDAL")
# from osgeo import gdal
import matplotlib.pyplot as plt
import os
import pathlib
from tqdm import tqdm
from time import time

gdal.PushErrorHandler('CPLQuietErrorHandler')
def load_tif_image(tif_path):
    gdal_header = gdal.Open(str(tif_path))
    return gdal_header.ReadAsArray()

def load_npy_image(npy_path):
    return np.load(str(npy_path))

def apply_legal_amazon_mask(input_image: np.array, amazon_mask: np.array):
    ''' Apply Legal Amazon mask '''
    for i in range(input_image.shape[0]):
        input_image[i, :, :][amazon_mask == 2.0] = 2
    return input_image

def extract_patches(image: np.ndarray, patch_size: int, stride: int) -> np.ndarray:
    if len(image.shape) == 3:
        window_shape_array = (image.shape[0], patch_size, patch_size)
        print(image.shape)
        return np.array(view_as_windows(image, window_shape_array, step=stride)).reshape((-1,) + window_shape_array)
    elif len(image.shape) == 2:
        window_shape_array = (patch_size, patch_size)
        print(image.shape)
        return np.array(view_as_windows(image, window_shape_array, step=stride)).reshape((-1,) + window_shape_array)

def extract_temporal_sorted_patches(deter_img, patch_size):
    patches = []
    for t in range(deter_img.shape[0]):
        img = deter_img[t]
        patches_t = extract_sorted_patches(img, patch_size=patch_size)
        patches.append(patches_t)

    return np.stack(patches, axis=0)

def extract_sorted_patches(img, patch_size):
    patches = []
    for i in range(0, img.shape[0] - patch_size + 1, patch_size):
        for j in range(0, img.shape[1] - patch_size + 1, patch_size):
            patch = img[i:i + patch_size, j:j + patch_size]
            patches.append(patch)
    return np.stack(patches, axis=0)

def reconstruct_sorted_patches(patches, img_shape, patch_size):
    img = np.zeros(img_shape)
    idx = 0
    for i in range(0, img_shape[0] - patch_size + 1, patch_size):
        for j in range(0, img_shape[1] - patch_size + 1, patch_size):
            img[i:i + patch_size, j:j + patch_size] = patches[idx]
            idx += 1
    return img

def preprocess_patches(img: np.ndarray, patch_size: int, overlap: int):
    print('Extracting patches...')
    stride = int((1-overlap)*patch_size)
    patches = extract_patches(img, patch_size=patch_size, stride=stride)
    # print(f'Patches extracted: {patches.shape}')
    return patches

def reconstruct_time_patches(preds: np.ndarray, patch_size: int=64, time_idx: int=43, original_img_shape: tuple=(2333, 3005), len_patches: int=1):
    #! OBS: 44 = 46 - 2
    # 46 é o número de quinzenas em grupos de 3. 2 são as duas quinzenas iniciais usadas para a primeira previsão.
    time_idx = preds.shape[0] // len_patches
    print('Time idx:', time_idx)
    div_time = preds.shape[0] // time_idx
    print('Div time:', div_time)
    patches = []
    for i in range(div_time):
        windowed_patch = preds[i * time_idx: (i + 1) * time_idx]
        # print(windowed_patch.shape)
        init_patches = windowed_patch[0]
        non_duplicated = windowed_patch[1:, -1]
        patches_t = np.concatenate((init_patches, non_duplicated), axis=0)
        patches.append(patches_t)
        # print(patches.shape)
    patches = np.stack(patches, axis=0)
    print('Grouped patches shape:', patches.shape)
    
    images_reconstructed = []
    for i in range(patches.shape[1]):
        # print(patches.shape)
        img_reconstructed = reconstruct_sorted_patches(patches[:, i], original_img_shape, patch_size=patch_size)
        images_reconstructed.append(img_reconstructed)
        
    # np.save('data/reconstructed_images.npy', np.stack(images_reconstructed, axis=0))
    return np.stack(images_reconstructed, axis=0)

def divide_pred_windows(patches: np.ndarray, min_def: float, window_size: int=6, pred_horizon: int=2, mask_patches: np.ndarray=None) -> np.ndarray:
    skipped_count = 0
    # if mask_patches is not None:
    windowed_mask_patches = []
    windowed_patches = []
    indexes = []
    total_iterations = patches.shape[0] * (patches.shape[1] - window_size + 1)
    with tqdm(total=total_iterations, desc='Dividing in prediction windows') as pbar:
        # Loop trough patches
        for i, patch in enumerate(patches):
            # Loop through time windows
            for j in range(patch.shape[0] - window_size + 1):
                windowed_patch = patch[j:j+window_size]
                label = windowed_patch[-pred_horizon:]
                # print(label.shape)
                _label = label[:, mask_patches[i] == 1]
                mean = np.mean(_label, axis=(0, 1))
                # Deal with Nan
                if np.isnan(mean): mean = 0
                if mean < 0: mean = 0
                if mean < min_def:
                    # print('entered if')
                    skipped_count += 1
                    pbar.update(1)
                    continue
                # print(np.mean(mask_patches[i], axis=(0, 1)))
                # print(label.shape, _label.shape)
                # print(np.mean(_label, axis=(0, 1)))
                indexes.append((i, j, j+window_size))
                windowed_patches.append(windowed_patch)
                if mask_patches is not None:
                    windowed_mask_patches.append(mask_patches[i])
                pbar.update(1)
    print(f'{skipped_count} Skipped Images')
    if mask_patches is not None:
        windowed_mask_patches = np.concatenate(windowed_mask_patches, axis=0).reshape((-1,) + mask_patches.shape[1:])
    else:
        windowed_mask_patches = None
    return np.concatenate(windowed_patches, axis=0).reshape((-1, window_size) + patches.shape[2:]), windowed_mask_patches, indexes


def save_patches(patches: np.ndarray, modality: str, save_path: pathlib.Path, window_size: int=5):
    folder_path = save_path / modality
    folder_path_input = folder_path / 'Input'
    folder_path_labels = folder_path / 'Labels'
    os.makedirs(folder_path_input, exist_ok=True)
    os.makedirs(folder_path_labels, exist_ok=True)

    saved_count = 0
    skipped_count = 0
    def_count = 0
    no_def_count = 0
    
    total_iterations = patches.shape[0] * (patches.shape[1] - window_size + 1)
    with tqdm(total=total_iterations, desc=f'{modality}:Saving Patches') as pbar:
        for i, patch in enumerate(patches):
            # print(patch.shape)
            for j in range(patch.shape[0] - window_size + 1):
                windowed_patch = patch[j:j+window_size]
                # print(windowed_patch.shape)
                # print(patch[j:j+window_size-1].shape, patch[j+window_size].shape)
                input_windowed_patch = windowed_patch[:-1]
                input_windowed_patch[input_windowed_patch == 50] = 0
                labels_windowed_patch = windowed_patch[-1]
                labels_windowed_patch[labels_windowed_patch == 50] = -1
                if np.mean(labels_windowed_patch > 0, axis=(0, 1)) < args.min_def:
                        skipped_count += 1
                        pbar.update(1)
                        continue
                labels_windowed_patch[labels_windowed_patch > 0] = 1.0
                def_count += np.sum(labels_windowed_patch > 0)
                no_def_count += np.sum(labels_windowed_patch == 0)
                labels_windowed_patch[labels_windowed_patch == -1] = 50.0
                # print(input_windowed_patch.shape, labels_windowed_patch.shape)
                np.save(os.path.join(folder_path_input, f'patch={i}_trimester_window={j}.npy'), input_windowed_patch)
                np.save(os.path.join(folder_path_labels, f'patch={i}_trimester_window={j}.npy'), labels_windowed_patch)
                saved_count += 1
                pbar.update(1)
    print(f'{saved_count} Saved Images')
    print(f'{skipped_count} Skipped Images')
    total_classes = def_count + no_def_count
    print(f'Deforestation: {def_count/total_classes} ')
    print(f'No Deforestation: {no_def_count/total_classes} ')

def apply_legal_amazon_mask(input_image: np.array, amazon_mask: np.array):
    ''' Apply Legal Amazon mask '''
    for i in range(input_image.shape[0]):
        input_image[i, :, :][amazon_mask == 50.0] = 50.0
    return input_image

if __name__== '__main__':
    parser = argparse.ArgumentParser(
    description='prepare the files to be used in the training/testing steps')

    parser.add_argument('--train_path', type=pathlib.Path,
        default = '/home/thiago/AmazonDeforestation_Prediction/AmazonData/Dataset_Felipe/train.tif',
        help = 'Path to DETR warnings image (.tif) file from year 2017 to 2019')
        
    parser.add_argument('--test_path', type=pathlib.Path,
        default = '/home/thiago/AmazonDeforestation_Prediction/AmazonData/Dataset_Felipe/test.tif',
        help = 'Path to DETR warnings image (.tif) file from year 2020 to 2022')

    parser.add_argument('--mask_path', type=pathlib.Path,
        default = '/home/thiago/AmazonDeforestation_Prediction/AmazonData/Dataset_Felipe/area.tif',
        help = 'Path to mask area containing categories for inside and outside legal amazon')

    parser.add_argument('--save_path', type=pathlib.Path,
        default = 'data/Dataset/DETR_Patches',
        help = 'Output path to where save patches')

    parser.add_argument('--min_def', type=float, default=0.02,
        help = 'Minimum deforestation threshold acceptable in each patch')

    parser.add_argument('--overlap', type=float, default=0.8,
        help = 'Minimum deforestation threshold acceptable in each patch')

    parser.add_argument('--patch_size', type=int, default=64,
        help = 'Size of each patch extracted from the whole image')

    parser.add_argument('--test_fill', type=int, default=1,
        help = 'Number of trimesters of test set to complete validation set')
    
    parser.add_argument('--aggregation', '-agg', type=str, default='max',
        help = 'Type of aggregation to use when grouping months in trimesters')

    args = parser.parse_args()

    print('Preprocessing training images')
    # 0-12->2017, 13-22->2018, 23-36-> 2019
    img_train = load_tif_image(args.train_path)
    # Mask separating pixels inside (1) and outside (2) legal amazon
    mask = load_tif_image(args.mask_path)
    #! Loading test to balance validation set
    # 0-12->2020, 13-22->2021, 23-36-> 2022
    img_test = load_tif_image(args.test_path)

    # Group in trimesters by the maximum
    if args.aggregation == 'max':
        img_test = img_test.reshape((3, -1, img_test.shape[1], img_test.shape[2])).max(axis=0)
        img_train = img_train.reshape((3, -1, img_train.shape[1], img_train.shape[2])).max(axis=0)
    if args.aggregation == 'sum':
        img_test = img_test.reshape((3, -1, img_test.shape[1], img_test.shape[2])).sum(axis=0)
        img_train = img_train.reshape((3, -1, img_train.shape[1], img_train.shape[2])).sum(axis=0)
    
    # Cut image H x W to have a integer number of patches
    height_cut = (img_train.shape[1] // args.patch_size) * args.patch_size
    width_cut = (img_train.shape[2] // args.patch_size) * args.patch_size
    
    img_train = img_train[:, :height_cut, :width_cut]
    mask = mask[:height_cut, :width_cut]
    img_test = img_test[:, :height_cut, :width_cut]

    mask[mask == 0.0] = 50.0
    mask[mask == 1] = 0.0

    img_train = apply_legal_amazon_mask(img_train, mask)
    img_test = apply_legal_amazon_mask(img_test, mask)
    
    # 8-12 tri->2019 + 0 tri -> 2020
    # img_val = np.concatenate((img_train[24:36, :, :], img_test[:args.test_fill, :, :]), axis=0)
    img_val = np.concatenate((img_train[8:, :, :], img_test[:args.test_fill, :, :]), axis=0)
    # 0-4 tri->2017, 4-8 tri->2018
    img_train = img_train[:8, :, :]
    
    print(f'Img train shape: {img_train.shape}')
    print(f'Img val shape: {img_val.shape}')
    print(f'Amazon mask shape shape: {mask.shape}')
    # print(f'Img test shape: {img_test.shape}')
    
    # Preprocessing
    print('Starting preprocess in training...')
    train_stride = int((1-args.overlap)*args.patch_size)
    patches_filt = preprocess_patches(img_train, args.patch_size, train_stride)

    print(patches_filt.shape)

    save_patches(patches_filt, 'Train', args.save_path)
        
    print('\nStarting preprocess in validation...')
    # patches_filt = preprocess_patches(img_val, args.patch_size, train_stride)
    patches_filt = preprocess_patches(img_val, args.patch_size, train_stride)
    
    save_patches(patches_filt, 'Val', args.save_path)
    


