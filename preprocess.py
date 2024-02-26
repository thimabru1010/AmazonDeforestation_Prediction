import argparse
import numpy as np
from skimage.util.shape import view_as_windows
from osgeo import gdal
import matplotlib.pyplot as plt
import os
import pathlib
from tqdm import tqdm
from time import time

gdal.PushErrorHandler('CPLQuietErrorHandler')

def plot_images(image1, image2, area):
    # Create a figure and axes with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # Display the image using imshow
    cax1 = ax1.imshow(img_train[30, :, :])
    # Add a colorbar to the plot
    cbar1 = fig.colorbar(cax1)

    # Optionally, you can set axis labels and a title
    ax1.set_xlabel('X Label')
    ax1.set_ylabel('Y Label')
    ax1.set_title('Image Plot')

    # Display the image using imshow
    cax2 = ax2.imshow(mask)
    # Add a colorbar to the plot
    cbar2 = fig.colorbar(cax2)

    # Optionally, you can set axis labels and a title
    ax2.set_xlabel('X Label')
    ax2.set_ylabel('Y Label')
    ax2.set_title('Image Plot')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

def load_tif_image(tif_path):
    gdal_header = gdal.Open(str(tif_path))
    return gdal_header.ReadAsArray()

def extract_patches(image: np.ndarray, patch_size: int, stride: int) -> np.ndarray:
    window_shape_array = (image.shape[0], patch_size, patch_size)
    print(image.shape)
    return np.array(view_as_windows(image, window_shape_array, step=stride)).reshape((-1,) + window_shape_array)

def preprocess_patches(img_train: np.ndarray, patch_size: int, overlap_stride: int):
    print('Extracting patches...')
    patches = extract_patches(img_train, patch_size=patch_size, stride=overlap_stride)
    print(f'Patches extracted: {patches.shape}')

    # Filter deforestation
    #! Here we are eliminating the patches along all temporal axis 
    #! if their sum along the months isn't above the threshold
    # print('Filtering no deforestation patches...')
    # print(patches.shape)
    # keep_patches = np.mean((patches == 1), axis=(1, 2, 3)) > args.min_def
    # patches_filt = patches[keep_patches]
    # print(f'Patches filtered: {patches_filt.shape}')
    # return patches_filt
    return patches

def prep_INPE25km(img_train: np.ndarray, img_test: np.ndarray, mask: np.ndarray, args: argparse.Namespace):
    pass

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
    


