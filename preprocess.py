import argparse
import numpy as np
from skimage.util.shape import view_as_windows
from osgeo import gdal
import matplotlib.pyplot as plt
import os
import pathlib
from tqdm import tqdm

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

parser.add_argument('--min_def', type=float, default=0.001,
    help = 'Minimum deforestation threshold acceptable in each patch')

parser.add_argument('--overlap', type=float, default=0.8,
    help = 'Minimum deforestation threshold acceptable in each patch')

parser.add_argument('--patch_size', type=int, default=256,
    help = 'Size of each patch extracted from the whole image')

parser.add_argument('--test_fill', type=int, default=3,
    help = 'Number of months of test set to complete validation set')

args = parser.parse_args()

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
    return np.array(view_as_windows(image, window_shape_array, step=stride)).reshape((-1,) + window_shape_array)

def preprocess_patches(img_train: np.ndarray, patch_size: int, overlap_stride: int):
    print('Extracting patches...')
    patches = extract_patches(img_train, patch_size=patch_size, stride=overlap_stride)
    patches = patches.reshape((-1, img_train.shape[0], patch_size, patch_size))
    print(f'Patches extracted: {patches.shape}')

    # Filter deforestation
    #! Here we are eliminating the patches along all temporal axis 
    #! if their sum along the months isn't above the threshold
    print('Filtering no deforestation patches...')
    keep_patches = np.mean((patches == 1), axis=(1, 2, 3)) > args.min_def
    patches_filt = patches[keep_patches]
    print(f'Patches filtered: {patches_filt.shape}')

    patches_filt = patches_filt.reshape((-1, 3, int(img_train.shape[0]/3), patch_size, patch_size))
    print(f'Reshaping in trimesters: {patches_filt.shape}')
    return patches_filt
    
def save_patches(patches: np.ndarray, modality: str, save_path: pathlib.Path, window_size: int=5, window_stride: int=1):
    folder_path = save_path / modality
    os.makedirs(folder_path, exist_ok=True)
                
    total_iterations = sum((patch.shape[1] - window_size + 1) // window_stride for patch in patches)
    with tqdm(total=total_iterations, desc='Saving Patches') as pbar:
        for i, patch in enumerate(patches):
            for j in range(0, patch.shape[1] - window_size + 1, window_stride):
                windowed_patch = patch[:, j:j+window_size]
                np.save(os.path.join(folder_path, f'patch={i}_trimester_window={j}.npy'), windowed_patch)
                pbar.update(1)

if __name__== '__main__':
    print('Preprocessing training images')
    # 0-12->2017, 13-22->2018, 23-36-> 2019
    img_train = load_tif_image(args.train_path)
    print(f'Img train shape: {img_train.shape}')

    # bin_counts, bin_edges = np.histogram(img_train, bins=np.arange(0, 255, step=1))

    # Mask separating pixels inside (1) and outside (2) legal amazon
    mask = load_tif_image(args.mask_path)
    print(mask.shape)

    mask[mask == 0.0] = 2.0
    mask[mask == 1] = 0.0

    bins = np.arange(-1, 255, step=1)
    bin_counts, bin_edges = np.histogram(mask, bins)

    # img_train = img_train + mask
    for i in range(img_train.shape[0]):
        img_train[i, :, :][mask == 2.0] = 2.0

    #! Loading test to balance validation set
    # 0-12->2020, 13-22->2021, 23-36-> 2022
    img_test = load_tif_image(args.test_path)
    # print(f'Img test shape: {img_test.shape}')
    
    # 23-36->2019 + 0-3->2020
    img_val = np.concatenate((img_train[24:36, :, :], img_test[:args.test_fill, :, :]), axis=0)
    
    # 0-12->2017, 13-22->2018
    img_train = img_train[:24, :, :]
    
    ## Preprocessing Training images
    print('Starting preprocess in training...')
    train_stride = int((1-args.overlap)*args.patch_size)
    patches_filt = preprocess_patches(img_train, args.patch_size, train_stride)

    save_patches(patches_filt, 'Train', args.save_path)
        
    ## Preprocess Validation images
    print('\nStarting preprocess in validation...')
    patches_filt = preprocess_patches(img_val, args.patch_size, train_stride)
    
    save_patches(patches_filt, 'Val', args.save_path)
        
    ## Preprocess Test images
    print('\nStarting preprocess in testing...')
    # 0-12->2020, 13-22->2021, 23-36-> 2022
    img_test = img_test[args.test_fill:, :, :]
    print(f'Img test shape: {img_test.shape}')
    
    # img_test = img_test + mask
    for i in range(img_test.shape[0]):
        img_test[i, :, :][mask == 2.0] = 2.0
    
    print('Extracting patches...')
    # For Test set we extract images with no overlap and with all kinds deforestation percentage
    patches_test = extract_patches(img_test, patch_size=args.patch_size, stride=args.patch_size).reshape((-1, 3, int(img_test.shape[0]/3), args.patch_size, args.patch_size))
    print(f'Patches extracted: {patches_test.shape}')
    
    save_patches(patches_test, 'Test', args.save_path)
    


