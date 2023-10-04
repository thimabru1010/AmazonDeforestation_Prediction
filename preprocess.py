import argparse
import numpy as np
from skimage.util.shape import view_as_windows
from osgeo import gdal
import matplotlib.pyplot as plt

def load_tif_image(tif_path):
    gdal_header = gdal.Open(tif_path)
    return gdal_header.ReadAsArray()

def extract_patches(image: np.ndarray, patch_size: int, stride: int) -> np.ndarray:
    
    window_shape_array = (image.shape[0], patch_size, patch_size)
    patches_array = np.array(view_as_windows(image, window_shape_array, step=stride))

    # patches_array = patches_array.reshape((-1,) + window_shape_array)
    patches_array = patches_array.reshape((-1, patch_size, patch_size))
    print('Patches extracted')

    return patches_array

# 0-11->2017, 12-21->2018, 22-35-> 2019
train_path = '/home/thiago/AmazonDeforestation_Prediction/AmazonData/Dataset_Felipe/train.tif'
img_train = load_tif_image(train_path)
print(img_train.shape)

print(img_train.shape)

bins = np.arange(0, 255, step=1)
bin_counts, bin_edges = np.histogram(img_train, bins)
# print(bin_counts)
# print(bin_edges)

# Mask separating pixels inside (1) and outside (2) legal amazon
mask_path = '/home/thiago/AmazonDeforestation_Prediction/AmazonData/Dataset_Felipe/area.tif'
mask = load_tif_image(mask_path)
print(mask.shape)

mask[mask == 0.0] = 2.0
mask[mask == 1] = 0.0

bins = np.arange(-1, 255, step=1)
bin_counts, bin_edges = np.histogram(mask, bins)
# print(bin_counts)
# print(bin_edges)

# img_train = img_train + mask
for i in range(img_train.shape[0]):
    img_train[i, :, :][mask == 2.0] = 2.0

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
# plt.tight_layout()

# # Show the plot
# plt.show()

img_val = img_train[24:36, :, :]
img_train = img_train[:24, :, :]


# 0-11->2020, 12-21->2021, 22-35-> 2022
# test_path = '/home/thiago/AmazonDeforestation_Prediction/AmazonData/Dataset_Felipe/test.tif'
# img_test = load_tif_image(test_path)

# img_train = img_train.reshape(3, -1, img_train.shape[1], img_train.shape[2])
# print(img_train.shape)

PATCH_OVERLAP = 0.8
PATCH_SIZE = 256
train_stride = int((1-PATCH_OVERLAP)*PATCH_SIZE)

patches = extract_patches(img_train, patch_size=PATCH_SIZE, stride=train_stride)
print(patches.shape)

# Filter deforestation
MIN_DEF = 0.001


patches = patches.reshape((-1, 3, int(img_train.shape[0]/3), PATCH_SIZE, PATCH_SIZE))
# patches = patches.reshape((-1, 24, PATCH_SIZE, PATCH_SIZE))
print(patches.shape)

# print(img_train.shape[0], int(img_train.shape[0]/3))
# keep_patches = np.mean((patches == 1), axis=(1, 2, 3)) > MIN_DEF
keep_patches = np.mean((patches == 1), axis=(3, 4)) > MIN_DEF
print(keep_patches.shape)
patches_filt = patches[keep_patches]
print(patches_filt.shape)

# patches_filt = patches_filt.reshape((24, PATCH_SIZE, PATCH_SIZE, -1))
patches_filt = patches_filt.reshape((-1, 3, int(img_train.shape[0]/3), PATCH_SIZE, PATCH_SIZE))
print(patches_filt.shape)


