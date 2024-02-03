import numpy as np
import pandas as pd
import geopandas as gpd
import GiovConfig as config
from tqdm import tqdm
from GiovanniDataset import GiovanniDataset
import torch
from pathlib import Path
import imageio

def load_data(config):
    #! Loads data
    # load legal amazon limits
    am_bounds = gpd.read_file(config.AMAZON_FRONTIER_DATA)
    # load frames idx detail
    frames_idx = pd.read_csv(config.TR_FRAMES_IDX, index_col=0)
    # load frames deforestation area history
    deforestation = pd.read_csv(config.TR_DEFORESTATION, index_col=0)
    deforestation["quarter_date"] = pd.to_datetime(deforestation["quarter_date"])

    # counties
    frames_county = pd.read_csv(config.TR_COUNTIES, index_col=0)
    counties_defor = pd.read_csv(config.TR_COUNTIES_DEFOR, index_col=0)

    # precipitations
    precip = pd.read_csv(config.TR_RAIN_AVG)
    precip["quarter_date"] = pd.to_datetime(precip["dt"])

    # terrain position index
    tpi = pd.read_csv(config.TR_TPI, skiprows=1)\
        .rename(columns={"Unnamed: 0": "frame_id"})

    # past scores
    past_scores = pd.read_csv(config.TR_PAST_SCORES)
    past_scores["variable"] = pd.to_datetime(past_scores["variable"])

    # night lights
    night_light = pd.read_csv(config.TR_NIGHT_LIGHT)
    night_light["dt"] = pd.to_datetime(night_light["dt"])
    
    return am_bounds, frames_idx, deforestation, frames_county, counties_defor, precip, tpi, past_scores, night_light

def create_grids(am_bounds, frames_idx, deforestation, frames_county, counties_defor, precip, tpi, past_scores, night_light):
    #! create limits history grid
    time_grid = np.zeros((len(config.TIME_STEPS), frames_idx["x"].max() - frames_idx["x"].min() + 1, frames_idx["y"].max() - frames_idx["y"].min() + 1))
    for t, dt in enumerate(config.TIME_STEPS):
        defor_area = (
            deforestation[
                deforestation["quarter_date"] == dt
            ].set_index("frame_id")["area"] +\
            pd.Series(0, index=frames_idx.index)
        ).fillna(0).sort_index()
        time_grid[t, :, :] = defor_area.values.reshape(time_grid[0, :, :].shape)
    
    # Population and density
    county_data = np.zeros((2, frames_idx["x"].max() - frames_idx["x"].min() + 1, frames_idx["y"].max() - frames_idx["y"].min() + 1))
    county_data[0] = (
        frames_county.set_index("frame_id")["populacao"] +\
        pd.Series(0, index=frames_idx.index)
    ).fillna(0).\
        values.reshape(county_data.shape[1:])

    county_data[1] = (
        frames_county.set_index("frame_id")["densidade"] +\
        pd.Series(0, index=frames_idx.index)
    ).fillna(0).\
        values.reshape(county_data.shape[1:])
        
    # Deforestation
    frames_counties_defor = pd.merge(
        counties_defor,
        frames_county[["frame_id", "county_id"]],
        on="county_id",
        how="right"
    )
    frames_counties_defor["quarter_date"] = pd.to_datetime(frames_counties_defor["quarter_date"])

    counties_time_grid = np.zeros((len(config.TIME_STEPS), frames_idx["x"].max() - frames_idx["x"].min() + 1, frames_idx["y"].max() - frames_idx["y"].min() + 1))
    for t, dt in tqdm(enumerate(config.TIME_STEPS)):
        defor_area = (
            frames_counties_defor[
                frames_counties_defor["quarter_date"] == dt
            ].set_index("frame_id")["area"] +\
            pd.Series(0, index=frames_idx.index)
        ).fillna(0).sort_index()
        counties_time_grid[t, :, :] = defor_area.values.reshape(counties_time_grid[0, :, :].shape)

    # Precipitations
    # create limits history grid
    precip_time_grid = np.zeros((len(config.TIME_STEPS), frames_idx["x"].max() - frames_idx["x"].min() + 1, frames_idx["y"].max() - frames_idx["y"].min() + 1))
    for t, dt in tqdm(enumerate(config.TIME_STEPS)):
        precip_sum = (
            precip[
                precip["quarter_date"] == dt
            ].set_index("frame_id")["precipitation"] +\
            pd.Series(0, index=frames_idx.index)
        ).fillna(0).sort_index()
        precip_time_grid[t, :, :] = precip_sum.values.reshape(counties_time_grid[0, :, :].shape)

    # Terrain
    cols = ["mean", "min", "max", "std"]
    tpi_array = np.zeros((len(cols), frames_idx["x"].max() - frames_idx["x"].min() + 1, frames_idx["y"].max() - frames_idx["y"].min() + 1))
    for icol, col in enumerate(cols):
        v = (
            tpi.set_index("frame_id")[col] +\
            pd.Series(0, index=frames_idx.index)
        ).fillna(0).sort_index()
        tpi_array[icol, :, :] = v.values.reshape(tpi_array[0, :, :].shape)

    # Past Scores
    # create history grid for scores
    scores_time_grid = np.zeros((len(config.TIME_STEPS), frames_idx["x"].max() - frames_idx["x"].min() + 1, frames_idx["y"].max() - frames_idx["y"].min() + 1))
    for t, dt in tqdm(enumerate(config.TIME_STEPS)):
        t_scores = (
            past_scores[
                past_scores["variable"] == dt
            ].set_index("frame_id")["value"] +\
            pd.Series(0, index=frames_idx.index)
        ).fillna(0).sort_index()
        scores_time_grid[t, :, :] = t_scores.values.reshape(scores_time_grid[0, :, :].shape)
        
    # Night Lights
    # create history grid for scores
    night_time_grid = np.zeros((2, len(config.TIME_STEPS), frames_idx["x"].max() - frames_idx["x"].min() + 1, frames_idx["y"].max() - frames_idx["y"].min() + 1))
    for t, dt in tqdm(enumerate(config.TIME_STEPS)):
        avg_light = (
            night_light[
                night_light["dt"] == dt
            ].set_index("frame_id")["avg_light"] +\
            pd.Series(0, index=frames_idx.index)
        ).fillna(0).sort_index()
        night_time_grid[0, t, :, :] = avg_light.values.reshape(night_time_grid[0, 0, :, :].shape)
        
        max_light = (
            night_light[
                night_light["dt"] == dt
            ].set_index("frame_id")["max_light"] +\
            pd.Series(0, index=frames_idx.index)
        ).fillna(0).sort_index()
        night_time_grid[1, t, :, :] = max_light.values.reshape(night_time_grid[0, 0, :, :].shape)

    print(time_grid.shape, county_data.shape, night_time_grid.shape)
    return time_grid, county_data, counties_time_grid, precip_time_grid, tpi_array, scores_time_grid, night_time_grid

def compute_frame_patches(frames_idx, deforestation, out_condition='both'):
    # Gather all the grids and creates input patches ? Also filters deforestation patches
    #! OBS: Alterate only the indexes, not the grids.
    # out_condition = "both"  # deforestation | borders | both
    bundle_step = 32
    patches = []
    for ix in tqdm(list(range(frames_idx["x"].min(), frames_idx["x"].max()+1, bundle_step))):
        fx = ix + config.INPUT_BOXES_SIZE
        for iy in range(frames_idx["y"].min(), frames_idx["y"].max()+1, bundle_step):
            fy = iy + config.INPUT_BOXES_SIZE

            iframes = frames_idx[
                (frames_idx["x"] >= ix) & 
                (frames_idx["x"] < fx) &
                (frames_idx["y"] >= iy) &
                (frames_idx["y"] < fy)
            ]
            
            if out_condition == "borders":
                if iframes["in_borders"].mean() >= 0.5:  # condition: bundle has to be at least half inside borders
                    patches.append(iframes.index)
                    
            elif out_condition == "deforestation":
                out_of_borders_frames = len(set(iframes.index) - set(deforestation["frame_id"].values))
                if out_of_borders_frames < len(iframes):  # condition: bundle has to contain some deforestation
                    patches.append(iframes.index) 

            elif out_condition == "both":
                out_of_borders_frames = len(set(iframes.index) - set(deforestation["frame_id"].values))
                if (out_of_borders_frames < len(iframes)) and (iframes["in_borders"].mean() >= 0.5):
                    patches.append(iframes.index)
            
            elif out_condition == "None":
                patches.append(iframes.index)
    return patches

def apply_normalization(county_data, counties_time_grid, train_time_idx, precip_time_grid, tpi_array, night_time_grid, norm_path):
    population_norm = np.load(norm_path / 'population.npy')
    counties_norm = np.load(norm_path / 'counties.npy')
    precip_norm = np.load(norm_path / 'precip.npy')
    tpi_norm = np.load(norm_path / 'tpi.npy')
    night_norm = np.load(norm_path / 'night.npy')
    
    # print(population_norm.shape, counties.shape, )
    
    # Data normalization
    # Normalizes population data
    pop_median = population_norm[0]
    pop_std = population_norm[1]
    den_median = population_norm[2]
    den_std = population_norm[3]
    norm_pop = (county_data[0, :, :] - pop_median) / pop_std
    norm_den = (county_data[1, :, :] - den_median) / den_std

    county_data[0, :, :] = norm_pop
    county_data[1, :, :] = norm_den

    # Normalizies counties and precipitation grids
    counties_tg_mean = counties_norm[0]
    counties_tg_std = counties_norm[1]
    
    precip_tg_mean = precip_norm[0]
    precip_tg_std = precip_norm[1]
    
    counties_time_grid = (counties_time_grid-counties_tg_mean) / counties_tg_std
    precip_time_grid = (precip_time_grid-precip_tg_mean) / precip_tg_std

    # Normalizes tpi
    for i in range(tpi_array.shape[0]):
        tpi_mean = tpi_norm[0][i, :, :]
        tpi_std = tpi_norm[1][i, :, :]
        tpi_array[i, :, :] = (tpi_array[i, :, :] - tpi_mean) / tpi_std

    # Normalizes Night Images
    for i in [0, 1]:
        night_tg_mean = night_norm[0][i, train_time_idx, :, :]
        night_tg_std = night_norm[1][i, train_time_idx, :, :]
        s = ((night_time_grid[i, :, :, :] - night_tg_mean) / night_tg_std)
        s[np.where(s > 3)] = 3
        night_time_grid[i, :, :, :] = s.copy()

    return county_data, counties_time_grid, precip_time_grid, tpi_array, night_time_grid

def normalize(county_data, counties_time_grid, train_time_idx, precip_time_grid, tpi_array, night_time_grid, save_path):
    # Data normalization
    # Normalizes population data
    
    pop_median = np.median(county_data[0, :, :])
    pop_std = 1e5
    den_median = np.median(county_data[1, :, :])
    den_std = 30
    norm_pop = (county_data[0, :, :] - pop_median) / pop_std
    norm_den = (county_data[1, :, :] - den_median) / den_std

    county_data[0, :, :] = norm_pop
    county_data[1, :, :] = norm_den

    # Normalizies counties and precipitation grids
    counties_tg_mean = counties_time_grid[train_time_idx, :, :].mean()
    counties_tg_std = counties_time_grid[train_time_idx, :, :].std()
    precip_tg_mean = precip_time_grid[train_time_idx, :, :].mean()
    precip_tg_std = precip_time_grid[train_time_idx, :, :].std()
    counties_time_grid = (counties_time_grid-counties_tg_mean) / counties_tg_std
    precip_time_grid = (precip_time_grid-precip_tg_mean) / precip_tg_std

    # Normalizes tpi ?
    tpi_means = []
    tpi_stds = []
    for i in range(tpi_array.shape[0]):
        tpi_mean = tpi_array[i, :, :].mean()
        tpi_std = tpi_array[i, :, :].std()
        tpi_array[i, :, :] = (tpi_array[i, :, :] - tpi_mean) / tpi_std
        tpi_means.append(tpi_mean)
        tpi_stds.append(tpi_std)
    
    tpi_means = np.stack(tpi_means, axis=0)
    tpi_stds = np.stack(tpi_stds, axis=0)
    
    # Normalizes Night Images
    night_means = []
    night_stds = []
    for i in [0, 1]:
        night_tg_mean = night_time_grid[i, train_time_idx, :, :].mean()
        night_tg_std = night_time_grid[i, train_time_idx, :, :].std()
        s = ((night_time_grid[i, :, :, :] - night_tg_mean) / night_tg_std)
        s[np.where(s > 3)] = 3
        night_time_grid[i, :, :, :] = s.copy()
        night_means.append(night_tg_mean)
        night_stds.append(night_tg_std)
    night_means = np.stack(night_means, axis=0)
    night_stds = np.stack(night_stds, axis=0)
    
    # np.save(save_path / 'population.npy', [pop_median, pop_std, den_median, den_std])
    # np.save(save_path / 'counties.npy', [counties_tg_mean, counties_tg_std])
    # np.save(save_path / 'precip.npy', [precip_tg_mean, precip_tg_std])
    # np.save(save_path / 'tpi.npy', [tpi_means, tpi_stds])
    # np.save(save_path / 'night.npy', [night_means, night_stds])
    return county_data, counties_time_grid, precip_time_grid, tpi_array, night_time_grid

def prep4dataset(config):
    am_bounds, frames_idx, deforestation, frames_county, counties_defor, precip, tpi, past_scores, night_light = load_data(config)
    time_grid, county_data, counties_time_grid, precip_time_grid, tpi_array, scores_time_grid, night_time_grid = create_grids(am_bounds, frames_idx, deforestation, frames_county, counties_defor, precip, tpi, past_scores, night_light)
    #! Normalizing DETER warnings by mean and std 
    for i in range(time_grid.shape[0]):
        time_grid_mean = time_grid[i, :, :].mean()
        time_grid_std = time_grid[i, :, :].std()
        time_grid[i, :, :] = (time_grid[i, :, :] - time_grid_mean) / time_grid_std
    train_time_idx = range(12)
    val_time_idx = range(12, 20)
    # test_time_idx = range(20,28) #! Not used yet
    train_data = time_grid[train_time_idx, :, :]
    val_data = time_grid[val_time_idx, :, :]
    save_path = Path('/home/thiago/AmazonDeforestation_Prediction/OpenSTL/data/data_Features/data/trusted')
    county_data, counties_time_grid, precip_time_grid, tpi_array, night_time_grid = normalize(county_data, counties_time_grid, train_time_idx, precip_time_grid, tpi_array, night_time_grid, save_path)
    patches = compute_frame_patches(frames_idx, deforestation, out_condition='both')
    # remove patches that represent reduced regions
    patches = [b for b in patches if (len(b)==len(patches[0]))]
    patches_sample_train = patches
    patches_sample_val = patches
    return train_data, val_data, patches_sample_train, patches_sample_val, frames_idx, county_data, counties_time_grid, \
        precip_time_grid, tpi_array, scores_time_grid, night_time_grid

def prep4dataset_test(config):
    am_bounds, frames_idx, deforestation, frames_county, counties_defor, precip, tpi, past_scores, night_light = load_data(config)
    time_grid, county_data, counties_time_grid, precip_time_grid, tpi_array, scores_time_grid, night_time_grid = create_grids(am_bounds, frames_idx, deforestation, frames_county, counties_defor, precip, tpi, past_scores, night_light)
    #! Normalizing DETER warnings by mean and std 
    for i in range(time_grid.shape[0]):
        time_grid_mean = time_grid[i, :, :].mean()
        time_grid_std = time_grid[i, :, :].std()
        time_grid[i, :, :] = (time_grid[i, :, :] - time_grid_mean) / time_grid_std
    train_time_idx = range(12)
    test_time_idx = range(20,28)
    test_data = time_grid[test_time_idx, :, :]
    norm_path = Path('/home/thiago/AmazonDeforestation_Prediction/OpenSTL/data/data_Features/data/trusted')
    # county_data, counties_time_grid, precip_time_grid, tpi_array, night_time_grid = apply_normalization(county_data, counties_time_grid, test_time_idx, precip_time_grid, tpi_array, night_time_grid, norm_path)
    county_data, counties_time_grid, precip_time_grid, tpi_array, night_time_grid = normalize(county_data, counties_time_grid, train_time_idx, precip_time_grid, tpi_array, night_time_grid, norm_path)
    patches = compute_frame_patches(frames_idx, deforestation, out_condition='None')
    # remove patches that represent reduced regions
    patches = [b for b in patches if (len(b)==len(patches[0]))]
    patches_sample_train = patches
    patches_sample_val = patches
    return test_data, patches_sample_train, patches_sample_val, frames_idx, county_data, counties_time_grid, \
        precip_time_grid, tpi_array, scores_time_grid, night_time_grid
        
def save_gif(save_path, img_frames):
  print("Saving GIF file")
  # with imageio.get_writer(gif_file, mode="I", fps = 1) as writer:
  with imageio.get_writer(save_path, mode="I", duration=1) as writer:
      for idx, frame in enumerate(img_frames):
          print("Adding frame to GIF file: ", idx + 1)
          writer.append_data(frame*255)

from osgeo import gdal
import cv2

def load_tif_image(tif_path):
    gdal_header = gdal.Open(str(tif_path))
    return gdal_header.ReadAsArray()

def apply_legal_amazon_mask(input_image: np.array, amazon_mask: np.array, assign_value: float=2.0):
    ''' Apply Legal Amazon mask '''
    print('DEBUG apply_legal_amazon_mask')
    for i in range(input_image.shape[0]):
        # print(input_image.max(), input_image.min())
        input_image[i, :, :][amazon_mask == 2.0] = assign_value
        # print(input_image.max(), input_image.min())
    return input_image

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
def plot_videos_side_by_side2(video1, video2, output_gif_path):
    # Determine the number of frames in the videos
    num_frames = min(video1.shape[0], video2.shape[0])

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # Create colorbars for each subplot
    cbar1 = fig.colorbar(ax[0].imshow(video1[0], cmap='viridis'), ax=ax[0])
    cbar2 = fig.colorbar(ax[1].imshow(video2[0], cmap='viridis'), ax=ax[1])

    # Set color bar labels
    cbar1.set_label('Values')
    cbar2.set_label('Values')

    # Function to update the plot for each frame
    def update(frame):
        # Get frames from both videos
        frame1 = video1[frame]
        frame2 = video2[frame]

        # Plot frames side by side
        ax[0].imshow(frame1, cmap='viridis')
        ax[1].imshow(frame2, cmap='viridis')
        
        # Add titles to each subplot
        ax[0].set_title(f'Avg - Frame {frame}')
        ax[1].set_title(f'Max - Frame {frame}')

    # Create animation
    animation = FuncAnimation(fig, update, frames=num_frames, interval=100)

    # Save animation as gif
    animation.save(output_gif_path, writer='imagemagick', fps=1, dpi=300)

    # Display the plot (optional)
    # plt.show()
    
def plot_videos_side_by_side(video1, output_gif_path):
    # Determine the number of frames in the videos
    num_frames = video1.shape[0]

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create colorbars for each subplot
    cbar1 = fig.colorbar(ax.imshow(video1[0, :, :], cmap='viridis'), ax=ax)

    # Set color bar labels
    cbar1.set_label('Values')

    # Function to update the plot for each frame
    def update(frame):
        # Get frames from both videos
        frame1 = video1[frame]

        # Plot frames side by side
        ax.imshow(frame1, cmap='viridis')
        
        # Add titles to each subplot
        ax.set_title(f'Frame {frame}')

    # Create animation
    animation = FuncAnimation(fig, update, frames=num_frames, interval=100)

    # Save animation as gif
    animation.save(output_gif_path, writer='imagemagick', fps=1, dpi=300)

    # Display the plot (optional)
    # plt.show()
    
if __name__=='__main__':
    # set device to GPU
    dev = "cuda:0"

    am_bounds, frames_idx, deforestation, frames_county, counties_defor, precip, tpi, past_scores, night_light = load_data(config)
    
    time_grid, county_data, counties_time_grid, precip_time_grid, tpi_array, scores_time_grid, night_time_grid = create_grids(am_bounds, frames_idx, deforestation, frames_county, counties_defor, precip, tpi, past_scores, night_light)
    
    # remove patches that represent reduced regions
    # patches = [b for b in patches if (len(b)==len(patches[0]))]

    train_time_idx = range(12)
    val_time_idx = range(12, 20)
    test_time_idx = range(20,28)

    train_data = time_grid[train_time_idx, :, :]
    val_data = time_grid[val_time_idx, :, :]

    save_path = Path('/home/thiago/AmazonDeforestation_Prediction/OpenSTL/data/data_Features/data/trusted')
    county_data, counties_time_grid, precip_time_grid, tpi_array, night_time_grid = normalize(county_data, counties_time_grid, train_time_idx, precip_time_grid, tpi_array, night_time_grid, save_path)
    
    print(time_grid.shape, county_data.shape, counties_time_grid.shape, precip_time_grid.shape, tpi_array.shape, scores_time_grid.shape, night_time_grid.shape)
    mask_path = '/home/thiago/AmazonDeforestation_Prediction/AmazonData/Dataset_Felipe/area.tif'
    mask = load_tif_image(mask_path)
    print(mask.shape)
    mask[mask == 0.0] = 2.0
    mask[mask == 1] = 0.0
    
    mask2 = mask.copy()
    # mask2[mask == 0.0] = -1
    # mask2[mask == 1] = 0.0
    
    input_image = night_time_grid
    print(input_image.shape)
    mask2 = cv2.resize(mask2, (time_grid.shape[2], time_grid.shape[1]), cv2.INTER_AREA)
    # cv2.imwrite('/home/thiago/AmazonDeforestation_Prediction/OpenSTL/data/data_Features/data/trusted/mask.jpg', mask2)
    
    input_image[0] = apply_legal_amazon_mask(input_image[0], mask2, assign_value=-1.0)
    input_image[1] = apply_legal_amazon_mask(input_image[1], mask2, assign_value=-1.0)
    # input_image[1] = apply_legal_amazon_mask(np.expand_dims(input_image[1], axis=0), mask2, assign_value=85)[0]
    # input_image = input_image / 1e6
    save_path = Path('/home/thiago/AmazonDeforestation_Prediction/OpenSTL/data/data_Features/data/trusted/train_img.gif')
    # save_gif(save_path, time_grid)
    # plot_videos_side_by_side(input_image, save_path)
    plot_videos_side_by_side2(input_image[0], input_image[1], save_path)
    
    # # Create a figure with two subplots
    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # # Plot the first image on the left subplot
    # im1 = axes[0].imshow(input_image[0], cmap='viridis')
    # axes[0].set_title('Population')

    # inp = input_image[1]
    # # inp[inp == 0] = -1.
    # # inp = inp / 1e6
    # # inp[inp == 0] = -1.0
    # # print(inp.mean(axis=(0, 1)))
    # # Plot the second image on the right subplot
    # im2 = axes[1].imshow(inp, cmap='plasma')
    # axes[1].set_title('Population Density')

    # # Add color bars to both subplots
    # cbar1 = fig.colorbar(im1, ax=axes[0])
    # cbar2 = fig.colorbar(im2, ax=axes[1])

    # # Adjust layout to prevent clipping of color bars
    # plt.tight_layout()

    # # Save the figure with a resolution of 300 dpi
    # plt.savefig('/home/thiago/AmazonDeforestation_Prediction/OpenSTL/data/data_Features/data/trusted/population.png', dpi=300)

    # # Show the figure (optional)
    # plt.show()
    patches = compute_frame_patches(frames_idx, deforestation, out_condition='both')

    # remove patches that represent reduced regions
    patches = [b for b in patches if (len(b)==len(patches[0]))]

    print(train_data.shape, val_data.shape)
    print(len(patches))

    patches_sample_train = patches
    patches_sample_val = patches

    train_set = GiovanniDataset(
        train_data, 
        patches_sample_train, 
        frames_idx, 
        county_data,
        counties_time_grid,
        precip_time_grid,
        tpi_array,
        None,
        scores_time_grid,
        night_time_grid,
        device=dev
    )

    val_set = GiovanniDataset(
        val_data, #previously = test_data
        patches_sample_val, 
        frames_idx, 
        county_data,
        counties_time_grid,
        precip_time_grid,
        tpi_array,
        None,
        scores_time_grid,
        night_time_grid,
        device=dev
    )

    trainloader = torch.utils.data.DataLoader(train_set)
    valloader = torch.utils.data.DataLoader(val_set)

    for inputs, labels in trainloader:
        print(inputs.shape, labels.shape)
        print(labels[0])
        break

    # from segmentation_models_pytorch.losses import FocalLoss

    # f_loss = FocalLoss("binary", gamma=3).to(dev)

    # # baseline: all zero
    # base_train_err = 0
    # for inputs, labels in trainloader:
    #     y_pred = torch.tensor(np.zeros(labels.shape)).to(dev)
    #     y_pred[:, 0, :, :] = 1
    #     print(y_pred.shape, labels.shape)
    #     # base_train_err += ce_loss(input=y_pred, target=labels)
    #     base_train_err += f_loss(y_pred=y_pred, y_true=labels)
    # base_train_err = base_train_err / len(trainloader)

    # print(f"Baseline Error (Train) = {base_train_err:.6f}")