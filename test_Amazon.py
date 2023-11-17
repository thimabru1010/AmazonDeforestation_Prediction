
import torch
from AmazonDataset import CustomDataset_Test
from pathlib import Path
import numpy as np
from openstl.api import BaseExperiment
from openstl.utils import create_parser, default_parser
import json
import imageio
from osgeo import gdal
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def load_tif_image(tif_path):
    gdal_header = gdal.Open(str(tif_path))
    return gdal_header.ReadAsArray()

def apply_legal_amazon_mask(input_image: np.array, amazon_mask: np.array):
    ''' Apply Legal Amazon mask '''
    print('DEBUG apply_legal_amazon_mask')
    for i in range(input_image.shape[0]):
        # print(input_image.max(), input_image.min())
        input_image[i, :, :][amazon_mask == 2.0] = 2
        # print(input_image.max(), input_image.min())
    return input_image
  
def save_gif(save_path, img_frames):
  print("Saving GIF file")
  # with imageio.get_writer(gif_file, mode="I", fps = 1) as writer:
  with imageio.get_writer(save_path, mode="I", duration=1) as writer:
      for idx, frame in enumerate(img_frames):
          print("Adding frame to GIF file: ", idx + 1)
          writer.append_data(frame*255)
          
def plot_videos_side_by_side(video1, video2, video3, output_gif_path):
    # Determine the number of frames in the videos
    num_frames = min(video1.shape[0], video2.shape[0], video3.shape[0])

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # Create colorbars for each subplot
    cbar1 = fig.colorbar(ax[0].imshow(video1[0], cmap='viridis'), ax=ax[0])
    cbar2 = fig.colorbar(ax[1].imshow(video2[0], cmap='viridis'), ax=ax[1])
    cbar3 = fig.colorbar(ax[2].imshow(video3[0], cmap='viridis'), ax=ax[2])

    # Set color bar labels
    cbar1.set_label('Probabilities')
    cbar2.set_label('Probabilities')
    cbar3.set_label('Class Argmax')

    # Function to update the plot for each frame
    def update(frame):
        # Get frames from both videos
        frame1 = video1[frame]
        frame2 = video2[frame]
        frame3 = video3[frame]

        # Plot frames side by side
        ax[0].imshow(frame1, cmap='viridis')
        ax[1].imshow(frame2, cmap='viridis')
        ax[2].imshow(frame3, cmap='viridis')
        
        # Add titles to each subplot
        ax[0].set_title(f'Probs Class 0 (No Def) - Frame {frame}')
        ax[1].set_title(f'Probs Class 1 (Def) - Frame {frame}')
        ax[2].set_title(f'Class Argmax - Frame {frame}')

    # Create animation
    animation = FuncAnimation(fig, update, frames=num_frames, interval=100)

    # Save animation as gif
    animation.save(output_gif_path, writer='imagemagick', fps=1, dpi=300)

    # Display the plot (optional)
    # plt.show()


EX_NAME = 'custom_exp23'
# root_dir = Path('/home/thiago/AmazonDeforestation_Prediction/OpenSTL/data/Dataset/DETR_Patches')
img_path = Path('/home/thiago/AmazonDeforestation_Prediction/AmazonData/Dataset_Felipe/test.tif')
exp_path = Path(f'/home/thiago/AmazonDeforestation_Prediction/OpenSTL/work_dirs/{EX_NAME}')

batch_size = 32
Debug = False
val_fill = 2


custom_training_config = {
    'pre_seq_length': 4,
    'aft_seq_length': 1,
    'total_length': 5,
    'batch_size': batch_size,
    'val_batch_size': batch_size,
    'epoch': 50,
    'lr': 1e-4,   
    'metrics': ['acc', 'Recall', 'Precision', 'f1_score'],

    'ex_name': f'{EX_NAME}', # custom_exp
    'dataname': 'custom',
    'in_shape': [4, 1, 64, 64], # T, C, H, W = self.args.in_shape
    'loss_weights': None,
    'early_stop_epoch': 10,
    'warmup_epoch': 0, #default = 0
    'sched': 'step',
    'decay_epoch': 3,
    'decay_rate': 0.5,
    'resume_from': None,
    'auto_resume': False,
    'test_time': True,
    'loss': 'focal'
}

custom_model_config = {
    # For MetaVP models, the most important hyperparameters are: 
    # N_S, N_T, hid_S, hid_T, model_type
    'method': 'SimVP',
    # Users can either using a config file or directly set these hyperparameters 
    # 'config_file': 'configs/custom/example_model.py',
    
    # Here, we directly set these parameters
    'model_type': 'gSTA',
    'N_S': 4,
    'N_T': 8,
    'hid_S': 64,
    'hid_T': 256
}
  
test_set = CustomDataset_Test(img_path=img_path, Debug=Debug, val_fill=val_fill)

print(len(test_set))

dataloader_test = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False, pin_memory=True)

args = create_parser().parse_args([])
config = args.__dict__

# update the training config
config.update(custom_training_config)
# update the model config
config.update(custom_model_config)
# fulfill with default values
default_values = default_parser()
for attribute in default_values.keys():
    if config[attribute] is None:
        config[attribute] = default_values[attribute]

exp = BaseExperiment(args, dataloaders=(dataloader_test, None, dataloader_test), nclasses=2)

print('>'*35 + ' testing  ' + '<'*35)
exp.test(classify=True)

#! Reconstruct predictions

# work_dir_path = '/home/thiago/AmazonDeforestation_Prediction/OpenSTL/work_dirs/custom_exp4/saved/preds.npy'
# work_dir_path = Path('/home/thiago/AmazonDeforestation_Prediction/OpenSTL/work_dirs/custom_exp4/saved')

mask_path = '/home/thiago/AmazonDeforestation_Prediction/AmazonData/Dataset_Felipe/area.tif'
mask = load_tif_image(mask_path)
print(mask.shape)
mask[mask == 0.0] = 2.0
mask[mask == 1] = 0.0


# preds = np.argmax(np.load(exp_path / 'saved' / 'preds.npy'), axis=2)
logits = np.load(exp_path / 'saved' / 'preds.npy')
print(logits.shape)
preds = F.softmax(torch.Tensor(logits), dim=2).numpy()
print('DEBUG Probs vs Logits')
print(logits.max(), logits.min())
print(preds.max(), preds.min())
# trues = np.load(exp_path / 'saved' / 'trues.npy')

# print(preds)
print(preds.shape)
# print(trues.shape)
# original_shape = (11, 2368, 3008)
window_size = 5

preds0 = preds[:, :, 0]
preds1 = preds[:, :, 1]
preds_argmax = np.argmax(preds, axis=2)
th = 0.7
preds_argmax = preds1.copy()
preds_argmax[preds1 >= th] = 1
preds_argmax[preds1 < th] = 0

# Plot the histograms side by side
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(preds0.reshape(-1), bins=10, range=(0, 1), edgecolor='black', alpha=0.7)
plt.title('No def Probs')
# plt.xlabel('Value')
# plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(preds1.reshape(-1), bins=10, range=(0, 1), edgecolor='black', alpha=0.7)
plt.title('Def Probs')
# plt.xlabel('Value')
# plt.ylabel('Frequency')

plt.tight_layout()
# Save the figure with 300 DPI
plt.savefig(exp_path / 'saved' / 'classes_probs_hist.jpeg', dpi=300)
# plt.show()

img_recs = []
for i, _pred in enumerate([preds0, preds1, preds_argmax]):
  img_rec = test_set.patches_reconstruction(_pred.reshape(-1, (12 - val_fill) - window_size + 1, 1, 64, 64))
  print(img_rec.shape)

  # Preds proportion:
  class_0 = np.sum(img_rec == 0)
  class_1 = np.sum(img_rec == 1)

  print(f"Class 0: {class_0/(class_0 + class_1)} - Class 1: {class_1/(class_0 + class_1)}")

  img_rec = apply_legal_amazon_mask(img_rec, mask)
  print(img_rec.max(), img_rec.min())
  img_recs.append(img_rec)

  class_0 = np.sum(img_rec == 0)
  class_1 = np.sum(img_rec == 1)
  print(f"Class 0: {class_0/(class_0 + class_1)} - Class 1: {class_1/(class_0 + class_1)}")

  if i == 2:
    save_path = exp_path / 'saved' / 'preds_rec.gif'
    save_gif(save_path, img_rec)
    
  
plot_videos_side_by_side(img_recs[0], img_recs[1], img_recs[2], exp_path / 'saved' / 'preds_probs.gif')