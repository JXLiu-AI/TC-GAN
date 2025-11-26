# Infrared and Visible Image Fusion via Texture Conditional Generative Adversarial Network (TC-GAN)

This repository contains the PyTorch implementation of **TC-GAN**, as described in the paper "Infrared and Visible Image Fusion via Texture Conditional Generative Adversarial Network".

## Overview

- `train.py`: Main training script for the TC-GAN model. It handles the training loop for both the generator and discriminator, utilizing a custom `GeneratorLoss` and logging metrics via `tensorboardX`.
- `model.py`: Defines the architecture for the Generator and Discriminator networks.
- `loss.py`: Implements the loss functions, including perceptual loss and Total Variation (TV) loss.
- `data_utils.py`: Contains dataset loaders (`TrainDatasetFromFolder`, `ValDatasetFromFolder`) and image transformation utilities.

## Dependencies

- Python 3.6+
- PyTorch
- torchvision
- tensorboardX
- tqdm

To install the required dependencies, run:

```bash
pip install torch torchvision tensorboardX tqdm
```

## Usage

### Training

To train the model, use the following command:

```bash
python3 train.py --crop_size 88 --upscale_factor 1 --num_epochs 1000
```

**Arguments:**
- `--crop_size`: Size of the crop for training images (default: 88).
- `--upscale_factor`: Upscale factor for super-resolution (choices: 1, 2, 4).
- `--num_epochs`: Total number of training epochs (default: 1000).

## Configuration & Notes

- **Dataset Paths**: The dataset paths in `train.py` are currently hardcoded (e.g., `'E:/Dataset/'`). Please update these paths to point to your local dataset directory.
- **Output Directory**: Generated images during training are saved to the path specified by `out_path` in `train.py` (default: `/data/ResnetFusion/training_results/SRF_<UPSCALE>/`). Ensure this directory exists or update the path.
- **Model Checkpoints**: Model weights are saved to paths like `E:/Project/NewProject/ResNetFusion-master/model1/`. You should update these paths to a valid location on your system.
- **Logging**: Training progress and metrics are logged using `tensorboardX`. To visualize the logs:

```bash
tensorboard --logdir runs
```

## GPU Support

The script automatically detects if CUDA is available and moves the model and tensors to the GPU. Ensure you have a CUDA-compatible PyTorch version installed.

## Tips

- You can adjust the `batch_size` in the `DataLoader` initialization within `train.py`.
- For better flexibility, consider modifying `train.py` to accept dataset and output paths as command-line arguments.
 
## TC-GAN CODE

Before running, please update the file paths in `image_test.py` and `MDM.py` to match your local environment (e.g., dataset directory, model checkpoint path in `model/`, and output directory).

**Steps:**

1. Edit the paths in `image_test.py`, then run to generate intermediate images:

   ```bash
   python3 image_test.py
   ```

   This will output intermediate fusion images to `final_resutls/` (or the path specified in the script).

2. Edit the paths in `MDM.py`, then run to generate final results:

   ```bash
   python3 MDM.py
   ```

   This will load the checkpoint from `model/` and output the final fused images to the specified directory.
