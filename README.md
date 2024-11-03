# Structural Dissimilarity Compensation Graph Autoencoder (SDCGA) for Multimodal Change Detection

This repository contains the implementation of the **Structural Dissimilarity Compensation Graph Autoencoder (SDCGA)** for multimodal change detection, specifically designed for detecting land-cover changes using heterogeneous data such as optical and SAR images.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Parameters](#parameters)
- [Results](#results)
- [Example](#example)
- [License](#license)
- [References](#references)

## Introduction

Multimodal change detection involves identifying changes between images captured at different times and using different sensors (e.g., optical and SAR). The SDCGA model utilizes a graph autoencoder with attention mechanisms to compensate for structural dissimilarities between modalities, improving change detection accuracy.

## Features

- **Graph-based Modeling**: Constructs graphs from superpixel segmentation to capture spatial relationships.
- **Attention Mechanisms**: Employs Graph Attention Networks (GAT) for better feature representation.
- **Structural Compensation**: Compensates for structural differences between modalities using learned parameters.
- **Flexible Input**: Supports multimodal data such as optical and SAR images.
- **Performance Metrics**: Calculates metrics like Overall Accuracy, Kappa coefficient, and F1 score.

## Requirements

- Python 3.6 or higher
- PyTorch 1.7 or higher
- Torch Geometric
- NumPy
- SciPy
- scikit-learn
- scikit-image
- matplotlib
- imageio
- OpenCV-Python

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/SDCGA.git
    cd SDCGA
    ```

2. **Set up a virtual environment (optional but recommended)**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:

    ```bash
    pip install -r requirements.txt
    ```

    **Note**: For `torch-geometric`, you may need to follow the [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), as it depends on your CUDA version and PyTorch installation.

    Example installation for `torch-geometric`:

    ```bash
    pip install torch
    pip install torch-geometric
    ```

    Make sure your PyTorch and `torch-geometric` versions are compatible.

## Usage

1. **Prepare your data**:

    - **First-time image (`image_t1`)**: e.g., SAR image.
    - **Second-time image (`image_t2`)**: e.g., optical image.
    - **Reference ground truth (`Ref_gt`)**: Ground truth change map for evaluation.

    Ensure that:

    - Images are in the same spatial resolution and aligned.
    - Images are in supported formats (e.g., TIFF).

2. **Modify the script or provide command-line arguments**:

    - Update the paths to your images in the script or pass them as arguments.

3. **Run the script**:

    ```bash
    python sdcga.py --image_t1_path path/to/your/image_t1.tif \
                    --image_t2_path path/to/your/image_t2.tif \
                    --ref_gt_path path/to/your/ref_gt.tif
    ```

4. **Adjust parameters (optional)**:

    You can modify parameters such as learning rate, number of epochs, etc., via command-line arguments. See the [Parameters](#parameters) section for details.

## Parameters

The script accepts several command-line arguments to adjust the model and processing:

- `--lr`: Learning rate (default: `0.01`)
- `--weight_decay`: Weight decay (default: `0.0001`)
- `--dropout`: Dropout rate (default: `0.0`)
- `--num_heads`: Number of attention heads in GATConv (default: `4`)
- `--hidden_dim`: Dimension of hidden layers (default: `16`)
- `--n_seg`: Approximate number of superpixels (default: `10000`)
- `--cmp`: Compactness parameter for superpixel segmentation (default: `15`)
- `--beta`: Penalty coefficient for delta (default: `1`)
- `--regularization`: Regularization term coefficient (default: `0.00001`)
- `--k_ratio`: Neighbor ratio for graph construction (default: `0.1`)
- `--epoch`: Number of training epochs (default: `300`)

Example of adjusting parameters:

```bash
python sdcga.py --lr 0.005 --epoch 500 --num_heads 8
