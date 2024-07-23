# Diffusion Model for Image Generation

This project implements a Diffusion Model(DDPM) for image generation using PyTorch. The model is trained on the Fashion MNIST dataset and can generate new fashion item images.

This implementation of DDPM is originally from [`Huggingface`](https://huggingface.co/blog/annotated-diffusion)

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Sampling Process](#sampling-process)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

Diffusion models are a class of generative models that learn to gradually denoise a completely noisy image. This project implements such a model using a U-Net architecture with attention mechanisms. The model is trained on the Fashion MNIST dataset and can generate new fashion item images.

## Requirements

The project requires the following main libraries:

- Python 3.7+
- PyTorch 1.7+
- torchvision
- einops
- tqdm
- matplotlib
- datasets

For a complete list of requirements, see the `requirements.txt` file.

## Project Structure


```sh
diffusion_model/
│
├── main.py
├── requirements.txt
│
├── src/
│   ├── init.py
│   ├── model.py
│   ├── dataset.py
│   ├── diffusion.py
│   ├── train.py
│   ├── sample.py
│   └── utils.py
│
└── results/
```


- `main.py`: The entry point of the program.
- `src/model.py`: Contains the U-Net model architecture.
- `src/dataset.py`: Handles dataset loading and preprocessing.
- `src/diffusion.py`: Implements the diffusion process.
- `src/train.py`: Contains the training loop.
- `src/sample.py`: Implements the sampling process.
- `src/utils.py`: Contains utility functions.
- `results/`: Directory where generated images are saved.

## Installation

1. Clone this repository:

```sh

git clone https://github.com/yourusername/diffusion-model.git
cd diffusion-model

```

2. Install the required packages:

```sh
   pip install -r requirements.txt
```


## Usage

To train the model and generate images:

```sh
python main.py
```


This will start the training process and periodically save generated images in the `results/` directory.

## Model Architecture

The model uses a U-Net architecture with the following key components:

- Residual blocks
- Group normalization
- Self-attention mechanisms
- Sinusoidal position embeddings for time steps

The U-Net consists of a series of downsampling layers followed by upsampling layers, with skip connections between corresponding layers.

## Training Process

The training process follows these steps:

1. Load and preprocess the Fashion MNIST dataset.
2. For each epoch and batch:
   - Sample a random timestep t.
   - Add noise to the input images according to t.
   - Predict the noise using the model.
   - Calculate the loss between predicted and actual noise.
   - Update the model parameters.
3. Periodically save generated samples.

## Sampling Process

The sampling process to generate new images involves:

1. Start with pure noise.
2. Iteratively denoise the image using the trained model.
3. For each timestep from T to 1:
   - Predict the noise in the current noisy image.
   - Remove a portion of the predicted noise.
   - Add a small amount of random noise (except at the final step).

## Results

After training, the model can generate new fashion item images. Examples of generated images can be found in the `results/` directory.

## Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



















