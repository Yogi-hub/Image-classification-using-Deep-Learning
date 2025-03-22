# Image Classification Using Deep Learning

This project demonstrates the implementation of deep learning techniques for classifying handwritten digits from the MNIST dataset. It includes a custom-built Multi-Layer Perceptron (MLP) from scratch and an improved version using PyTorch, incorporating data augmentation and advanced neural network architectures. (**Note:** This project is intended to study the effectiveness/working of MLP for image classification, hence CNN is not used)

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This repository contains a Jupyter Notebook that explores image classification using the MNIST dataset. The project is divided into two main parts:
1. **Basic MLP Implementation**: A custom MLP built from scratch using NumPy, featuring fully-connected layers, activation functions (ReLU and Sigmoid), and mini-batch gradient descent.
2. **Improved MLP with PyTorch**: An enhanced MLP model implemented using PyTorch, with increased hidden layers, data augmentation (Gaussian noise and horizontal flipping), and training using the Adam optimizer and Cross-Entropy Loss.

The goal is to compare the performance of a basic MLP with an improved version and demonstrate how data augmentation and architectural enhancements improve classification accuracy.

## Features
- Loading and preprocessing of the MNIST dataset using Keras and PyTorch.
- Visualization of original and augmented MNIST images.
- Custom implementation of dense layers, activation functions, and gradient descent.
- Training and testing of neural networks with performance metrics (accuracy and loss).
- Data augmentation techniques (Gaussian noise and random horizontal flipping) to improve model robustness.

## Requirements
To run this project, you'll need the following dependencies:
- Python 3.8+
- NumPy
- Matplotlib
- Keras
- PyTorch
- Torchvision
- Torchinfo

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Yogi-hub/Image-classification-using-Deep-Learning.git
   cd Image-classification-using-Deep-Learning
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note*: If a `requirements.txt` file doesn't exist yet, you can create one with the following content:
   ```
   numpy
   matplotlib
   tensorflow
   torch
   torchvision
   torchinfo
   ```

## Usage
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Image_Classification.ipynb
   ```
2. Run the cells sequentially to:
   - Load and preprocess the MNIST dataset.
   - Visualize sample images.
   - Train and test the basic MLP model.
   - Train and test the improved PyTorch-based MLP model with augmented data.

3. Modify hyperparameters (e.g., number of epochs, learning rate, batch size) in the notebook to experiment with different configurations.

## Implementation Details
### 1. Basic MLP
- **Architecture**: Multiple dense layers with configurable activation functions (ReLU or Sigmoid).
- **Training**: Mini-batch gradient descent with Mean Squared Error (MSE) loss.
- **Preprocessing**: Flattening and normalizing MNIST images, one-hot encoding labels.
- **Performance**: Initial testing showed poor accuracy, highlighting limitations of the basic model.

### 2. Improved MLP with PyTorch
- **Architecture**: Increased complexity with 100 hidden units in a single hidden layer (configurable).
- **Data Augmentation**: Gaussian noise (mean=0, std=0.2) and random horizontal flipping (p=0.7).
- **Training**: 30 epochs using Adam optimizer (lr=0.01) and Cross-Entropy Loss.
- **Evaluation**: Accuracy and loss computed on both training and test sets.

### Key Libraries
- **Keras**: For initial MNIST data loading.
- **PyTorch**: For advanced MLP implementation and training.
- **Matplotlib**: For visualizing sample images and augmented data.

## Results
- **Basic MLP**: Poor performance with predictions often stuck on a single class (e.g., predicting "2" repeatedly).
- **Improved MLP**: Achieved a testing accuracy of ~94.95% with a Cross-Entropy Loss of ~0.289, demonstrating significant improvement due to data augmentation and architectural enhancements.

Sample output from the improved model:
```
Epoch 30, Loss (CE): 0.30141  Accuracy = 94.219
Testing accuracy = 94.95  Loss (CE): 0.28901511430740356
```

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add feature"`).
4. Push to your branch (`git push origin feature-branch`).
5. Open a pull request.

Please ensure your code follows good practices and includes comments where necessary.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
