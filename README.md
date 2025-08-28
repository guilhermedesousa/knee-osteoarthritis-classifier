# Knee Osteoarthritis Classifier

This project implements a deep learning pipeline for classifying knee osteoarthritis (KOA) severity using convolutional neural networks (CNNs) and vision transformers (ViTs). It includes training, evaluation, and visualization tools such as Grad-CAM and AUC-ROC curves.

## Features

- **Model Training**: Supports multiple architectures (e.g., ResNet, DenseNet, Vision Transformers).
- **Loss Functions**: Implements both Cross-Entropy and CORN loss.
- **Early Stopping**: Stops training when validation loss stops improving.
- **Conformal Prediction**: Provides prediction intervals for uncertainty estimation.
- **Grad-CAM Visualization**: Generates heatmaps to interpret model predictions.
- **AUC-ROC Curve**: Evaluates model performance with ROC curves.
- **Hyperparameter Optimization**: Uses Optuna for automated hyperparameter tuning.

## Project Structure

- `koa_cnns_training.ipynb`: Main notebook for training, evaluation, and visualization.
- `pgc_preprocess_images.ipynb`: Preprocessing pipeline for x-ray images.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- transformers
- coral-pytorch
- pytorch-grad-cam
- optuna
- ptflops
- scikit-learn
- matplotlib
- seaborn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/knee-osteoarthritis-classifier.git
   cd knee-osteoarthritis-classifier