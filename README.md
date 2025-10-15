# Deep Learning Projects Collection

A comprehensive collection of deep learning projects and experiments covering various neural network architectures, optimization techniques, and applications.

## Neural Network Fundamentals

### `backpropagation-classification.ipynb`
Implementation of backpropagation algorithm for binary classification problems.
Uses numpy to build a neural network from scratch with custom forward and backward propagation functions.

### `backpropagation-regression.ipynb`
Backpropagation implementation for regression tasks with continuous target variables.
Demonstrates gradient descent optimization and parameter updates in regression scenarios.

### `batch-norm-example.ipynb`
Comprehensive example of batch normalization technique in neural networks.
Shows how batch normalization improves training stability and convergence speed.

### `dropout-classification-example.ipynb`
Implementation of dropout regularization for classification problems.
Demonstrates how dropout prevents overfitting by randomly setting neurons to zero during training.

### `dropout-notebook.ipynb`
Detailed exploration of dropout regularization technique.
Includes visualization of how dropout affects model performance and generalization.

### `early-stopping.ipynb`
Implementation of early stopping technique to prevent overfitting.
Shows how to monitor validation loss and stop training when performance plateaus.

### `regularization.ipynb`
Comprehensive guide to various regularization techniques in deep learning.
Covers L1, L2 regularization, dropout, and data augmentation methods.

### `xavier_optimizer.ipynb`
Implementation of Xavier/Glorot initialization for neural network weights.
Demonstrates proper weight initialization techniques for stable training.

### `zero-initialization-sigmoid.ipynb`
Analysis of the problems with zero weight initialization in neural networks.
Shows why zero initialization leads to symmetric problems and poor learning.

## Convolutional Neural Networks (CNNs)

### `CNN_horse_human_cat_dog.ipynb`
Multi-class image classification using CNN for horses, humans, cats, and dogs.
Implements data preprocessing, model architecture, and training pipeline for image recognition.

### `cnn.ipynb`
Basic CNN implementation and architecture exploration.
Covers convolutional layers, pooling, and fully connected layers in image classification.

### `cnn-fashion-mnist-pytorch-gpu.ipynb`
Fashion-MNIST classification using PyTorch with GPU acceleration.
Demonstrates CNN implementation with PyTorch framework and CUDA optimization.

### `fashion_mnist.ipynb`
Fashion-MNIST dataset classification using various neural network architectures.
Explores different approaches to classify clothing items in grayscale images.

### `visualising-cnn.ipynb`
Visualization techniques for understanding CNN feature maps and activations.
Shows how to interpret what different layers learn in convolutional networks.

### `keras-padding-demo.ipynb`
Demonstration of padding techniques in Keras CNN layers.
Explores valid vs same padding and their effects on output dimensions.

### `keras-pooling-demo.ipynb`
Comprehensive guide to pooling operations in CNNs.
Covers max pooling, average pooling, and global pooling techniques.

### `Data_augmentation_and_CNN.ipynb`
Image data augmentation techniques combined with CNN training.
Shows how data augmentation improves model generalization and performance.

## Recurrent Neural Networks (RNNs)

### `Simple_RNN.ipynb`
Basic Simple RNN implementation and architecture exploration.
Demonstrates vanilla RNN structure and parameter calculations for sequence data.

### `encoding-simplernn.ipynb`
Text encoding and preprocessing for Simple RNN models.
Shows how to convert text data into numerical sequences for RNN training.

### `sentiment-analysis-simplernn.ipynb`
Sentiment analysis using Simple RNN for text classification.
Implements binary sentiment classification on text data using recurrent networks.

### `deep-rnns.ipynb`
Deep RNN architectures with multiple recurrent layers.
Explores stacking RNN layers and handling vanishing gradient problems.

### `lstm-project.ipynb`
Long Short-Term Memory (LSTM) network implementation for sequence modeling.
Demonstrates LSTM architecture for handling long-term dependencies in sequences.

### `TimeSeriesForecasting_LSTMs.ipynb`
Time series forecasting using LSTM networks.
Applies LSTMs to predict future values in temporal data sequences.

### `pytorch-lstm-next-word-predictor.ipynb`
Next word prediction model using PyTorch LSTM implementation.
Builds a language model to predict the next word in a sequence.

## Advanced Architectures

### `BERT.ipynb`
Bidirectional Encoder Representations from Transformers implementation.
Demonstrates pre-trained BERT model usage for various NLP tasks.

### `DCGAN.ipynb`
Deep Convolutional Generative Adversarial Network implementation.
Creates realistic images using GAN architecture with CNN components.

### `GAN_MNISTipynb`
Generative Adversarial Network for MNIST digit generation.
Implements GAN to generate synthetic handwritten digit images.

### `Control_Net.ipynb`
ControlNet architecture for controllable image generation.
Advanced GAN variant that allows fine-grained control over generated content.

## Transfer Learning and Fine-tuning

### `Tranfer_learning_fine tuning.ipynb`
Transfer learning and fine-tuning techniques for pre-trained models.
Shows how to adapt pre-trained models for new tasks with limited data.

### `Transfer_learning_100_daysipynb`
100-day challenge project on transfer learning applications.
Comprehensive exploration of transfer learning across different domains.

### `transfer-learning-optuna.ipynb`
Transfer learning with Optuna hyperparameter optimization.
Combines transfer learning with automated hyperparameter tuning.

### `Custom _Finetuning_Training-Dreambooth.ipynb`
Custom fine-tuning using DreamBooth technique for personalization.
Advanced fine-tuning approach for custom object/person generation.

## Optimization and Hyperparameter Tuning

### `optuna-ann-fashion-mnist-pytorch-gpu-optimized-optuna.ipynb`
Optuna hyperparameter optimization for Fashion-MNIST classification.
Automated hyperparameter search using Optuna framework with GPU acceleration.

### `feature-scaling.ipynb`
Feature scaling techniques for neural network preprocessing.
Covers normalization, standardization, and their impact on training.

## Data Analysis and Engineering

### `Exploratory Data Analysis..ipynb`
Exploratory data analysis techniques for machine learning projects.
Statistical analysis and visualization of datasets before model training.

### `Feature_engineering_Column_transfrom.ipynb`
Feature engineering and column transformation techniques.
Data preprocessing and feature creation for improved model performance.

### `em_algorithm.py`
Expectation-Maximization algorithm implementation in Python.
Unsupervised learning algorithm for finding maximum likelihood estimates.

## Specialized Applications

### `Text-Summarize-Finetuning.ipynb`
Text summarization using fine-tuned transformer models.
Implements abstractive text summarization with pre-trained language models.

### `Decision_tree.ipynb`
Decision tree implementation and visualization.
Classical machine learning algorithm for classification and regression tasks.

## Additional Resources

### `Emoji Prediction`
Emoji prediction model for text input.
Predicts appropriate emojis based on text sentiment and context.

---

## Getting Started

Each notebook is self-contained and includes:
- Dataset loading and preprocessing
- Model architecture implementation
- Training and evaluation code
- Results visualization and analysis

## Dependencies

Most notebooks require:
- TensorFlow/Keras
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Transformers (HuggingFace)

## Usage

1. Clone the repository
2. Install required dependencies
3. Run individual notebooks in Jupyter environment
4. Modify hyperparameters and experiment with different configurations

---

*This collection represents a comprehensive journey through deep learning concepts, from basic neural networks to advanced architectures like transformers and GANs.*
