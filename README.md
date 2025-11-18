# Neural-Network-using-Numpy
Neural Network Implementation from Scratch
A Python implementation of a fully-connected neural network built from scratch using NumPy. This implementation includes advanced features like L2 regularization, learning rate decay, and proper weight initialization.
Performance
#Achieved Results: 96.5% Test Accuracy

The implementation demonstrates excellent performance on complex classification tasks:

Handles high-dimensional datasets (20 features, 3 classes)

Robust to noisy and redundant features

Effective learning with proper regularization

Consistent performance across training/validation/test splits

Key Factors for High Accuracy:
Deep Architecture: 4 hidden layers (128-64-32 neurons)

Optimal Regularization: Balanced L2 regularization (λ=0.0001)

Extended Training: 1500 epochs with learning rate decay

Proper Scaling: StandardScaler for feature normalization
Features
Multi-layer architecture with configurable hidden layers

ReLU activation for hidden layers with He initialization

Softmax activation for multi-class classification output

L2 regularization to prevent overfitting

Learning rate decay for better convergence

Batch gradient descent optimization

Comprehensive metrics including training/validation accuracy tracking

Architecture
The neural network supports flexible architecture configuration:

Input layer size determined by feature dimensions

Multiple hidden layers with ReLU activation

Output layer with Softmax activation for classification

He initialization for better ReLU performance

