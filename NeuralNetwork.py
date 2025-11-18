import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score

class ImprovedNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, reg_lambda=0.001):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        
        # He initialization for ReLU
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes)-1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i]))
            self.biases.append(np.zeros((1, layer_sizes[i+1])))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        # Hidden layers with ReLU
        for i in range(len(self.weights)-1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            self.activations.append(self.relu(z))
        
        # Output layer with Softmax
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        self.activations.append(self.softmax(z))
        
        return self.activations[-1]
    
    def compute_loss(self, y, y_hat):
        m = y.shape[0]
        log_likelihood = -np.log(y_hat[range(m), y.argmax(axis=1)])
        data_loss = np.sum(log_likelihood) / m
        
        # L2 regularization
        reg_loss = 0
        for weight in self.weights:
            reg_loss += 0.5 * self.reg_lambda * np.sum(weight * weight)
        
        return data_loss + reg_loss
    
    def backward(self, X, y, output):
        m = X.shape[0]
        deltas = [None] * len(self.weights)
        
        # Output layer delta
        deltas[-1] = (output - y) / m
        
        # Backpropagate through hidden layers
        for i in range(len(self.weights)-2, -1, -1):
            deltas[i] = np.dot(deltas[i+1], self.weights[i+1].T) * self.relu_derivative(self.z_values[i])
        
        # Calculate gradients with regularization
        self.dW = []
        self.db = []
        for i in range(len(self.weights)):
            dW = np.dot(self.activations[i].T, deltas[i]) + self.reg_lambda * self.weights[i]
            db = np.sum(deltas[i], axis=0, keepdims=True)
            self.dW.append(dW)
            self.db.append(db)
    
    def update_parameters(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.dW[i]
            self.biases[i] -= self.learning_rate * self.db[i]
    
    def train(self, X, y, epochs, X_val=None, y_val=None, verbose=True):
        for epoch in range(epochs):
            # Forward and backward pass
            output = self.forward(X)
            loss = self.compute_loss(y, output)
            self.backward(X, y, output)
            self.update_parameters()
            
            # Learning rate decay
            if epoch % 200 == 0 and epoch > 0:
                self.learning_rate *= 0.9
            
            if verbose and epoch % 100 == 0:
                train_acc = self.accuracy(X, y)
                val_acc = self.accuracy(X_val, y_val) if X_val is not None else 0
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    def accuracy(self, X, y):
        predictions = np.argmax(self.forward(X), axis=1)
        true_labels = np.argmax(y, axis=1)
        return accuracy_score(true_labels, predictions)

# Enhanced test with better configuration
print("Testing IMPROVED neural network...")

# Generate larger, more complex dataset
X, y = make_classification(n_samples=5000, n_features=20, n_informative=18, 
                          n_redundant=2, n_classes=3, n_clusters_per_class=1,
                          random_state=42)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

y_onehot = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Deeper architecture
layer_sizes = [X_train.shape[1], 128, 64, 32, y_train.shape[1]]  # 4 layers
nn = ImprovedNeuralNetwork(layer_sizes, learning_rate=0.01, reg_lambda=0.0001)

print("Training improved network...")
nn.train(X_train, y_train, epochs=1500, X_val=X_val, y_val=y_val)

test_accuracy = nn.accuracy(X_test, y_test)
print(f"\n IMPROVED Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
