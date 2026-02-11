# Step 1: Import Libraries
import numpy as np
import matplotlib.pyplot as plt
# Step 2: Define Activation Function

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)
# Step 3: Initialize Dataset

# Input features
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Target outputs
y = np.array([[0],
              [1],
              [1],
              [0]])
np.random.seed(0)

input_size = 2
hidden_size = 4
output_size = 1

# Small random values for weights
W1 = np.random.randn(input_size, hidden_size) * 0.1
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * 0.1
b2 = np.zeros((1, output_size))



learning_rate = 0.1
epochs = 10000
loss_history = []

for epoch in range(epochs):

    # ==========================
    # STEP 5: Forward Propagation
    # ==========================
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    y_hat = sigmoid(z2)

    # ==========================
    # STEP 6: Compute Loss
    # ==========================
    n = y.shape[0]
    loss = (1 / n) * np.sum((y - y_hat) ** 2)
    loss_history.append(loss)

    # ==========================
    # STEP 7: Backpropagation
    # ==========================
    error_output = y_hat - y
    dz2 = error_output * sigmoid_derivative(y_hat)

    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    error_hidden = np.dot(dz2, W2.T)
    dz1 = error_hidden * sigmoid_derivative(a1)

    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # ==========================
    # STEP 8: Update Weights & Biases
    # ==========================
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1



print("Final Predictions:")
print(np.round(y_hat, 3))


plt.figure()
plt.plot(loss_history)
plt.title("Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.show()



x_values = np.linspace(-10, 10, 100)
y_values = sigmoid(x_values)

plt.figure()
plt.plot(x_values, y_values)
plt.title("Sigmoid Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()



weight_range = np.linspace(-2, 2, 100)
loss_surface = []

original_weight = W1[0, 0]

for w in weight_range:
    W1[0, 0] = w

    z1_temp = np.dot(X, W1) + b1
    a1_temp = sigmoid(z1_temp)
    z2_temp = np.dot(a1_temp, W2) + b2
    y_temp = sigmoid(z2_temp)

    temp_loss = np.mean((y - y_temp) ** 2)
    loss_surface.append(temp_loss)

# Restore original weight
W1[0, 0] = original_weight

plt.figure()
plt.plot(weight_range, loss_surface)
plt.title("Loss Landscape (Varying One Weight)")
plt.xlabel("Weight Value")
plt.ylabel("Loss")
plt.show()


# Step 11
test_input = np.array([[1, 1]])
z1_test = np.dot(test_input, W1) + b1
a1_test = sigmoid(z1_test)

z2_test = np.dot(a1_test, W2) + b2
prediction = sigmoid(z2_test)


print("Test Input:", test_input)
print("Predicted Output:", np.round(prediction, 3))
