import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # 100 random points in [0, 2]
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + Gaussian noise

# Linear regression implementation
def linear_regression(X, y):
    # Adding bias term (x0 = 1)
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
    # Calculate theta (coefficients)
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta_best

# Train the model
theta_best = linear_regression(X, y)

# Make predictions
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((X_new.shape[0], 1)), X_new]  # Add bias term
y_predict = X_new_b.dot(theta_best)

# Print the coefficients
print(f"Coefficients: {theta_best.flatten()}")

# Visualization
plt.scatter(X, y, label='Data points')
plt.plot(X_new, y_predict, color='red', label='Predicted line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression')
plt.show()
