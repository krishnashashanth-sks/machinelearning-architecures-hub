from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import rand
from scipy.linalg import eigh
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from utils import update_deep_esn_states

### Data
# 1. Define a time-series dataset (synthetic for demonstration)
# Generate a time series with a sine wave and some noise
np.random.seed(42) # for reproducibility
time_steps = np.arange(0, 1000)
data = np.sin(time_steps / 100 * 2 * np.pi) + np.random.normal(0, 0.1, len(time_steps))

df_time_series = pd.DataFrame({
    'time': pd.to_datetime(time_steps, unit='D', origin=pd.Timestamp('2000-01-01')),
    'value': data
})

print("Synthetic Time Series Data Head:")
print(df_time_series.head())

plt.figure(figsize=(12, 6))
plt.plot(df_time_series['time'], df_time_series['value'])
plt.title('Synthetic Time Series Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# 2. Apply preprocessing steps (e.g., normalization, scaling)
# Reshape data to be 2D for scaler
data_values = df_time_series['value'].values.reshape(-1, 1)

# Initialize MinMaxScaler to scale data to [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scaler.fit_transform(data_values)

# Update the DataFrame with scaled values
df_time_series['scaled_value'] = scaled_data

print("Scaled Time Series Data Head:")
print(df_time_series.head())
# 3. Split the preprocessed data sequentially into training, validation, and testing sets
# Define split ratios
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# Calculate split indices
total_samples = len(df_time_series)
train_size = int(total_samples * train_ratio)
val_size = int(total_samples * val_ratio)

# Ensure that the sizes sum up to total_samples, test_size takes remaining
if (train_size + val_size) > total_samples:
    val_size = total_samples - train_size
    test_size = 0
else:
    test_size = total_samples - train_size - val_size

# Split the scaled data
train_data = df_time_series['scaled_value'].iloc[:train_size].values
val_data = df_time_series['scaled_value'].iloc[train_size : train_size + val_size].values
test_data = df_time_series['scaled_value'].iloc[train_size + val_size:].values

print(f"Training data shape: {train_data.shape}")
print(f"Validation data shape: {val_data.shape}")
print(f"Testing data shape: {test_data.shape}")

# Display a segment of the splits to confirm
print("\nFirst 5 training data points:")
print(train_data[:5])
print("\nFirst 5 validation data points:")
print(val_data[:5])
print("\nFirst 5 testing data points:")
print(test_data[:5])

### Reservoir core compoenents
# 1. Define key hyperparameters for the reservoir
reservoir_size = 500  # Number of neurons in the reservoir
input_dim = 1         # Dimension of the input data (univariate time series)
spectral_radius = 0.9 # Desired spectral radius of the reservoir weight matrix
sparsity = 0.1        # Sparsity of the internal reservoir connections

print(f"Reservoir Size: {reservoir_size}")
print(f"Input Dimension: {input_dim}")
print(f"Spectral Radius: {spectral_radius}")
print(f"Sparsity: {sparsity}")

# 2. Initialize the input weight matrix (W_in)
# W_in connects the input to the reservoir neurons.
# Dimensions: (reservoir_size, input_dim)
W_in = (np.random.rand(reservoir_size, input_dim) * 2) - 1 # Uniform distribution between -1 and 1

print(f"\nW_in shape: {W_in.shape}")

# 3. Initialize the internal reservoir weight matrix (W_res) as a sparse matrix
# Dimensions: (reservoir_size, reservoir_size)
W_res_sparse = rand(reservoir_size, reservoir_size, density=sparsity, format="csc")
W_res = W_res_sparse.toarray() # Convert to dense array for eigenvalue calculation

# Scale the non-zero elements (optional, but often done to control initial dynamics)
W_res = W_res * ((np.random.rand(reservoir_size, reservoir_size) * 2) - 1)

print(f"W_res shape before scaling: {W_res.shape}")
print(f"W_res sparsity (approx): {1 - np.count_nonzero(W_res) / (reservoir_size * reservoir_size)}")

# 4. Normalize the W_res matrix by scaling its spectral radius
# Calculate eigenvalues of W_res. For non-symmetric matrices, use np.linalg.eigvals
# For very large matrices, this can be computationally intensive.
eigenvalues = np.linalg.eigvals(W_res)
current_spectral_radius = np.max(np.abs(eigenvalues))

# Scale W_res to the desired spectral_radius
if current_spectral_radius != 0:
    W_res = W_res * (spectral_radius / current_spectral_radius)

print(f"Current spectral radius before scaling: {current_spectral_radius:.4f}")

# Verify spectral radius after scaling (recalculate)
eigenvalues_scaled = np.linalg.eigvals(W_res)
verified_spectral_radius = np.max(np.abs(eigenvalues_scaled))
print(f"Verified spectral radius after scaling: {verified_spectral_radius:.4f}")

# 5. Initialize the reservoir state vector x with zeros
# Dimensions: (reservoir_size, 1)
x = np.zeros((reservoir_size, 1))

print(f"\nInitial reservoir state (x) shape: {x.shape}")

### Second layer
# --- 1. Define hyperparameters for a second reservoir layer ---
reservoir_size_2 = 300 # Number of neurons in the second reservoir
spectral_radius_2 = 0.8 # Desired spectral radius for the second reservoir
sparsity_2 = 0.2        # Sparsity of internal connections in the second reservoir
input_scaling_2 = 0.5   # Scaling for inputs to the second reservoir

print(f"\nLayer 2 Hyperparameters:")
print(f"Reservoir Size (Layer 2): {reservoir_size_2}")
print(f"Spectral Radius (Layer 2): {spectral_radius_2}")
print(f"Sparsity (Layer 2): {sparsity_2}")
print(f"Input Scaling (Layer 2): {input_scaling_2}")

# --- 2. Initialize the input weight matrix for Layer 2 (W_in_2) ---
input_dim_2 = input_dim + reservoir_size # This assumes x_1 from previous cell is the state of Layer 1

W_in_2 = (np.random.rand(reservoir_size_2, input_dim_2) * 2 - 1) * input_scaling_2 # Scale weights for input

print(f"\nW_in_2 shape: {W_in_2.shape}")

# --- 3. Initialize the internal reservoir weight matrix for Layer 2 (W_res_2) ---
W_res_2_sparse = rand(reservoir_size_2, reservoir_size_2, density=sparsity_2, format="csc")
W_res_2 = W_res_2_sparse.toarray()

# Scale the non-zero elements (optional, but often done to control initial dynamics)
W_res_2 = W_res_2 * ((np.random.rand(reservoir_size_2, reservoir_size_2) * 2) - 1)

print(f"W_res_2 shape before scaling: {W_res_2.shape}")
print(f"W_res_2 sparsity (approx): {1 - np.count_nonzero(W_res_2) / (reservoir_size_2 * reservoir_size_2)}")

# Normalize W_res_2 by scaling its spectral radius
eigenvalues_2 = np.linalg.eigvals(W_res_2)
current_spectral_radius_2 = np.max(np.abs(eigenvalues_2))

if current_spectral_radius_2 != 0:
    W_res_2 = W_res_2 * (spectral_radius_2 / current_spectral_radius_2)

print(f"Current spectral radius (Layer 2) before scaling: {current_spectral_radius_2:.4f}")

# Verify spectral radius after scaling
eigenvalues_scaled_2 = np.linalg.eigvals(W_res_2)
verified_spectral_radius_2 = np.max(np.abs(eigenvalues_scaled_2))
print(f"Verified spectral radius (Layer 2) after scaling: {verified_spectral_radius_2:.4f}")

# Initialize the reservoir state vector x_2 with zeros
x_2 = np.zeros((reservoir_size_2, 1))

print(f"\nInitial reservoir state (x_2) shape: {x_2.shape}")
print("Deep ESN Layer 2 components initialized successfully.")

# 6. Collect reservoir states across the training data

# Define washout period (number of initial time steps to discard to let reservoir dynamics stabilize)
washout_len = 100

# Initialize lists to store states
# We will store the augmented state which includes input, x1, and x2
all_augmented_states = []
all_targets = []

# Initialize current states for both layers
x1_curr = x # x was initialized as the state for Layer 1 in a previous step
x2_curr = x_2 # x_2 was initialized as the state for Layer 2 in a previous step

# Iterate through the training data to collect states
# For time series prediction, the target is usually the next value in the series.
# So, for input train_data[i], the target is train_data[i+1].
# This means we iterate up to train_data.shape[0] - 1.

print(f"Collecting states for {len(train_data) - 1} training steps with a washout of {washout_len} steps...")

for t in range(len(train_data) - 1):
    # Current input u(t) and target y(t) for the next step
    u_t = np.array([[train_data[t]]]) # Input must be 2D for dot product
    y_t_target = np.array([[train_data[t+1]]]) # Target is the next value

    # Update deep ESN states
    x1_curr, x2_curr = update_deep_esn_states(u_t, x1_curr, x2_curr, W_in, W_res, W_in_2, W_res_2)

    # After washout, collect states and targets
    if t >= washout_len:
        # The augmented state is [u(t); x1(t); x2(t)]
        # Concatenate u_t, x1_curr, x2_curr
        augmented_state_t = np.vstack((u_t, x1_curr, x2_curr))
        all_augmented_states.append(augmented_state_t.flatten()) # Flatten for easier matrix creation
        all_targets.append(y_t_target.flatten()) # Flatten targets too

# Convert lists to NumPy arrays
# X is the state matrix, R in ESN literature, where each row is an augmented state vector
X_train = np.array(all_augmented_states)
# Y_train is the target matrix, where each row is the corresponding target output
Y_train = np.array(all_targets)

print(f"Collected X_train shape: {X_train.shape}")
print(f"Collected Y_train shape: {Y_train.shape}")
print("Reservoir states collected successfully for training the output layer.")

# 7. Train the output layer (W_out)
# W_out maps the augmented reservoir states to the target output.
# This is typically a linear regression problem.
# Using Tikhonov regularization (Ridge Regression) is common for stability.

# Regularization parameter (alpha or lambda) for Ridge Regression
# A small value (e.g., 1e-8) is often used to ensure numerical stability without strong regularization.
alpha = 1e-8

# Calculate W_out using a pseudo-inverse with Tikhonov regularization:
# W_out = (X_train.T @ X_train + alpha * I)^-1 @ X_train.T @ Y_train
# where I is the identity matrix.

# (X_train.T @ X_train) is (input_dim + reservoir_size + reservoir_size_2, input_dim + reservoir_size + reservoir_size_2)
# Add identity matrix scaled by alpha
identity_matrix = alpha * np.eye(X_train.shape[1])

# Solve for W_out
# np.linalg.solve is more numerically stable than direct inverse calculation
W_out = np.linalg.solve(X_train.T @ X_train + identity_matrix, X_train.T @ Y_train)

print(f"W_out shape: {W_out.shape}")
print("Output layer (W_out) trained successfully using Ridge Regression.")

### Evaluation

# 1. Initialize the reservoir states to zeros for the evaluation phase
# Use the same shapes as the original initialized states x and x_2
x1_eval = np.zeros_like(x)
x2_eval = np.zeros_like(x_2)

# 2. Create an empty list to store the model's predictions
predictions = []

print(f"Generating predictions for {len(test_data) - 1} steps in the test set...")

# 3. Iterate through the test_data to generate predictions
# We predict up to the second to last element, as target for t is t+1
for t in range(len(test_data) - 1):
    # Extract current input u_t from test_data[t]
    u_t_eval = np.array([[test_data[t]]]) # Input must be 2D for dot product

    # Update deep ESN states using the evaluation states
    x1_eval, x2_eval = update_deep_esn_states(u_t_eval, x1_eval, x2_eval, W_in, W_res, W_in_2, W_res_2)

    # Form the augmented state for prediction: [u(t); x1(t); x2(t)]
    augmented_state_t_eval = np.vstack((u_t_eval, x1_eval, x2_eval))

    # Calculate the prediction y_pred using W_out
    # W_out is (801, 1), augmented_state_t_eval is (801, 1). So W_out.T @ augmented_state_t_eval gives (1, 1)
    y_pred = (W_out.T @ augmented_state_t_eval)[0, 0]

    # Append the scalar prediction to the predictions list
    predictions.append(y_pred)

# 4. Convert the predictions list to a NumPy array
predictions = np.array(predictions)

# 6. Calculate the actual target values for the test set
# Since we predicted test_data[t+1] using test_data[t], the actual targets are test_data[1:]
actual_targets = test_data[1:]

# Check shapes before calculating metrics
print(f"\nPredictions shape: {predictions.shape}")
print(f"Actual targets shape: {actual_targets.shape}")

# 7. Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(actual_targets, predictions))

# 8. Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(actual_targets, predictions)

# 9. Print the calculated RMSE and MAE
print(f"\nModel Evaluation on Test Set:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")