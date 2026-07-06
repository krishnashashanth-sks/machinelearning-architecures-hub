import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from minisom import MiniSom
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
data = iris.data
target = iris.target
target_names = iris.target_names

# Scale the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Define SOM grid dimensions and input dimension
som_grid_rows = 10
som_grid_cols = 10
input_dim = data_scaled.shape[1] # Automatically derive input_dim from the scaled data
# Instantiate MiniSom without pca_init
som = MiniSom(som_grid_rows, som_grid_cols, input_dim, sigma=0.5, learning_rate=0.5, random_seed=42)

# Initialize SOM weights using pca_weights_init()
som.pca_weights_init(data_scaled)

# Train the SOM
num_iteration = 1000
som.train_random(data_scaled, num_iteration, verbose=True)

# Get SOM weight vectors and reshape for clustering
som_weights = som.get_weights()
# The weights are usually in the shape (rows, cols, input_dim). We need to flatten the grid dimensions.
reshaped_som_weights = som_weights.reshape(-1, input_dim)

# Instantiate KMeans with 3 clusters (common for Iris dataset)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

# Fit KMeans to the reshaped SOM weight vectors
kmeans.fit(reshaped_som_weights)

# Get cluster labels for each neuron
neuron_clusters = kmeans.labels_

print(f"Number of neurons clustered: {len(neuron_clusters)}")
print(f"Unique cluster labels: {set(neuron_clusters)}")
# Reshape neuron_clusters back into a 2D grid
clustered_som_grid = neuron_clusters.reshape(som_grid_rows, som_grid_cols)

# Create a figure and an axes object for plotting
plt.figure(figsize=(10, 8))
plt.imshow(clustered_som_grid, cmap='viridis', origin='lower')
plt.colorbar(label='Cluster Label')
plt.title('SOM with K-Means Clusters and Data Points')
plt.xlabel('SOM Grid X-coordinate')
plt.ylabel('SOM Grid Y-coordinate')

# Add data points to the plot, colored by their original target classes
colors = ['red', 'green', 'blue'] # Assuming 3 classes for Iris dataset
for i, data_point in enumerate(data_scaled):
    # Find the Best Matching Unit (BMU) for each data point
    bmu_row, bmu_col = som.winner(data_point)

    # Plot the data point at its BMU's coordinates, colored by its original class
    plt.plot(bmu_col + 0.5, bmu_row + 0.5, 'o', markerfacecolor=colors[target[i]],
             markeredgecolor=colors[target[i]], markersize=5, alpha=0.8)

# Create a legend for the original target classes
# This requires plotting dummy points for the legend
for i, class_name in enumerate(target_names):
    plt.plot([], [], 'o', markerfacecolor=colors[i],
             markeredgecolor=colors[i], markersize=5, label=class_name)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.grid(False)
plt.show()