from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from model import RandomKitchenSinks
from visualize import plot_decision_boundary

# --- Generate Synthetic Data (Non-linear) ---
X, y = make_moons(n_samples=200, noise=0.15, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Original Feature Shape:", X_train.shape)

# --- 1. Train a Linear Model on Original Features ---
linear_model_original = LogisticRegression(random_state=42)
linear_model_original.fit(X_train, y_train)
y_pred_original = linear_model_original.predict(X_test)
accuracy_original = accuracy_score(y_test, y_pred_original)
print(f"Accuracy with Linear Model on Original Features: {accuracy_original:.4f}")

# --- 2. Train a Linear Model on RKS Transformed Features ---
# Initialize and fit RKS transformer
# gamma chosen to be somewhat appropriate for the make_moons dataset scale
rks_transformer = RandomKitchenSinks(n_components=1000, gamma=5.0)
rks_transformer.fit(X_train)

# Transform features
X_train_rks = rks_transformer.transform(X_train)
X_test_rks = rks_transformer.transform(X_test)

print("RKS Transformed Feature Shape:", X_train_rks.shape)

# Train a linear model on transformed features
linear_model_rks = LogisticRegression(random_state=42, max_iter=200)
linear_model_rks.fit(X_train_rks, y_train)
y_pred_rks = linear_model_rks.predict(X_test_rks)
accuracy_rks = accuracy_score(y_test, y_pred_rks)
print(f"Accuracy with Linear Model on RKS Transformed Features: {accuracy_rks:.4f}")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plot_decision_boundary(X=X_test, y=y_test, model=linear_model_original,tittle= "Linear Model (Original Features)")

plt.subplot(1, 2, 2)

plot_decision_boundary(X=X_test, y=y_test, model=rks_transformer, linear_model_rks=linear_model_rks, tittle="Linear Model (RKS Transformed Features)")

plt.tight_layout()
plt.show()
