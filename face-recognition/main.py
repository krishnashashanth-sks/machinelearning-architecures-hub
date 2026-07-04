import kagglehub
from skimage.feature import hog
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import os

path=kagglehub.dataset_download("vasukipatel/face-recognition-dataset")

# The 'path' variable holds the root of the extracted Kaggle dataset.
# The actual face images are located directly within a nested 'Faces/Faces' subdirectory,
# with the person's name encoded in the filename (e.g., 'Amitabh Bachchan_58.jpg').
dataset_dir = os.path.join(path, 'Faces', 'Faces')

images = []
labels = []
label_map = {}
current_label = 0

print(f"Loading images from :{dataset_dir}")

# Iterate through each item (expected to be an image file) in the dataset_dir
for image_filename in os.listdir(dataset_dir):
  image_path = os.path.join(dataset_dir, image_filename)

  # Ensure it's a file and has an image extension
  if os.path.isfile(image_path) and image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
    # Extract person_name from the filename (e.g., 'Amitabh Bachchan_58.jpg' -> 'Amitabh Bachchan')
    # This assumes the format 'Name_Number.ext'
    base_name = os.path.splitext(image_filename)[0] # Gets 'Amitabh Bachchan_58'
    parts = base_name.rsplit('_', 1) # Splits into ['Amitabh Bachchan', '58'] or ['Kashyap', '6']
    person_name = parts[0] if len(parts) > 1 else base_name # Handle cases where there might not be an underscore (though unlikely here)

    if not person_name:
        print(f"Warning: Could not extract person name from filename: {image_filename}. Skipping.")
        continue

    # If the person_name is not yet in label_map, assign it a new numerical label
    if person_name not in label_map:
      label_map[person_name] = current_label
      current_label += 1

    # Get the numerical label for the current person
    person_label = label_map[person_name]

    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
      images.append(img)
      labels.append(person_label)
    else:
      print(f"Warning: Could not load image: {image_path}")
  else:
    print(f"Skipping non-image file or non-existent file in faces dataset: {image_filename}")

print(f"Loaded {len(images)} images from {len(label_map)} individuals.")
print(f"Label mapping: {label_map}")

# Convert lists to numpy arrays for easier processing later
images = np.array(images, dtype=object) # Use dtype=object for variable size images initially
labels = np.array(labels)

print("Images loaded successfully. Next, we will resize these images.")

target_size=(100,100)
resized_images=[]
print(f"Resizing all images to {target_size}...")
for i,img in enumerate(images):
  if img is not None:
    if img.size>0:
      # Explicitly convert img to a numerical type expected by cv2.resize
      # The kernel state shows img is an ndarray but with dtype=object, which cv2.resize doesn't like.
      # Assuming the actual pixel data within img is uint8, we convert it.
      img_numerical = img.astype(np.uint8)
      resized_img=cv2.resize(img_numerical,target_size,interpolation=cv2.INTER_AREA)
      resized_images.append(resized_img)
    else:
      print(f"Warning:image at index {i} is empty skipping resize")
  else:
    print(f"Warning: Image at index {i} is None skipping resize")

images = np.array(resized_images) # Overwrite the original images array with resized ones

X_train,X_test,y_train,y_test=train_test_split(images,labels,test_size=0.2,random_state=42,stratify=labels)

X_train_hog_features=[]
print("Extracting HOG features from X_train...")
for i,image in enumerate(X_train):
  if image is not None and image.size>0:
    if image.ndim==3:
      image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    if image.max()>1.0:
      image=image.astype(np.uint8)
    features=hog(image,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),block_norm='L2-Hys',feature_vector=True)
    X_train_hog_features.append(features)
  else:
    print(f"Warning: Skipping empty or invalid image at index {i} in X_train")
print(f"Extracted HOG features for {len(X_train_hog_features)} images in X_train")

X_test_hog_features=[]
print("Extracting HOG features for X_test...")
for i,image in enumerate(X_test):
  if image is not None and image.size>0:
    if image.ndim==3:
      image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    if image.max()>1.0:
      image=image.astype(np.uint8)
    features=hog(image,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),block_norm='L2-Hys',feature_vector=True)
    X_test_hog_features.append(features)
  else:
    print(f"Warning: Skipping empty or invalid image at index {i} in X_test")
print(f"Extracted HOG features for {len(X_test_hog_features)} images in X_test")

X_train_hog_features = np.array(X_train_hog_features)
X_test_hog_features = np.array(X_test_hog_features)

print(f"Shape of X_train_hog_features: {X_train_hog_features.shape}")
print(f"Shape of X_test_hog_features: {X_test_hog_features.shape}")

# Choose Support Vector Machine (SVM) as the classifier
# Instantiate the classifier. Using a linear kernel for simplicity and adding a random_state for reproducibility.
classifier = SVC(kernel='linear', random_state=42)

print("Training the SVC classifier...")
classifier.fit(X_train_hog_features, y_train)
print("Classifier trained successfully.")

print("Making predictions on the test data...")
y_pred = classifier.predict(X_test_hog_features)
print("Predictions made successfully.")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

num_examples = 10
indices = np.random.choice(len(X_test), num_examples, replace=False)

# Reverse the label_map to get names from numerical labels
reversed_label_map = {v: k for k, v in label_map.items()}

plt.figure(figsize=(15, 10))
plt.suptitle('Facial Recognition Examples', fontsize=16)

for i, idx in enumerate(indices):
    image = X_test[idx]
    true_label = y_test[idx]
    predicted_label = y_pred[idx]

    true_label_name = reversed_label_map.get(true_label, f"Unknown_{true_label}")
    predicted_label_name = reversed_label_map.get(predicted_label, f"Unknown_{predicted_label}")

    plt.subplot(2, num_examples // 2, i + 1) # Adjust subplot grid based on num_examples
    plt.imshow(image, cmap='gray')

    title_color = 'green' if true_label == predicted_label else 'red'
    plt.title(f"True: {true_label_name}\nPred: {predicted_label_name}", color=title_color)
    plt.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap
plt.show()
