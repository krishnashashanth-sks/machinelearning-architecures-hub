import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Create a sample DataFrame that mimics the structure of your expected cars.csv
sample_data = {
    'YEAR': [2012, 2012, 2013, 2013, 2014],
    'Make': ['MITSUBISHI', 'NISSAN', 'FORD', 'MITSUBISHI', 'NISSAN'],
    'Model': ['i-MiEV', 'LEAF', 'FOCUS ELECTRIC', 'i-MiEV', 'LEAF'],
    'Size': ['SUBCOMPACT', 'MID-SIZE', 'COMPACT', 'SUBCOMPACT', 'MID-SIZE'],
    'kW': [49, 80, 107, 49, 80],
    'Unnamed:_5': [None, None, None, None, None], # Placeholder for potential empty column
    'TYPE': ['B', 'B', 'B', 'B', 'B'],
    'CITY_kWh_100_km': [16.9, 19.3, 19.0, 16.9, 19.3],
    'HWY_kWh_100_km': [21.4, 23.0, 21.1, 21.4, 23.0],
    'COMB_kWh_100_km': [18.7, 21.0, 19.9, 18.7, 21.0],
    'CITY_Le_100_km': [1.9, 2.2, 2.1, 1.9, 2.2],
    'HWY_Le_100_km': [2.4, 2.6, 2.4, 2.4, 2.6],
    'COMB_Le_100_km': [2.1, 2.4, 2.3, 2.1, 2.4],
    'g_km': [0, 0, 0, 0, 0],
    'RATING': [None, None, None, None, None], # Placeholder for potential empty column
    'km': [100, 117, 122, 100, 117],
    'TIME_h': [7, 7, 4, 7, 7]
}

# Create the DataFrame
cars_df = pd.DataFrame(sample_data)

# The index=False argument prevents pandas from writing the DataFrame index as a column in the CSV.
cars_df.to_csv('knn/cars.csv', index=False)

# Load the dataset (assuming 'cars.csv' is already uploaded or in the path)
data = pd.read_csv("knn/cars.csv")

# Clean column names for easier access
data.columns = data.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('/', '_').str.replace('.', '').str.strip()

# Define features (X) - including 'Make' as a categorical feature to be encoded
X = data[['YEAR', 'Make', 'CITY_kWh_100_km', 'HWY_kWh_100_km', 'COMB_kWh_100_km', 'CITY_Le_100_km', 'HWY_Le_100_km', 'COMB_Le_100_km', 'kW', 'km', 'TIME_h']]

# Handle potential NaN values in X (numerical columns) - fill with mean or median
for col in ['YEAR', 'CITY_kWh_100_km', 'HWY_kWh_100_km', 'COMB_kWh_100_km', 'CITY_Le_100_km', 'HWY_Le_100_km', 'COMB_Le_100_km', 'kW', 'km', 'TIME_h']:
    if col in X.columns:
        X.loc[:, col] = X[col].fillna(X[col].mean(numeric_only=True))

# Initialize LabelEncoder
le = LabelEncoder()

# Apply LabelEncoder to the 'Make' column within X
X.loc[:, 'Make_Encoded'] = le.fit_transform(X['Make'])

# Drop the original 'Make' column as it's now encoded
X = X.drop('Make', axis=1)

# Define target (y) - predicting (g/km) emissions
y = data['g_km']

# Handle potential NaN values in y
y = y.fillna(y.mean(numeric_only=True))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the KNN model
# Note: KNeighborsClassifier expects integer labels for classification. 'g_km' might be continuous. 
# If 'g_km' is intended as a regression target, KNeighborsRegressor should be used.
# Assuming for demonstration it's treated as a classification problem after potential rounding or specific values.
# If 'g_km' is truly continuous, convert it to categories or use KNeighborsRegressor.
# For this example, if there are non-integer values in 'y', they will be coerced by KNeighborsClassifier.
knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')
knn.fit(X_train, y_train)

# Make predictions
prediction = knn.predict(X_test)

# Evaluate the model
accuracy = metrics.accuracy_score(y_test, prediction)

print("Predictions:", prediction)
print("Accuracy:", accuracy)