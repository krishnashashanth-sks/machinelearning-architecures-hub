from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
import random
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.sql.functions import col
from pyspark.sql.functions import lag, window, avg, dayofweek, month, hour, lit
from pyspark.sql.window import Window
from pyspark.sql.functions import count, to_date, substring_index, when, col, sum, trim
from pyspark.sql.functions import col, lit
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType
import random
from datetime import datetime, timedelta
import findspark
from pyspark.sql import SparkSession

# 1. Initialize findspark
findspark.init()

# 2. Initialize a SparkSession named 'BatchLayerProcessing'
spark = SparkSession.builder \
    .appName('BatchLayerProcessing') \
    .getOrCreate()

print("SparkSession 'BatchLayerProcessing' initialized successfully.")

# 3. Generate a conceptual historical dataset

num_users = 100
num_events_per_user = 1000
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 1, 1)
event_types = ["page_view", "product_click", "add_to_cart", "purchase", "login", "logout", "search"]
product_ids = [f"P{i:03d}" for i in range(1, 51)]
categories = ["electronics", "books", "clothing", "home_goods", "sports"]

historical_data = []

for i in range(num_users):
    user_id = f"user_{i:03d}"
    for j in range(num_events_per_user):
        event_type = random.choice(event_types)
        # Random timestamp within the year
        time_offset_seconds = random.randint(0, int((end_date - start_date).total_seconds()))
        timestamp = start_date + timedelta(seconds=time_offset_seconds)

        payload = ""
        session_id = None

        if event_type == "product_click" or event_type == "add_to_cart" or event_type == "purchase":
            product_id = random.choice(product_ids)
            category = random.choice(categories)
            payload = f"product_id={product_id}, category={category}"
            if event_type == "purchase":
                payload += f", amount={round(random.uniform(5.0, 500.0), 2)}"
            session_id = f"sess_{random.randint(100000, 999999)}"
        elif event_type == "page_view":
            page_path = random.choice(["/home", "/products", "/cart", "/checkout", "/about"])
            payload = f"page={page_path}"
            session_id = f"sess_{random.randint(100000, 999999)}"
        elif event_type == "search":
            query = random.choice(["laptop", "book", "t-shirt", "shoes", "mug"])
            payload = f"query={query}"
            session_id = f"sess_{random.randint(100000, 999999)}"

        event_data = {
            "user_id": user_id,
            "event_type": event_type,
            "timestamp": timestamp,
            "payload": payload,
            "session_id": session_id # Can be None
        }
        historical_data.append(event_data)

# 4. Convert the generated historical data into a Spark DataFrame

# Define schema for the DataFrame
schema = StructType([
    StructField("user_id", StringType(), True),
    StructField("event_type", StringType(), True),
    StructField("timestamp", TimestampType(), True),
    StructField("payload", StringType(), True),
    StructField("session_id", StringType(), True)
])

df_historical = spark.createDataFrame(historical_data, schema=schema)

print("Spark DataFrame created successfully.")
df_historical.printSchema()
df_historical.show(5, truncate=False)
# 5. Implement data cleaning and preprocessing operations

# Drop duplicates based on a combination of user_id, event_type, and timestamp
# Assuming these three fields together define a unique event for this dataset.
initial_count = df_historical.count()
df_cleaned = df_historical.dropDuplicates(['user_id', 'event_type', 'timestamp'])
duplicate_count = initial_count - df_cleaned.count()
print(f"Initial record count: {initial_count}")
print(f"Duplicate records removed: {duplicate_count}")
print(f"Cleaned record count: {df_cleaned.count()}")

# Handle potential missing session_id by filling with a default value 'unknown_session'
# Note: In our generated data, session_id is None for some events, which Spark interprets as null.
df_cleaned = df_cleaned.na.fill({'session_id': 'unknown_session'})

print("Data cleaning (deduplication and handling missing session_id) completed.")
df_cleaned.printSchema()
df_cleaned.show(5, truncate=False)
# 6. Perform complex aggregations over the historical data

# A. Daily event counts per user
df_daily_events = df_cleaned.groupBy(to_date("timestamp").alias("event_date"), "user_id") \
                            .agg(count("event_type").alias("daily_event_count"))

print("\n--- Daily Event Counts per User ---")
df_daily_events.show(5)

# B. Total activity per product category
# Extract product category from payload for relevant event types
df_with_category = df_cleaned.withColumn(
    "product_category",
    when(col("event_type").isin("product_click", "add_to_cart", "purchase"),
         trim(substring_index(substring_index(col("payload"), "category=", -1), ",", 1)))
    .otherwise(lit(None))
)

df_category_activity = df_with_category.filter(col("product_category").isNotNull())\
                                     .groupBy("product_category")\
                                     .agg(count("event_type").alias("total_activity_count"))

print("\n--- Total Activity per Product Category ---")
df_category_activity.show(5)
# 7. Implement advanced feature engineering for machine learning model training

# A. Lagged Features: Previous day's event count for each user
# First, ensure data is ordered by user and timestamp
window_spec_user_date = Window.partitionBy("user_id").orderBy("timestamp")

# Join with daily event counts to get a base for lagged features
df_user_daily_activity = df_cleaned.groupBy("user_id", to_date("timestamp").alias("event_date")) \
                                    .agg(count("event_type").alias("daily_event_count"))

# Re-join to get event-level data with daily counts
df_features = df_cleaned.join(df_user_daily_activity,
                              (df_cleaned.user_id == df_user_daily_activity.user_id) &
                              (to_date(df_cleaned.timestamp) == df_user_daily_activity.event_date),
                              "left") \
                        .select(df_cleaned["*"], df_user_daily_activity["daily_event_count"])

# Define a window for lagged features (lag of 1 day for daily_event_count)
window_spec_lag = Window.partitionBy("user_id").orderBy(to_date("timestamp"))

df_features = df_features.withColumn(
    "prev_day_event_count",
    lag("daily_event_count", 1).over(window_spec_lag)
).fillna(0, subset=["prev_day_event_count"]) # Fill nulls for first day with 0

print("\n--- Lagged Features (Previous Day Event Count) ---")
df_features.select("user_id", "timestamp", to_date("timestamp").alias("event_date"), "daily_event_count", "prev_day_event_count").orderBy("user_id", "timestamp").show(10)

# B. Rolling Statistics: 7-day rolling average of 'daily_event_count' for each user
window_spec_rolling = Window.partitionBy("user_id").orderBy(to_date("timestamp")).rowsBetween(-6, 0)

df_features = df_features.withColumn(
    "7_day_avg_event_count",
    avg("daily_event_count").over(window_spec_rolling)
)

print("\n--- Rolling Statistics (7-day Avg Event Count) ---")
df_features.select("user_id", to_date("timestamp").alias("event_date"), "daily_event_count", "7_day_avg_event_count").orderBy("user_id", "event_date").show(10)

# C. Time-based Features: Day of week, month, hour
df_features = df_features.withColumn("day_of_week", dayofweek("timestamp")) \
                             .withColumn("month", month("timestamp")) \
                             .withColumn("hour", hour("timestamp"))

print("\n--- Time-based Features (Day of Week, Month, Hour) ---")
df_features.select("user_id", "timestamp", "day_of_week", "month", "hour").show(5)


print("\n--- Final Processed Data with Engineered Features ---")
df_features.printSchema()
df_features.show(5, truncate=False)

# 1. Prepare features for machine learning

# Identify numerical and categorical features
# Numerical features: 'daily_event_count', 'prev_day_event_count', '7_day_avg_event_count', 'day_of_week', 'month', 'hour'
# Categorical features: 'user_id', 'event_type' (will be target label), 'session_id'

# Filter out rows with nulls in key numerical features that might have resulted from windowing for simplicity
# In a real scenario, imputation or more sophisticated handling would be used.
# For this conceptual task, we will filter for simplicity.
df_filtered_features = df_features.dropna(subset=[
    'daily_event_count',
    'prev_day_event_count',
    '7_day_avg_event_count'
])

# Convert 'event_type' to numerical labels using StringIndexer
# This will be our target variable for classification
indexer = StringIndexer(inputCol="event_type", outputCol="label")
df_indexed = indexer.fit(df_filtered_features).transform(df_filtered_features)

# Index categorical features: 'session_id'
# 'user_id' can have too many categories for one-hot encoding, so we'll exclude it for simplicity in this conceptual example.
# If user_id were to be used, it might require embedding or another dimensionality reduction technique.
indexer_session = StringIndexer(inputCol="session_id", outputCol="session_id_indexed")
df_indexed = indexer_session.fit(df_indexed).transform(df_indexed)

# One-Hot Encode indexed categorical features
encoder = OneHotEncoder(inputCols=["session_id_indexed"], outputCols=["session_id_vec"])
df_encoded = encoder.fit(df_indexed).transform(df_indexed)

# Define numerical and one-hot encoded features for the vector assembler
numerical_cols = ['daily_event_count', 'prev_day_event_count', '7_day_avg_event_count', 'day_of_week', 'month', 'hour']
categorical_vec_cols = ['session_id_vec']

# Combine all features into a single vector using VectorAssembler
assembler = VectorAssembler(
    inputCols=numerical_cols + categorical_vec_cols,
    outputCol="features"
)
df_ml = assembler.transform(df_encoded)

print("Features prepared for ML training.")
df_ml.select("user_id", "event_type", "label", "features").show(5, truncate=False)
df_ml.printSchema()
# 2. Split the data into training and testing sets
# For reproducibility, set a seed.
(training_data, test_data) = df_ml.randomSplit([0.8, 0.2], seed=42)

print(f"Training data count: {training_data.count()}")
print(f"Test data count: {test_data.count()}")

print("Data split into training and testing sets successfully.")

# 3. Train a Machine Learning Model

# Initialize a LogisticRegression model
lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=10)

# Train the model on the training data
lr_model = lr.fit(training_data)

print("Logistic Regression model trained successfully.")
# 4. Evaluate the trained model

# Make predictions on the test data
predictions = lr_model.transform(test_data)

# Instantiate MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

# Calculate accuracy
accuracy = evaluator.evaluate(predictions)
print(f"Test set accuracy = {accuracy}")

# Calculate F1-score
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = evaluator_f1.evaluate(predictions)
print(f"Test set F1-score = {f1_score}")

print("Model evaluation completed.")

predictions.select("user_id", "event_type", "label", "prediction", "probability").show(10, truncate=False)