import pandas as pd
import json
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Setup MLflow Tracking
# -----------------------------
mlflow.set_experiment("Profitability_Prediction_Class2")

# -----------------------------
# 2. Load Data and Encodings
# -----------------------------
print("Loading data...")
df = pd.read_csv("Global_Superstore2.csv", encoding="latin1")

# The columns our model expects as inputs
feature_columns = [
    "Ship Mode", "Segment", "City", "State", "Country", 
    "Market", "Region", "Sub-Category", "Sales", "Quantity", 
    "Discount", "Profit", "Shipping Cost", "Order Priority"
]
target_column = "Category"

# We map the target variable to numbers based on our CLASS_MAP in app.py
target_map = {"Furniture": 0, "Office Supplies": 1, "Technology": 2}
df[target_column] = df[target_column].map(target_map)

# Drop any rows with missing values in our features or target
df = df.dropna(subset=feature_columns + [target_column])

X = df[feature_columns].copy()
y = df[target_column]

# Load encodings to encode categorical variables in X just like in app.py
with open("all_encodings.json", "r") as f:
    encoding_map = json.load(f)

for col, mapping in encoding_map.items():
    if col in X.columns:
        X[col] = X[col].map(mapping).fillna(-1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 3. Train Model with MLflow
# -----------------------------
print("Training model and logging to MLflow...")

with mlflow.start_run() as run:
    # Define hyperparameters
    n_estimators = 100
    max_depth = 10000
    
    # Log parameters to MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("features_count", len(feature_columns))

    # Train the model
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate the model
    predictions = rf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Log metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)

    # Log the model itself
    mlflow.sklearn.log_model(rf, "random_forest_model")
    
    print(f"--- Run Completed! ---")
    print(f"Run ID: {run.info.run_id}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Run 'mlflow ui' in your terminal to see the results!")
