import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# 1. Load Data
def load_data(file_path):
    print(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

# 2. Clean and Transform Data
def preprocess_data(df):
    print("Starting data preprocessing...")

    # Split into features and target (assuming last column is the target)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Handle missing values in numerical columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    num_imputer = SimpleImputer(strategy='mean')
    X[num_cols] = num_imputer.fit_transform(X[num_cols])

    # Handle categorical columns
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        X[col].fillna("Missing", inplace=True)
        X[col] = LabelEncoder().fit_transform(X[col])

    # Encode target column if it's categorical
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    return X, y

# 3. Feature Scaling
def scale_data(X):
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# 4. Save Preprocessed Data
def save_processed_data(X, y, output_dir="processed_output"):
    print(f"Saving processed data to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(X).to_csv(f"{output_dir}/features.csv", index=False)
    pd.DataFrame(y, columns=["target"]).to_csv(f"{output_dir}/target.csv", index=False)

# 5. Main Pipeline
def run_etl_pipeline(csv_file_path):
    df = load_data(csv_file_path)
    X, y = preprocess_data(df)
    X_scaled = scale_data(X)
    save_processed_data(X_scaled, y)
    print("ETL pipeline completed successfully.")

if __name__ == "__main__":
    run_etl_pipeline("train.csv") 
