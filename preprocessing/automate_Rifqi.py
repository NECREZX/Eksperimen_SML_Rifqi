import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path):
    print(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def preprocess_data(df):
    print("Starting preprocessing...")
    # Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Balance'] = df['Balance'].fillna(df['Balance'].mean())
    
    # Encode categorical variables
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    
    # Feature Scaling
    scaler = StandardScaler()
    scaled_features = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    df[scaled_features] = scaler.fit_transform(df[scaled_features])
    
    print("Preprocessing completed.")
    return df

def save_data(df, output_path):
    print(f"Saving preprocessed data to {output_path}")
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_file = os.path.join(os.path.dirname(__file__), "..", "churn_raw", "churn.csv")
    output_dir = os.path.join(os.path.dirname(__file__), "churn_preprocessing")
    output_file = os.path.join(output_dir, "churn_processed.csv")
    
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_data(input_file)
    processed_df = preprocess_data(df)
    save_data(processed_df, output_file)
