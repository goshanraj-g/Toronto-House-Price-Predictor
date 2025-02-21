import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath)
    # Clean column names by stripping extra whitespace
    data.columns = data.columns.str.strip()
    
    print("Dataset Overview:")
    print("-" * 50)
    print(f"Number of records: {len(data):,}")
    print(f"Number of features: {len(data.columns)}")
    print("\nFirst couple records:")
    print(data.head())
    
    missing_values = data.isnull().sum()
    if missing_values.any():
        print("\nMissing values found:")
        print(missing_values[missing_values > 0])
    
    return data

def create_model_pipeline(numerical_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ))
    ])
    
    return model

def evaluate_model(model, X_test, y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Performance Metrics:")
    print("-" * 50)
    print(f"Mean absolute error: ${mae:,.2f}")
    print(f"Mean absolute percentage error: {mape:.2f}%")
    print(f"R2 score: {r2:.3f}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Predictions')
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel("Actual Prices ($)")
    plt.ylabel("Predicted Prices ($)")
    plt.title("Actual vs Predicted Toronto House Prices")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('price_prediction_plot.png')
    plt.close()
    
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, color='green', label='Residuals')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted Prices ($)")
    plt.ylabel("Residuals ($)")
    plt.title("Residual Plot")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('residual_plot.png')
    plt.close()
    
    return {"mae": mae, "mape": mape, "r2": r2}

def save_results(results, filename="model_results.json"):
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {filename}")

def predict_price(model, bedrooms, bathrooms, region):
    input_data = pd.DataFrame({
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'region': [region]
    })
    prediction = model.predict(input_data)
    return prediction[0]

def main():
    BASE_DIR = os.path.dirname(__file__)
    csv_path = os.path.join(BASE_DIR, "dataset", "clean_combined_toronto_property_data.csv")
    
    data = load_and_prepare_data(csv_path)
    
    numerical_features = ['bedrooms', 'bathrooms']
    categorical_features = ['region']
    
    if not set(numerical_features + categorical_features).issubset(data.columns):
        print("Error: The required columns are missing from the dataset!")
        return
    
    X = data[numerical_features + categorical_features]
    y = data['price']  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = create_model_pipeline(numerical_features, categorical_features)
    print("\nTraining model...")
    model.fit(X_train, y_train)
    print("Model training complete!")
    
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    
    results = evaluate_model(model, X_test, y_test, y_pred)
    save_results(results)
    
    print("\nExample Prediction:")
    sample_prediction = predict_price(
        model,
        bedrooms=4, 
        bathrooms=3, 
        region="Vaughan, ON"
    )
    print(f"Predicted price: ${sample_prediction:,.2f}")

if __name__ == "__main__":
    main()
