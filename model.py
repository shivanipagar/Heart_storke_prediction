import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt

# Load datasets
stroke_df = pd.read_csv("healthcare-dataset-stroke-data.csv")
hospitals_df = pd.read_csv("HospitalsInIndia.csv")

# Clean hospital data
hospitals_df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
hospitals_df.dropna(subset=['Pincode'], inplace=True)

# Check the column names in the hospitals DataFrame
print(hospitals_df.columns)


# Preprocess stroke dataset
stroke_df['bmi'] = stroke_df.groupby('gender')['bmi'].transform(lambda x: x.fillna(x.median()))
stroke_df.drop(columns='id', inplace=True)

# Encode categorical features
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    stroke_df[col] = le.fit_transform(stroke_df[col])
    label_encoders[col] = le

# Scale numerical features
num_cols = ['age', 'avg_glucose_level', 'bmi']
scaler = MinMaxScaler()
stroke_df[num_cols] = scaler.fit_transform(stroke_df[num_cols])

# Define features and target
X = stroke_df.drop(columns=['stroke'])
y = stroke_df['stroke']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Balance dataset with SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=20)
rf_model.fit(X_train_bal, y_train_bal)

# Selected top features (based on feature importance)
top_features = X.columns.tolist()

# =========================
# Functions
# =========================

def predict_stroke(user_input):
    """
    Predict stroke probability based on user input.
    user_input: dict
    Returns: probability (float), prediction (0/1)
    """
    # Prepare input
    input_df = pd.DataFrame([user_input])

    # Encode categorical columns
    for col in categorical_cols:
        if col in input_df.columns:
            input_df[col] = label_encoders[col].transform(input_df[col])

    # Scale numerical columns
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Predict
    input_data = input_df[top_features]
    probability = rf_model.predict_proba(input_data)[:, 1][0]
    prediction = rf_model.predict(input_data)[0]

    print("\n--- Debug Info ---")
    print("Encoded Input:", input_df)
    print(f"Predicted Probability: {probability:.4f}, Prediction: {prediction}")
    print("------------------\n")

    return probability, prediction

def recommend_hospitals(city, area):
    """
    Find hospitals near a city and area.
    Returns a list of dicts for JSON API.
    """
    city = city.lower().strip()
    area = area.lower().strip()

    hospitals_df['City'] = hospitals_df['City'].str.lower().str.strip()
    hospitals_df['LocalAddress'] = hospitals_df['LocalAddress'].str.lower().str.strip()

    nearby = hospitals_df[
        (hospitals_df['City'] == city) &
        (hospitals_df['LocalAddress'].str.contains(area, na=False))
    ]

    if not nearby.empty:
        # Rename columns to match frontend expectation
        result = nearby.rename(columns={
            'Hospital': 'Hospital Name',
            'LocalAddress': 'Local Address',
            'City': 'City',
            'State': 'State',
            'Pincode': 'Pincode'
        })

        # Return only the expected fields (if available)
        expected_cols = ['Hospital Name', 'Local Address', 'City', 'State', 'Pincode']
        result = result[[col for col in expected_cols if col in result.columns]]
        
        return result.to_dict(orient='records')
    else:
        return []


if __name__ == "__main__":
    sample_input = {
        'gender': 'Male',
        'age': 67,
        'hypertension': 0,
        'heart_disease': 1,
        'ever_married': 'Yes',
        'work_type': 'Private',
        'Residence_type': 'Urban',
        'avg_glucose_level': 228.69,
        'bmi': 36.6,
        'smoking_status': 'formerly smoked'
    }
    prob, pred = predict_stroke(sample_input)
    print(f"Stroke Risk Probability: {prob:.2f}", "| Prediction:", pred)


    importances = rf_model.feature_importances_
    features = X.columns
    plt.figure(figsize=(10, 6))
    plt.barh(features, importances)
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
