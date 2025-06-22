# test_model.py
import joblib
import pandas as pd

# Load model
model = joblib.load('model/knn_model.pkl')

# Sample input with correct data types
sample_data = {
    'ca': [0],
    'Cp': [3],        # Integer
    'Sex': [1],        # Integer
    'oldpeak': [2.3],  # Float
    'Thal': ['2'],     # String!
    'thalach': [150],  # Integer
    'Exang': [1]       # Integer
}
input_df = pd.DataFrame(sample_data)

print("Model type:", type(model))
print("Input data types:\n", input_df.dtypes)
print("Prediction:", model.predict(input_df))
print("Probabilities:", model.predict_proba(input_df))