import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi


def load_and_preprocess_data():
    """Load and preprocess heart dataset"""
    # Load data
    df = pd.read_csv('heart.csv')
    
    # Recode thal variable (as in R analysis)
    bins = [-1, 1, 2, 10]
    labels = ['0', '1', '2']
    df['Thal'] = pd.cut(df['thal'], bins=bins, labels=labels)
    
    # Rename columns
    df = df.rename(columns={
        'sex': 'Sex',
        'cp': 'Cp',
        'fbs': 'Fbs',
        'restecg': 'Restecg',
        'exang': 'Exang',
        'slope': 'Slope',
        'target': 'Target'
    })
    
    # Select relevant columns
    final_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 
                  'Sex', 'Cp', 'Fbs', 'Restecg', 'Exang', 'Slope', 'Thal', 'Target']
    
    return df[final_cols]