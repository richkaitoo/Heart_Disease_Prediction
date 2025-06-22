# modeling.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def evaluate_models(df, trials=1000, test_size=0.5):
    """Evaluate models over multiple trials"""
    results = {'Logistic Regression': [], 'LDA': [], 'KNN (k=1)': []}
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), 
             ['Cp', 'Sex', 'Thal', 'Exang']),
            ('num', StandardScaler(), ['ca', 'oldpeak', 'thalach'])
        ])
    
    X = df[['ca', 'Cp', 'Sex', 'oldpeak', 'Thal', 'thalach', 'Exang']]
    y = df['Target'].astype(int)
    
    for _ in range(trials):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y
        )
        
        # Logistic Regression
        lr_pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000))
        ])
        lr_pipe.fit(X_train, y_train)
        results['Logistic Regression'].append(accuracy_score(y_test, lr_pipe.predict(X_test)))
        
        # LDA
        lda_pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LinearDiscriminantAnalysis())
        ])
        lda_pipe.fit(X_train, y_train)
        results['LDA'].append(accuracy_score(y_test, lda_pipe.predict(X_test)))
        
        # KNN
        knn_pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', KNeighborsClassifier(n_neighbors=1))
        ])
        knn_pipe.fit(X_train, y_train)
        results['KNN (k=1)'].append(accuracy_score(y_test, knn_pipe.predict(X_test)))
    
    return pd.DataFrame(results)

def final_knn_evaluation(df, test_size=0.5):
    """Evaluate final KNN model with detailed metrics"""
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), 
             ['Cp', 'Sex', 'Thal', 'Exang']),
            ('num', StandardScaler(), ['ca', 'oldpeak', 'thalach'])
        ])
    
    X = df[['ca', 'Cp', 'Sex', 'oldpeak', 'Thal', 'thalach', 'Exang']]
    y = df['Target'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    knn_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=1))
    ])
    knn_pipe.fit(X_train, y_train)
    
    y_pred = knn_pipe.predict(X_test)
    y_prob = knn_pipe.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'sensitivity': tp / (tp + fn),
        'specificity': tn / (tn + fp),
        'confusion_matrix': cm.tolist(),  # Convert to list for serialization
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    return metrics, y_prob, y_test, knn_pipe  # Return pipeline for saving