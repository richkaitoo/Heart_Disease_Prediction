# app.py
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Load model and data
@st.cache_resource
def load_model():
    return joblib.load('model/knn_model.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('data/heart.csv')

# Initialize
model = load_model()
df = load_data()

# App title and description
st.title('❤️ Heart Disease Risk Predictor')
st.markdown("""
Interactive tool for predicting heart disease risk using machine learning.
Adjust patient parameters on the left to see prediction results.
""")

# Sidebar for inputs
st.sidebar.header('Patient Parameters')

# Input widgets with validation
ca = st.sidebar.slider('Major Vessels (0-3)', 0, 3, 1)
cp = st.sidebar.selectbox('Chest Pain Type', 
                         options=[0, 1, 2, 3],
                         format_func=lambda x: 
                         ['Typical Angina', 'Atypical Angina', 
                          'Non-anginal Pain', 'Asymptomatic'][x])
sex = st.sidebar.radio('Gender', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
oldpeak = st.sidebar.slider('ST Depression', 0.0, 6.0, 1.0)
thal = st.sidebar.selectbox('Thalassemia', 
                           options=[0, 1, 2],
                           format_func=lambda x: 
                           ['Normal', 'Fixed Defect', 'Reversible Defect'][x])
thalach = st.sidebar.slider('Max Heart Rate', 70, 200, 150)
exang = st.sidebar.radio('Exercise Angina', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')

# Create feature vector with correct data types
# Thal needs to be string, Cp needs to be integer
input_data = [[ca, cp, sex, oldpeak, str(thal), thalach, exang]]  # Convert only Thal to string
columns = ['ca', 'Cp', 'Sex', 'oldpeak', 'Thal', 'thalach', 'Exang']

# Make prediction
if st.sidebar.button('Predict Risk'):
    input_df = pd.DataFrame(input_data, columns=columns)
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    risk_prob = proba[1]  # Probability of heart disease
    
    # Display results
    st.subheader('Prediction Result')
    if prediction == 1:
        st.error(f'High Risk of Heart Disease ({risk_prob:.1%} probability)')
        st.markdown("**Recommendation:** Immediate cardiology referral")
    else:
        st.success(f'Low Risk of Heart Disease ({1-risk_prob:.1%} probability)')
        st.markdown("**Recommendation:** Routine follow-up")
    
    # Show probability gauge
    st.progress(risk_prob)
    st.caption(f'Risk probability: {risk_prob:.1%}')

# EDA Visualizations
st.header('Data Exploration')
tab1, tab2, tab3 = st.tabs(["Correlation", "Feature Distributions", "Model Performance"])

with tab1:
    st.subheader('Correlation Matrix')
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)

with tab2:
    st.subheader('Feature vs Target')
    feature = st.selectbox('Select feature', 
                         ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'])
    fig, ax = plt.subplots()
    sns.boxplot(x='target', y=feature, data=df, ax=ax)
    st.pyplot(fig)

with tab3:
    st.subheader('Model Performance')
    try:
        # Load pre-saved test data
        pred_data = pd.read_csv('outputs/results/prediction_data.csv')
        fpr, tpr, _ = roc_curve(pred_data['y_test'], pred_data['y_prob'])
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        st.pyplot(fig)
        
    except FileNotFoundError:
        st.warning("Performance data not available. Run main.py first")

# Dataset preview
st.header('Dataset Preview')
st.dataframe(df.head(10))
st.caption(f'Full dataset contains {len(df)} patient records')