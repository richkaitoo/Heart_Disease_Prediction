# main.py
from data_loader import load_and_preprocess_data
from eda import plot_correlation, plot_categorical_distributions
from modeling import evaluate_models, final_knn_evaluation
from visualization import plot_model_comparison, plot_roc_curve
import os
import pandas as pd
import joblib

def main():
    # Create output directories
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/results', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    
    # Download and preprocess data
    if not os.path.exists('heart.csv'):
        download_dataset()
    df = load_and_preprocess_data()
    
    # Exploratory Data Analysis
    plot_correlation(df, 'outputs/figures/correlation_matrix.png')
    plot_categorical_distributions(df, 'outputs/figures/categorical_distributions.png')
    
    # Model Evaluation
    results = evaluate_models(df, trials=1000)
    results.to_csv('outputs/results/model_results.csv', index=False)
    plot_model_comparison(results, 'outputs/figures/model_comparison.png')
    
    # Final KNN Evaluation
    knn_metrics, y_prob, y_test, knn_pipe = final_knn_evaluation(df)
    
    print("\nFinal KNN Performance:")
    print(f"Accuracy: {knn_metrics['accuracy']:.3f}")
    print(f"Sensitivity: {knn_metrics['sensitivity']:.3f}")
    print(f"Specificity: {knn_metrics['specificity']:.3f}")
    print(f"AUC: {knn_metrics['roc_auc']:.3f}")
    
    # Save the trained model
    joblib.dump(knn_pipe, 'model/knn_model.pkl')
    print("Model saved to model/knn_model.pkl")
    
    # Save metrics
    pd.DataFrame([knn_metrics]).to_csv('outputs/results/final_knn_metrics.csv', index=False)
    
    # Save test predictions for app
    pd.DataFrame({'y_test': y_test, 'y_prob': y_prob}).to_csv(
        'outputs/results/prediction_data.csv', index=False
    )
    print("Test predictions saved to outputs/results/prediction_data.csv")
    
    # Plot ROC curve
    plot_roc_curve(y_test, y_prob, 'outputs/figures/roc_curve.png')

if __name__ == "__main__":
    main()