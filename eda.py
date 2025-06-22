import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_correlation(df, save_path=None):
    """Plot correlation matrix for numerical features"""
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    corr = df[num_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Correlation Between Numerical Attributes')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_categorical_distributions(df, save_path=None):
    """Boxplots for categorical features vs target"""
    cat_cols = ['Sex', 'Cp', 'Fbs', 'Restecg', 'Exang', 'Slope', 'Thal']
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for i, col in enumerate(cat_cols):
        sns.boxplot(x=col, y='Target', data=df, ax=axes[i])
        axes[i].set_title(f'{col} vs Target')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()