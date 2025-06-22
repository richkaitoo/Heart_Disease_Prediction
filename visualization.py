import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_comparison(results_df, save_path=None):
    """Plot model comparison boxplot"""
    plt.figure(figsize=(10, 6))
    error_df = 1 - results_df
    melted_df = error_df.melt(var_name='Model', value_name='Error')
    
    sns.boxplot(x='Model', y='Error', data=melted_df)
    plt.title('Model Comparison (1000 Trials)')
    plt.ylabel('Test Error Rate')
    plt.ylim(0, 0.5)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_roc_curve(y_true, y_prob, save_path=None):
    """Plot ROC curve"""
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (KNN)')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()