"""
Utility functions for ML model evaluation and visualization
Course: Introduction to Machine Learning - End Term Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from itertools import cycle


def plot_confusion_matrices(results, label_encoder, figsize=(12, 15), cmap="Blues"):
    """
    Plot confusion matrices for multiple models in a grid layout.
    
    Parameters:
    -----------
    results : dict
        Dictionary with model names as keys and result dictionaries as values.
        Each result dict should contain 'confusion_matrix' key.
    label_encoder : LabelEncoder
        Fitted label encoder with class names.
    figsize : tuple, default=(12, 15)
        Figure size (width, height).
    cmap : str, default="Blues"
        Colormap for heatmaps.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object.
    """
    num_models = len(results)
    cols = 2
    rows = math.ceil(num_models / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (name, data) in enumerate(results.items()):
        sns.heatmap(
            data["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap=cmap,
            ax=axes[i],
            cbar_kws={'label': 'Count'}
        )
        axes[i].set_title(f"Confusion Matrix — {name}", fontsize=12, fontweight='bold')
        axes[i].set_xlabel("Predicted Label", fontsize=10)
        axes[i].set_ylabel("True Label", fontsize=10)
        axes[i].set_xticklabels(label_encoder.classes_, rotation=45)
        axes[i].set_yticklabels(label_encoder.classes_, rotation=0)
    
    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    return fig

def plot_classwise_comparison_comprehensive(results, label_encoder, figsize=(16, 5)):
    """
    Create comprehensive class-wise metric comparison across all models.
    
    Parameters:
    -----------
    results : dict
        Dictionary with model results.
    label_encoder : LabelEncoder
        Fitted label encoder.
    figsize : tuple
        Figure size.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    metrics = ['precision', 'recall', 'f1-score']
    classes = label_encoder.classes_
    
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        x = np.arange(len(results))
        width = 0.25
        
        colors = ['#2ecc71', '#e74c3c', '#3498db']  # Green, Red, Blue
        
        for class_idx, class_name in enumerate(classes):
            metric_values = []
            for name, data in results.items():
                report = data['classification_report']
                class_label = str(class_idx)
                metric_values.append(report[class_label][metric])
            
            offset = (class_idx - 1) * width
            ax.bar(x + offset, metric_values, width, label=class_name, 
                   alpha=0.8, color=colors[class_idx], edgecolor='black')
        
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric.capitalize(), fontsize=11, fontweight='bold')
        ax.set_title(f'{metric.capitalize()} by Class', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(results.keys(), rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(title='Class', fontsize=9, loc='lower right')
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Class-wise Performance Comparison: All Models', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    return fig


def plot_class_wise_metrics(results, label_encoder, figsize=(12, 15)):
    """
    Plot class-wise precision, recall, and F1-score for multiple models.
    
    Parameters:
    -----------
    results : dict
        Dictionary with model names as keys and result dictionaries as values.
        Each result dict should contain 'classification_report' key.
    label_encoder : LabelEncoder
        Fitted label encoder with class names.
    figsize : tuple, default=(12, 15)
        Figure size (width, height).
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object.
    """
    metrics = ['precision', 'recall', 'f1-score']
    num_models = len(results)
    cols = 2
    rows = math.ceil(num_models / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (name, data) in enumerate(results.items()):
        report_df = pd.DataFrame(data['classification_report']).T
        class_rows = [r for r in report_df.index if r not in ['accuracy', 'macro avg', 'weighted avg']]
        class_report_df = report_df.loc[class_rows, metrics]
        class_report_df = class_report_df.apply(pd.to_numeric)
        
        class_report_df.plot(kind='bar', ax=axes[i], rot=0)
        axes[i].set_title(f'Class-wise Precision/Recall/F1 — {name}', fontsize=11, fontweight='bold')
        axes[i].set_ylabel('Score', fontsize=10)
        axes[i].set_xlabel('Class', fontsize=10)
        axes[i].set_ylim(0, 1.05)
        axes[i].legend(fontsize=9)
        axes[i].grid(axis='y', alpha=0.3)
        axes[i].set_xticklabels([label_encoder.classes_[int(idx)] for idx in class_rows], rotation=0)
    
    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, model_name, top_n=15, figsize=(10, 6)):
    """
    Plot feature importances for tree-based models.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model with feature_importances_ attribute.
    feature_names : list
        List of feature names.
    model_name : str
        Name of the model for the title.
    top_n : int, default=15
        Number of top features to display.
    figsize : tuple, default=(10, 6)
        Figure size (width, height).
    
    Returns:
    --------
    fi_df : pd.DataFrame
        DataFrame with features and their importance scores (sorted).
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"Model {model_name} does not have feature_importances_ attribute.")
        return None
    
    fi = model.feature_importances_
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': fi})
    fi_df = fi_df.sort_values('importance', ascending=False)
    
    # Plot
    plt.figure(figsize=figsize)
    sns.barplot(data=fi_df.head(top_n), x='importance', y='feature', 
                palette='viridis', hue='feature', legend=False)
    plt.title(f'Top {top_n} Feature Importances — {model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Importance', fontsize=11, fontweight='bold')
    plt.ylabel('Feature', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return fi_df


def plot_train_test_comparison(comparison_df, figsize=(12, 6)):
    """
    Plot training vs test accuracy with gap annotations.
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        DataFrame with columns: 'Model', 'Train Accuracy', 'Test Accuracy', 'Gap (Train - Test)'
    figsize : tuple, default=(12, 6)
        Figure size (width, height).
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(comparison_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, comparison_df['Train Accuracy'], width,
                   label='Train Accuracy', alpha=0.8, color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, comparison_df['Test Accuracy'], width,
                   label='Test Accuracy', alpha=0.8, color='coral', edgecolor='black')
    
    # Add gap annotations
    for i, (train_acc, test_acc) in enumerate(zip(comparison_df['Train Accuracy'],
                                                    comparison_df['Test Accuracy'])):
        gap = train_acc - test_acc
        ax.text(i, max(train_acc, test_acc) + 0.02, f'Δ={gap:.3f}',
                ha='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Training vs. Test Accuracy: Overfitting Detection', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_error_analysis(error_analysis, figsize=(14, 5)):
    """
    Plot prediction confidence distribution and error rate by class.
    
    Parameters:
    -----------
    error_analysis : pd.DataFrame
        DataFrame with columns: 'correct', 'confidence', 'actual'
    figsize : tuple, default=(14, 5)
        Figure size (width, height).
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Confidence distribution for correct vs incorrect
    axes[0].hist(error_analysis[error_analysis['correct']]['confidence'],
                 bins=30, alpha=0.7, label='Correct', color='green', edgecolor='black')
    axes[0].hist(error_analysis[~error_analysis['correct']]['confidence'],
                 bins=30, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
    axes[0].set_xlabel('Prediction Confidence', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0].set_title('Prediction Confidence: Correct vs. Incorrect', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Error rate by actual class
    error_by_class = error_analysis.groupby('actual', observed=True).apply(
        lambda x: (~x['correct']).sum() / len(x) * 100,
        include_groups=False
    ).sort_values(ascending=False)
    
    axes[1].bar(error_by_class.index, error_by_class.values,
                color=['coral', 'steelblue', 'lightgreen'],
                edgecolor='black', alpha=0.8)
    axes[1].set_xlabel('True Class', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Error Rate (%)', fontsize=11, fontweight='bold')
    axes[1].set_title('Error Rate by True Class', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (cls, val) in enumerate(error_by_class.items()):
        axes[1].text(i, val + 0.1, f'{val:.2f}%', ha='center',
                     fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_f1_comparison(summary_df, label_names, figsize=(12, 6)):
    """
    Plot class-wise F1 scores across all models.
    
    Parameters:
    -----------
    summary_df : pd.DataFrame
        DataFrame with model performance metrics including F1 scores per class.
    label_names : array-like
        Array of class label names.
    figsize : tuple, default=(12, 6)
        Figure size (width, height).
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(summary_df))
    width = 0.2
    
    for idx, label in enumerate(label_names):
        offset = (idx - 1) * width
        ax.bar(x + offset, summary_df[f'F1_{label}'], width,
               label=f'{label}', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Class-wise F1 Scores Across All Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['Model'], rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend(title='Class', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_accuracy_comparison(summary_df, figsize=(10, 6)):
    """
    Plot overall accuracy comparison across models with color coding.
    
    Parameters:
    -----------
    summary_df : pd.DataFrame
        DataFrame with columns including 'Model' and 'Accuracy'.
    figsize : tuple, default=(10, 6)
        Figure size (width, height).
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['green' if acc > 0.95 else 'orange' if acc > 0.75 else 'red'
              for acc in summary_df['Accuracy']]
    
    bars = ax.bar(summary_df['Model'], summary_df['Accuracy'],
                  color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, acc in zip(bars, summary_df['Accuracy']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom',
                fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Overall Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticklabels(summary_df['Model'], rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_roc_curves(pipelines, X_test, y_test, label_encoder, 
                    model_names=['RandomForest', 'XGBoost', 'DecisionTree'],
                    figsize=(18, 5)):
    """
    Plot ROC curves for multiple models (one-vs-rest for multi-class).
    
    Parameters:
    -----------
    pipelines : dict
        Dictionary of fitted sklearn pipelines with model names as keys.
    X_test : pd.DataFrame or array-like
        Test features.
    y_test : array-like
        True test labels (encoded as integers).
    label_encoder : LabelEncoder
        Fitted label encoder with class names.
    model_names : list, default=['RandomForest', 'XGBoost', 'DecisionTree']
        List of model names to plot.
    figsize : tuple, default=(18, 5)
        Figure size (width, height).
    
    Returns:
    --------
    auc_summary : pd.DataFrame
        DataFrame with AUC scores for each model and class.
    fig : matplotlib.figure.Figure
        The created figure object.
    """
    # Binarize labels for multi-class ROC
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_bin.shape[1]
    
    fig, axes = plt.subplots(1, len(model_names), figsize=figsize)
    if len(model_names) == 1:
        axes = [axes]
    
    auc_summary = []
    
    for model_idx, model_name in enumerate(model_names):
        ax = axes[model_idx]
        
        # Get predicted probabilities
        y_proba = pipelines[model_name].predict_proba(X_test)
        
        # Compute ROC curve and AUC for each class
        class_colors = cycle(['#9b59b6', '#f39c12', '#1abc9c'])  # Purple, Orange, Teal
        
        class_aucs = {}
        for i, (class_name, color) in enumerate(zip(label_encoder.classes_, class_colors)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            class_aucs[f'AUC_{class_name}'] = roc_auc
            
            ax.plot(fpr, tpr, color=color, lw=2.5,
                    label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.6, label='Random Classifier')
        
        # Formatting
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
        ax.set_title(f'ROC Curves — {model_name}', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_aspect('equal')
        
        # Compute macro and weighted AUC
        macro_auc = roc_auc_score(y_test_bin, y_proba, average='macro')
        weighted_auc = roc_auc_score(y_test_bin, y_proba, average='weighted')
        
        auc_summary.append({
            'Model': model_name,
            'Macro AUC': round(macro_auc, 4),
            'Weighted AUC': round(weighted_auc, 4),
            **{k: round(v, 4) for k, v in class_aucs.items()}
        })
    
    plt.tight_layout()
    
    auc_df = pd.DataFrame(auc_summary)
    return auc_df, fig


def plot_auc_comparison(auc_df, label_encoder, figsize=(14, 5)):
    """
    Plot macro/weighted AUC and per-class AUC comparison.
    
    Parameters:
    -----------
    auc_df : pd.DataFrame
        DataFrame with AUC scores from plot_roc_curves().
    label_encoder : LabelEncoder
        Fitted label encoder with class names.
    figsize : tuple, default=(14, 5)
        Figure size (width, height).
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Macro and Weighted AUC
    ax1 = axes[0]
    x = np.arange(len(auc_df))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, auc_df['Macro AUC'], width,
                    label='Macro AUC', color='steelblue', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, auc_df['Weighted AUC'], width,
                    label='Weighted AUC', color='coral', alpha=0.8, edgecolor='black')
    
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
    ax1.set_title('Macro vs Weighted AUC', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(auc_df['Model'], rotation=0)
    ax1.set_ylim([0.95, 1.0])
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: Per-class AUC
    ax2 = axes[1]
    class_cols = [col for col in auc_df.columns if col.startswith('AUC_')]
    auc_class_data = auc_df[['Model'] + class_cols].set_index('Model')
    
    auc_class_data.plot(kind='bar', ax=ax2, width=0.8,
                        color=['#9b59b6', '#f39c12', '#1abc9c'],
                        edgecolor='black', alpha=0.8)
    
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Class AUC Scores', fontsize=13, fontweight='bold')
    ax2.set_xticklabels(auc_df['Model'], rotation=0)
    ax2.set_ylim([0.95, 1.0])
    ax2.legend(title='Class', labels=label_encoder.classes_, fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def compute_overfitting_metrics(pipelines, X_train, X_test, y_train, y_test):
    """
    Compute train and test accuracy for all models to detect overfitting.
    
    Parameters:
    -----------
    pipelines : dict
        Dictionary of fitted sklearn pipelines.
    X_train, X_test : pd.DataFrame or array-like
        Training and test features.
    y_train, y_test : array-like
        Training and test labels.
    
    Returns:
    --------
    comparison_df : pd.DataFrame
        DataFrame with train/test accuracy and gap for each model.
    """
    from sklearn.metrics import accuracy_score
    
    train_test_comparison = []
    
    for name, pipe in pipelines.items():
        y_train_pred = pipe.predict(X_train)
        y_test_pred = pipe.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        train_test_comparison.append({
            'Model': name,
            'Train Accuracy': round(train_acc, 4),
            'Test Accuracy': round(test_acc, 4),
            'Gap (Train - Test)': round(train_acc - test_acc, 4)
        })
        
        print(f"\n{name}:")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Gap: {train_acc - test_acc:.4f}")
    
    comparison_df = pd.DataFrame(train_test_comparison)
    comparison_df = comparison_df.sort_values('Gap (Train - Test)', ascending=False)
    
    return comparison_df
