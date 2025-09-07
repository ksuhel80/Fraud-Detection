import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve, roc_curve
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and return metrics"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'pr_auc': average_precision_score(y_test, y_pred_proba)
    }
    
    return metrics, y_pred, y_pred_proba

def plot_confusion_matrix(y_true, y_pred, normalize=True, title=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        color_continuous_scale='Blues',
        title=title or "Confusion Matrix"
    )
    
    fig.update_layout(
        xaxis=dict(tickvals=[0, 1], ticktext=['Legitimate', 'Fraud']),
        yaxis=dict(tickvals=[0, 1], ticktext=['Legitimate', 'Fraud'])
    )
    
    return fig

def plot_roc_curve(y_true, y_scores, title=None):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color='darkorange', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='navy', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=title or "Receiver Operating Characteristic (ROC) Curve",
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05])
    )
    
    return fig

def plot_precision_recall_curve(y_true, y_scores, title=None):
    """Plot precision-recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name=f'Precision-Recall Curve (AUC = {pr_auc:.3f})',
        line=dict(color='blue', width=2)
    ))
    
    # Add baseline (fraction of positives)
    baseline = sum(y_true) / len(y_true)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[baseline, baseline],
        mode='lines',
        name=f'Baseline (Positive Rate = {baseline:.3f})',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=title or "Precision-Recall Curve",
        xaxis_title='Recall',
        yaxis_title='Precision',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05])
    )
    
    return fig

def plot_feature_importance(model, feature_names, top_n=20, title=None):
    """Plot feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError("Model does not have feature importances or coefficients")
    
    # Create DataFrame for visualization
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Select top N features
    top_features = feature_importance_df.head(top_n)
    
    # Create bar chart
    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title=title or f"Top {top_n} Feature Importances"
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title='Importance',
        yaxis_title='Feature'
    )
    
    return fig

def plot_threshold_analysis(y_true, y_scores, title=None):
    """Plot threshold analysis"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'threshold': list(thresholds) + [1],  # Add 1 for the last point
        'precision': precision,
        'recall': recall
    })
    
    # Calculate F1 score
    df['f1'] = 2 * (df['precision'] * df['recall']) / (df['precision'] + df['recall'])
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Precision vs Threshold', 'Recall vs Threshold', 'F1 Score vs Threshold', 'Precision vs Recall'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Precision vs Threshold
    fig.add_trace(
        go.Scatter(x=df['threshold'], y=df['precision'], mode='lines', name='Precision'),
        row=1, col=1
    )
    
    # Recall vs Threshold
    fig.add_trace(
        go.Scatter(x=df['threshold'], y=df['recall'], mode='lines', name='Recall'),
        row=1, col=2
    )
    
    # F1 Score vs Threshold
    fig.add_trace(
        go.Scatter(x=df['threshold'], y=df['f1'], mode='lines', name='F1 Score'),
        row=2, col=1
    )
    
    # Precision vs Recall
    fig.add_trace(
        go.Scatter(x=df['recall'], y=df['precision'], mode='lines', name='Precision-Recall'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text=title or "Threshold Analysis",
        height=800,
        showlegend=False
    )
    
    return fig