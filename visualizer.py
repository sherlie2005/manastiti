import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def plot_model_comparisons(results_dict):
    """Create visualization comparing model performances across different train-test splits."""
    # Prepare data for plotting
    data = []
    for split, results in results_dict.items():
        for model, prob in results.items():
            data.append({
                'Split Ratio': split,
                'Model': model,
                'Probability': prob * 100  # Convert to percentage
            })
    
    df = pd.DataFrame(data)
    
    # Create heatmap
    fig = px.imshow(
        df.pivot(index='Model', columns='Split Ratio', values='Probability'),
        labels=dict(x='Train-Test Split', y='Model', color='Probability (%)'),
        aspect='auto',
        color_continuous_scale='RdYlBu'
    )
    
    fig.update_layout(
        title='Model Performance Across Different Train-Test Splits',
        xaxis_title='Train-Test Split Ratio',
        yaxis_title='Model',
        height=600
    )
    
    return fig

def plot_feature_importance(feature_names, importances):
    """Create bar plot of feature importances."""
    fig = go.Figure(data=[
        go.Bar(
            x=importances,
            y=feature_names,
            orientation='h'
        )
    ])
    
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        height=500
    )
    
    return fig
