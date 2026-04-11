"""Evaluation and visualization module for carbon footprint models."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluator class for carbon footprint prediction models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.setup_plotting()
        
    def setup_plotting(self) -> None:
        """Setup plotting style and parameters."""
        plt.style.use(self.config['visualization']['style'])
        sns.set_palette(self.config['visualization']['color_palette'])
        
    def create_leaderboard(self, results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Create a model performance leaderboard.
        
        Args:
            results: Dictionary containing model results
            
        Returns:
            DataFrame containing leaderboard
        """
        logger.info("Creating model leaderboard")
        
        leaderboard_data = []
        for model_name, metrics in results.items():
            leaderboard_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Test RMSE': metrics['test_rmse'],
                'Test MAE': metrics['test_mae'],
                'Test R²': metrics['test_r2'],
                'Test SMAPE': metrics['test_smape']
            })
        
        leaderboard = pd.DataFrame(leaderboard_data)
        leaderboard = leaderboard.sort_values('Test RMSE').reset_index(drop=True)
        
        logger.info("Leaderboard created successfully")
        return leaderboard
    
    def plot_model_comparison(self, results: Dict[str, Dict[str, float]], save_path: str = None) -> None:
        """Plot model comparison charts.
        
        Args:
            results: Dictionary containing model results
            save_path: Path to save the plot
        """
        logger.info("Creating model comparison plots")
        
        fig, axes = plt.subplots(2, 2, figsize=self.config['visualization']['figure_size'])
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        models = list(results.keys())
        model_names = [name.replace('_', ' ').title() for name in models]
        
        # RMSE comparison
        rmse_values = [results[model]['test_rmse'] for model in models]
        axes[0, 0].bar(model_names, rmse_values, color='skyblue')
        axes[0, 0].set_title('Test RMSE Comparison')
        axes[0, 0].set_ylabel('RMSE (kg CO₂e)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        mae_values = [results[model]['test_mae'] for model in models]
        axes[0, 1].bar(model_names, mae_values, color='lightcoral')
        axes[0, 1].set_title('Test MAE Comparison')
        axes[0, 1].set_ylabel('MAE (kg CO₂e)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # R² comparison
        r2_values = [results[model]['test_r2'] for model in models]
        axes[1, 0].bar(model_names, r2_values, color='lightgreen')
        axes[1, 0].set_title('Test R² Comparison')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # SMAPE comparison
        smape_values = [results[model]['test_smape'] for model in models]
        axes[1, 1].bar(model_names, smape_values, color='gold')
        axes[1, 1].set_title('Test SMAPE Comparison')
        axes[1, 1].set_ylabel('SMAPE (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_predictions_vs_actual(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray, 
        model_name: str,
        save_path: str = None
    ) -> None:
        """Plot predictions vs actual values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Path to save the plot
        """
        logger.info(f"Creating predictions vs actual plot for {model_name}")
        
        plt.figure(figsize=self.config['visualization']['figure_size'])
        
        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.6, s=50)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        
        plt.xlabel('Actual Carbon Footprint (kg CO₂e)')
        plt.ylabel('Predicted Carbon Footprint (kg CO₂e)')
        plt.title(f'{model_name.replace("_", " ").title()} - Predictions vs Actual\nR² = {r2:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Predictions vs actual plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(
        self, 
        feature_importance: pd.Series, 
        model_name: str,
        save_path: str = None
    ) -> None:
        """Plot feature importance.
        
        Args:
            feature_importance: Series containing feature importance scores
            model_name: Name of the model
            save_path: Path to save the plot
        """
        logger.info(f"Creating feature importance plot for {model_name}")
        
        plt.figure(figsize=self.config['visualization']['figure_size'])
        
        # Sort features by importance
        sorted_features = feature_importance.sort_values(ascending=True)
        
        # Create horizontal bar plot
        bars = plt.barh(range(len(sorted_features)), sorted_features.values, color='steelblue')
        
        # Customize plot
        plt.yticks(range(len(sorted_features)), sorted_features.index)
        plt.xlabel('Feature Importance')
        plt.title(f'{model_name.replace("_", " ").title()} - Feature Importance')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def create_interactive_dashboard(
        self, 
        results: Dict[str, Dict[str, float]], 
        feature_importance: Dict[str, pd.Series] = None
    ) -> go.Figure:
        """Create an interactive dashboard using Plotly.
        
        Args:
            results: Dictionary containing model results
            feature_importance: Dictionary containing feature importance for each model
            
        Returns:
            Plotly figure object
        """
        logger.info("Creating interactive dashboard")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('RMSE Comparison', 'MAE Comparison', 
                          'R² Comparison', 'SMAPE Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        models = list(results.keys())
        model_names = [name.replace('_', ' ').title() for name in models]
        
        # Add traces for each metric
        fig.add_trace(
            go.Bar(x=model_names, y=[results[model]['test_rmse'] for model in models],
                   name='RMSE', marker_color='skyblue'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=model_names, y=[results[model]['test_mae'] for model in models],
                   name='MAE', marker_color='lightcoral'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=model_names, y=[results[model]['test_r2'] for model in models],
                   name='R²', marker_color='lightgreen'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=model_names, y=[results[model]['test_smape'] for model in models],
                   name='SMAPE', marker_color='gold'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Carbon Footprint Model Performance Dashboard",
            showlegend=False,
            height=800
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Models", row=1, col=1)
        fig.update_xaxes(title_text="Models", row=1, col=2)
        fig.update_xaxes(title_text="Models", row=2, col=1)
        fig.update_xaxes(title_text="Models", row=2, col=2)
        
        fig.update_yaxes(title_text="RMSE (kg CO₂e)", row=1, col=1)
        fig.update_yaxes(title_text="MAE (kg CO₂e)", row=1, col=2)
        fig.update_yaxes(title_text="R² Score", row=2, col=1)
        fig.update_yaxes(title_text="SMAPE (%)", row=2, col=2)
        
        logger.info("Interactive dashboard created successfully")
        return fig
    
    def generate_evaluation_report(
        self, 
        results: Dict[str, Dict[str, float]], 
        feature_importance: Dict[str, pd.Series] = None,
        save_path: str = None
    ) -> str:
        """Generate a comprehensive evaluation report.
        
        Args:
            results: Dictionary containing model results
            feature_importance: Dictionary containing feature importance for each model
            save_path: Path to save the report
            
        Returns:
            Report text
        """
        logger.info("Generating evaluation report")
        
        report = []
        report.append("# Carbon Footprint Model Evaluation Report\n")
        
        # Model performance summary
        report.append("## Model Performance Summary\n")
        leaderboard = self.create_leaderboard(results)
        report.append(leaderboard.to_string(index=False))
        report.append("\n")
        
        # Best model
        best_model = leaderboard.iloc[0]['Model']
        report.append(f"## Best Performing Model: {best_model}\n")
        report.append(f"- Test RMSE: {leaderboard.iloc[0]['Test RMSE']:.2f} kg CO₂e\n")
        report.append(f"- Test MAE: {leaderboard.iloc[0]['Test MAE']:.2f} kg CO₂e\n")
        report.append(f"- Test R²: {leaderboard.iloc[0]['Test R²']:.3f}\n")
        report.append(f"- Test SMAPE: {leaderboard.iloc[0]['Test SMAPE']:.2f}%\n\n")
        
        # Feature importance analysis
        if feature_importance:
            report.append("## Feature Importance Analysis\n")
            for model_name, importance in feature_importance.items():
                if not importance.empty:
                    report.append(f"### {model_name.replace('_', ' ').title()}\n")
                    sorted_features = importance.sort_values(ascending=False)
                    for feature, score in sorted_features.items():
                        report.append(f"- {feature}: {score:.3f}\n")
                    report.append("\n")
        
        # Recommendations
        report.append("## Recommendations\n")
        report.append("- Consider ensemble methods to combine the best performing models\n")
        report.append("- Collect more diverse data to improve model generalization\n")
        report.append("- Implement feature engineering to capture non-linear relationships\n")
        report.append("- Use cross-validation for more robust model evaluation\n")
        
        report_text = "".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved to {save_path}")
        
        return report_text
