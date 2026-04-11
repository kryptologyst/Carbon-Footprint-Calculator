#!/usr/bin/env python3
"""Main training script for carbon footprint prediction models."""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import CarbonFootprintDataGenerator
from src.models import ModelTrainer
from src.evaluation import ModelEvaluator


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train carbon footprint prediction models')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--output-dir', type=str, default='assets',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize components
    logger.info("Initializing data generator")
    data_generator = CarbonFootprintDataGenerator(config)
    
    logger.info("Initializing model trainer")
    model_trainer = ModelTrainer(config)
    
    logger.info("Initializing evaluator")
    evaluator = ModelEvaluator(config)
    
    # Generate dataset
    logger.info("Generating dataset")
    n_samples = config['data']['n_samples']
    X, y = data_generator.create_dataset(n_samples)
    
    # Split data
    logger.info("Splitting data")
    X_train, X_test, y_train, y_test = data_generator.split_data(X, y)
    
    # Scale features
    logger.info("Scaling features")
    X_train_scaled, X_test_scaled = data_generator.scale_features(X_train, X_test)
    
    # Train models
    logger.info("Training models")
    results = model_trainer.train_all_models(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Create leaderboard
    logger.info("Creating leaderboard")
    leaderboard = evaluator.create_leaderboard(results)
    print("\n" + "="*50)
    print("CARBON FOOTPRINT MODEL LEADERBOARD")
    print("="*50)
    print(leaderboard.to_string(index=False))
    print("="*50)
    
    # Save leaderboard
    leaderboard_path = output_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    logger.info(f"Leaderboard saved to {leaderboard_path}")
    
    # Generate visualizations
    logger.info("Generating visualizations")
    
    # Model comparison plot
    comparison_path = output_dir / "model_comparison.png"
    evaluator.plot_model_comparison(results, save_path=str(comparison_path))
    
    # Predictions vs actual for best model
    best_model_name = leaderboard.iloc[0]['Model'].lower().replace(' ', '_')
    if best_model_name in model_trainer.models:
        y_pred = model_trainer.models[best_model_name].predict(X_test_scaled)
        predictions_path = output_dir / f"{best_model_name}_predictions.png"
        evaluator.plot_predictions_vs_actual(
            y_test, y_pred, best_model_name, save_path=str(predictions_path)
        )
    
    # Feature importance plots
    feature_importance = {}
    for model_name in ['random_forest', 'xgboost']:
        if model_name in model_trainer.models:
            try:
                importance = model_trainer.get_feature_importance(model_name)
                feature_importance[model_name] = importance
                importance_path = output_dir / f"{model_name}_feature_importance.png"
                evaluator.plot_feature_importance(
                    importance, model_name, save_path=str(importance_path)
                )
            except ValueError:
                logger.warning(f"Feature importance not available for {model_name}")
    
    # Generate evaluation report
    logger.info("Generating evaluation report")
    report_path = output_dir / "evaluation_report.md"
    report = evaluator.generate_evaluation_report(
        results, feature_importance, save_path=str(report_path)
    )
    
    # Create interactive dashboard
    logger.info("Creating interactive dashboard")
    dashboard = evaluator.create_interactive_dashboard(results, feature_importance)
    dashboard_path = output_dir / "dashboard.html"
    dashboard.write_html(str(dashboard_path))
    logger.info(f"Interactive dashboard saved to {dashboard_path}")
    
    # Print summary
    logger.info("Training completed successfully!")
    logger.info(f"Results saved to {output_dir}")
    
    print(f"\nBest model: {leaderboard.iloc[0]['Model']}")
    print(f"Test RMSE: {leaderboard.iloc[0]['Test RMSE']:.2f} kg CO₂e")
    print(f"Test R²: {leaderboard.iloc[0]['Test R²']:.3f}")


if __name__ == "__main__":
    main()
