#!/usr/bin/env python3
"""Quick demo script for carbon footprint calculator."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import CarbonFootprintDataGenerator
from src.models import ModelTrainer
from src.evaluation import ModelEvaluator
import yaml


def main():
    """Run a quick demo of the carbon footprint calculator."""
    print("🌱 Carbon Footprint Calculator - Quick Demo")
    print("=" * 50)
    
    # Load configuration
    config_path = Path(__file__).parent / "configs" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate smaller dataset for demo
    config['data']['n_samples'] = 1000
    
    # Initialize components
    data_generator = CarbonFootprintDataGenerator(config)
    model_trainer = ModelTrainer(config)
    evaluator = ModelEvaluator(config)
    
    print("📊 Generating dataset...")
    X, y = data_generator.create_dataset(config['data']['n_samples'])
    
    print("🔄 Splitting and scaling data...")
    X_train, X_test, y_train, y_test = data_generator.split_data(X, y)
    X_train_scaled, X_test_scaled = data_generator.scale_features(X_train, X_test)
    
    print("🤖 Training models...")
    # Train only faster models for demo
    config['models']['neural_network']['enabled'] = False
    config['models']['neural_network']['epochs'] = 10
    
    results = model_trainer.train_all_models(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    print("\n📈 Model Performance:")
    print("-" * 30)
    leaderboard = evaluator.create_leaderboard(results)
    print(leaderboard.to_string(index=False))
    
    print("\n🎯 Example Predictions:")
    print("-" * 30)
    
    # Example lifestyle profiles
    profiles = {
        'Eco-conscious': [200, 150, 5, 0, 0.9],
        'Average': [800, 300, 15, 0.5, 0.5],
        'High-impact': [1500, 500, 30, 2, 0.2]
    }
    
    feature_names = ['miles_driven', 'electricity_usage', 'meat_consumption', 'flights', 'recycling_rate']
    
    for profile_name, values in profiles.items():
        profile_df = pd.DataFrame([values], columns=feature_names)
        profile_scaled = pd.DataFrame(
            data_generator.scaler.transform(profile_df),
            columns=feature_names
        )
        
        # Get prediction from best model
        best_model_name = leaderboard.iloc[0]['Model'].lower().replace(' ', '_')
        if best_model_name in model_trainer.models:
            prediction = model_trainer.models[best_model_name].predict(profile_scaled)[0]
            print(f"{profile_name:15}: {prediction:.1f} kg CO₂e")
    
    print("\n✨ Demo completed successfully!")
    print("🚀 Run 'streamlit run demo/app.py' for the interactive demo")
    print("📚 See README.md for full documentation")


if __name__ == "__main__":
    main()
