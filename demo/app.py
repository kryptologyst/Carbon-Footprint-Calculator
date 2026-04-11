"""Streamlit demo for carbon footprint prediction."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import CarbonFootprintDataGenerator
from models import ModelTrainer
from evaluation import ModelEvaluator


def load_config():
    """Load configuration."""
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_trained_models():
    """Load pre-trained models (placeholder for demo)."""
    config = load_config()
    
    # Generate a small dataset for demo
    data_generator = CarbonFootprintDataGenerator(config)
    X, y = data_generator.create_dataset(1000)
    X_train, X_test, y_train, y_test = data_generator.split_data(X, y)
    X_train_scaled, X_test_scaled = data_generator.scale_features(X_train, X_test)
    
    # Train models
    model_trainer = ModelTrainer(config)
    results = model_trainer.train_all_models(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    return model_trainer, data_generator, results


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Carbon Footprint Calculator",
        page_icon="🌱",
        layout="wide"
    )
    
    st.title("🌱 Carbon Footprint Calculator")
    st.markdown("""
    This interactive tool helps you estimate your monthly carbon footprint based on your lifestyle choices.
    The model uses machine learning to predict CO₂ emissions from various activities.
    """)
    
    # Load models
    with st.spinner("Loading models..."):
        model_trainer, data_generator, results = load_trained_models()
    
    # Sidebar for user input
    st.sidebar.header("Your Lifestyle Data")
    
    miles_driven = st.sidebar.slider(
        "Miles Driven (km/month)",
        min_value=0,
        max_value=2000,
        value=800,
        help="Average monthly distance driven"
    )
    
    electricity_usage = st.sidebar.slider(
        "Electricity Usage (kWh/month)",
        min_value=0,
        max_value=1000,
        value=300,
        help="Monthly electricity consumption"
    )
    
    meat_consumption = st.sidebar.slider(
        "Meat Consumption (kg/month)",
        min_value=0,
        max_value=50,
        value=15,
        help="Monthly meat consumption"
    )
    
    flights = st.sidebar.slider(
        "Number of Flights (per month)",
        min_value=0,
        max_value=10,
        value=0,
        help="Number of flights taken per month"
    )
    
    recycling_rate = st.sidebar.slider(
        "Recycling Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Proportion of waste recycled (0-1)"
    )
    
    # Create input data
    user_input = pd.DataFrame({
        'miles_driven': [miles_driven],
        'electricity_usage': [electricity_usage],
        'meat_consumption': [meat_consumption],
        'flights': [flights],
        'recycling_rate': [recycling_rate]
    })
    
    # Scale input data
    user_input_scaled = pd.DataFrame(
        data_generator.scaler.transform(user_input),
        columns=user_input.columns
    )
    
    # Make predictions
    st.header("Your Carbon Footprint Prediction")
    
    col1, col2, col3, col4 = st.columns(4)
    
    predictions = {}
    for model_name, model in model_trainer.models.items():
        if hasattr(model, 'predict'):
            pred = model.predict(user_input_scaled)[0]
            predictions[model_name] = pred
    
    # Display predictions
    if predictions:
        avg_prediction = np.mean(list(predictions.values()))
        
        with col1:
            st.metric(
                "Average Prediction",
                f"{avg_prediction:.1f} kg CO₂e",
                help="Average prediction across all models"
            )
        
        with col2:
            st.metric(
                "Best Model",
                f"{min(predictions.values()):.1f} kg CO₂e",
                help="Prediction from best performing model"
            )
        
        with col3:
            st.metric(
                "Range",
                f"{min(predictions.values()):.1f} - {max(predictions.values()):.1f} kg CO₂e",
                help="Range of predictions across models"
            )
        
        with col4:
            # Calculate percentile
            percentile = np.percentile([avg_prediction], 50)
            st.metric(
                "Percentile",
                f"50th",
                help="Your carbon footprint percentile"
            )
    
    # Model comparison
    st.header("Model Performance Comparison")
    
    # Create leaderboard
    evaluator = ModelEvaluator(load_config())
    leaderboard = evaluator.create_leaderboard(results)
    
    st.dataframe(leaderboard, use_container_width=True)
    
    # Visualization
    st.header("Model Performance Visualization")
    
    # Create interactive plot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('RMSE Comparison', 'MAE Comparison', 
                      'R² Comparison', 'SMAPE Comparison')
    )
    
    models = list(results.keys())
    model_names = [name.replace('_', ' ').title() for name in models]
    
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
    
    fig.update_layout(
        title_text="Model Performance Dashboard",
        showlegend=False,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.header("Feature Importance Analysis")
    
    feature_importance_data = {}
    for model_name in ['random_forest', 'xgboost']:
        if model_name in model_trainer.models:
            try:
                importance = model_trainer.get_feature_importance(model_name)
                feature_importance_data[model_name] = importance
            except ValueError:
                continue
    
    if feature_importance_data:
        # Create feature importance plot
        fig_importance = go.Figure()
        
        for model_name, importance in feature_importance_data.items():
            fig_importance.add_trace(go.Bar(
                x=importance.index,
                y=importance.values,
                name=model_name.replace('_', ' ').title(),
                orientation='v'
            ))
        
        fig_importance.update_layout(
            title="Feature Importance Comparison",
            xaxis_title="Features",
            yaxis_title="Importance Score",
            barmode='group'
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Recommendations
    st.header("Recommendations to Reduce Your Carbon Footprint")
    
    recommendations = []
    
    if miles_driven > 1000:
        recommendations.append("🚗 Consider carpooling or using public transportation to reduce driving")
    
    if electricity_usage > 400:
        recommendations.append("⚡ Switch to energy-efficient appliances and LED bulbs")
    
    if meat_consumption > 20:
        recommendations.append("🥩 Reduce meat consumption and try plant-based alternatives")
    
    if flights > 2:
        recommendations.append("✈️ Consider video conferencing instead of frequent flights")
    
    if recycling_rate < 0.7:
        recommendations.append("♻️ Increase recycling efforts and reduce waste generation")
    
    if not recommendations:
        recommendations.append("🌱 Great job! Your lifestyle choices are already quite sustainable!")
    
    for rec in recommendations:
        st.write(rec)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Disclaimer**: This tool is for educational and research purposes only. 
    Carbon footprint calculations are estimates based on simplified models and may not reflect 
    actual emissions. For accurate carbon footprint assessment, consult with environmental 
    professionals or use certified carbon accounting tools.
    
    **Author**: [kryptologyst](https://github.com/kryptologyst) | 
    **Project**: Carbon Footprint Calculator
    """)


if __name__ == "__main__":
    main()
