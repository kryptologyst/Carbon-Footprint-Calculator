"""Unit tests for carbon footprint calculator."""

import pytest
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import CarbonFootprintDataGenerator
from models import ModelTrainer, CarbonFootprintNeuralNetwork
from evaluation import ModelEvaluator


@pytest.fixture
def config():
    """Load test configuration."""
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def data_generator(config):
    """Create data generator instance."""
    return CarbonFootprintDataGenerator(config, random_seed=42)


@pytest.fixture
def sample_data(data_generator):
    """Generate sample data for testing."""
    return data_generator.create_dataset(100)


class TestDataGenerator:
    """Test data generation functionality."""
    
    def test_data_generator_initialization(self, config):
        """Test data generator initialization."""
        generator = CarbonFootprintDataGenerator(config, random_seed=42)
        assert generator.config == config
        assert generator.random_seed == 42
    
    def test_generate_features(self, data_generator):
        """Test feature generation."""
        features = data_generator.generate_features(100)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == 100
        assert list(features.columns) == [
            'miles_driven', 'electricity_usage', 'meat_consumption', 
            'flights', 'recycling_rate'
        ]
        
        # Check non-negative values
        assert (features['miles_driven'] >= 0).all()
        assert (features['electricity_usage'] >= 0).all()
        assert (features['meat_consumption'] >= 0).all()
        assert (features['flights'] >= 0).all()
        assert (features['recycling_rate'] >= 0).all()
        assert (features['recycling_rate'] <= 1).all()
    
    def test_generate_target(self, data_generator):
        """Test target generation."""
        features = data_generator.generate_features(100)
        targets = data_generator.generate_target(features)
        
        assert isinstance(targets, pd.Series)
        assert len(targets) == 100
        assert targets.name == 'carbon_footprint'
        assert (targets >= 0).all()
    
    def test_create_dataset(self, data_generator):
        """Test complete dataset creation."""
        X, y = data_generator.create_dataset(100)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y) == 100
        assert X.shape[1] == 5
        assert y.name == 'carbon_footprint'
    
    def test_split_data(self, data_generator, sample_data):
        """Test data splitting."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = data_generator.split_data(X, y)
        
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
    
    def test_scale_features(self, data_generator, sample_data):
        """Test feature scaling."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = data_generator.split_data(X, y)
        X_train_scaled, X_test_scaled = data_generator.scale_features(X_train, X_test)
        
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape
        assert list(X_train_scaled.columns) == list(X_train.columns)
        assert list(X_test_scaled.columns) == list(X_test.columns)


class TestModelTrainer:
    """Test model training functionality."""
    
    def test_model_trainer_initialization(self, config):
        """Test model trainer initialization."""
        trainer = ModelTrainer(config)
        assert trainer.config == config
        assert trainer.device in ['cuda', 'mps', 'cpu']
    
    def test_neural_network_initialization(self):
        """Test neural network initialization."""
        model = CarbonFootprintNeuralNetwork(input_size=5, hidden_sizes=[64, 32])
        
        # Test forward pass
        x = np.random.randn(10, 5)
        output = model(torch.FloatTensor(x))
        
        assert output.shape == (10, 1)
    
    def test_linear_regression_training(self, config, sample_data):
        """Test linear regression training."""
        trainer = ModelTrainer(config)
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        X_train_scaled, X_test_scaled = trainer.scale_features(X_train, X_test)
        
        metrics = trainer.train_linear_regression(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        
        assert 'train_rmse' in metrics
        assert 'test_rmse' in metrics
        assert 'train_r2' in metrics
        assert 'test_r2' in metrics
        assert metrics['test_rmse'] > 0
        assert metrics['test_r2'] <= 1.0
    
    def test_random_forest_training(self, config, sample_data):
        """Test random forest training."""
        trainer = ModelTrainer(config)
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        X_train_scaled, X_test_scaled = trainer.scale_features(X_train, X_test)
        
        metrics = trainer.train_random_forest(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        
        assert 'train_rmse' in metrics
        assert 'test_rmse' in metrics
        assert metrics['test_rmse'] > 0
        assert 'random_forest' in trainer.models
    
    def test_xgboost_training(self, config, sample_data):
        """Test XGBoost training."""
        trainer = ModelTrainer(config)
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        X_train_scaled, X_test_scaled = trainer.scale_features(X_train, X_test)
        
        metrics = trainer.train_xgboost(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        
        assert 'train_rmse' in metrics
        assert 'test_rmse' in metrics
        assert metrics['test_rmse'] > 0
        assert 'xgboost' in trainer.models
    
    def test_feature_importance(self, config, sample_data):
        """Test feature importance extraction."""
        trainer = ModelTrainer(config)
        X, y = sample_data
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        X_train_scaled, X_test_scaled = trainer.scale_features(X_train, X_test)
        
        # Train a tree-based model
        trainer.train_random_forest(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        
        importance = trainer.get_feature_importance('random_forest')
        assert isinstance(importance, pd.Series)
        assert len(importance) == 5
        assert (importance >= 0).all()
        assert abs(importance.sum() - 1.0) < 1e-6  # Should sum to 1


class TestModelEvaluator:
    """Test evaluation functionality."""
    
    def test_evaluator_initialization(self, config):
        """Test evaluator initialization."""
        evaluator = ModelEvaluator(config)
        assert evaluator.config == config
    
    def test_create_leaderboard(self, config):
        """Test leaderboard creation."""
        evaluator = ModelEvaluator(config)
        
        # Mock results
        results = {
            'linear_regression': {
                'test_rmse': 45.2,
                'test_mae': 35.8,
                'test_r2': 0.89,
                'test_smape': 12.3
            },
            'random_forest': {
                'test_rmse': 42.1,
                'test_mae': 33.2,
                'test_r2': 0.91,
                'test_smape': 11.8
            }
        }
        
        leaderboard = evaluator.create_leaderboard(results)
        
        assert isinstance(leaderboard, pd.DataFrame)
        assert len(leaderboard) == 2
        assert 'Model' in leaderboard.columns
        assert 'Test RMSE' in leaderboard.columns
        assert leaderboard.iloc[0]['Test RMSE'] == 42.1  # Should be sorted by RMSE
    
    def test_calculate_metrics(self, config):
        """Test metrics calculation."""
        evaluator = ModelEvaluator(config)
        
        y_true = pd.Series([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])
        
        metrics = evaluator._calculate_metrics(y_true, y_pred, y_true, y_pred)
        
        assert 'train_rmse' in metrics
        assert 'test_rmse' in metrics
        assert 'train_r2' in metrics
        assert 'test_r2' in metrics
        assert metrics['test_rmse'] > 0
        assert metrics['test_r2'] > 0


if __name__ == "__main__":
    pytest.main([__file__])
