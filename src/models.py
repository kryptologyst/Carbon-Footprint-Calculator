"""Machine learning models for carbon footprint prediction."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging

logger = logging.getLogger(__name__)


class CarbonFootprintNeuralNetwork(nn.Module):
    """Neural network for carbon footprint prediction."""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], dropout_rate: float = 0.2):
        """Initialize the neural network.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class ModelTrainer:
    """Trainer class for carbon footprint prediction models."""
    
    def __init__(self, config: Dict[str, Any], device: str = None):
        """Initialize the model trainer.
        
        Args:
            config: Configuration dictionary
            device: Device to use for PyTorch models ('cuda', 'mps', or 'cpu')
        """
        self.config = config
        self.device = device or self._get_device()
        self.models = {}
        self.results = {}
        
        logger.info(f"Using device: {self.device}")
        
    def _get_device(self) -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        test_size = self.config['data']['test_size']
        random_seed = self.config['data']['random_seed']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed
        )
        
        logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale features using StandardScaler.
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            Tuple of scaled (X_train, X_test)
        """
        from sklearn.preprocessing import StandardScaler
        
        logger.info("Scaling features")
        
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        logger.info("Feature scaling completed")
        return X_train_scaled, X_test_scaled
    
    def train_linear_regression(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Train and evaluate linear regression model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Training Linear Regression model")
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
        
        self.models['linear_regression'] = model
        self.results['linear_regression'] = metrics
        
        logger.info(f"Linear Regression RMSE: {metrics['test_rmse']:.2f}")
        return metrics
    
    def train_random_forest(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Train and evaluate random forest model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Training Random Forest model")
        
        model_config = self.config['models']['random_forest']
        model = RandomForestRegressor(
            n_estimators=model_config['n_estimators'],
            max_depth=model_config['max_depth'],
            random_state=self.config['data']['random_seed']
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
        
        self.models['random_forest'] = model
        self.results['random_forest'] = metrics
        
        logger.info(f"Random Forest RMSE: {metrics['test_rmse']:.2f}")
        return metrics
    
    def train_xgboost(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Train and evaluate XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Training XGBoost model")
        
        model_config = self.config['models']['xgboost']
        model = xgb.XGBRegressor(
            n_estimators=model_config['n_estimators'],
            max_depth=model_config['max_depth'],
            learning_rate=model_config['learning_rate'],
            random_state=self.config['data']['random_seed']
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
        
        self.models['xgboost'] = model
        self.results['xgboost'] = metrics
        
        logger.info(f"XGBoost RMSE: {metrics['test_rmse']:.2f}")
        return metrics
    
    def train_neural_network(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Train and evaluate neural network model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Training Neural Network model")
        
        model_config = self.config['models']['neural_network']
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
        X_test_tensor = torch.FloatTensor(X_test.values)
        y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1)
        
        # Move to device
        X_train_tensor = X_train_tensor.to(self.device)
        y_train_tensor = y_train_tensor.to(self.device)
        X_test_tensor = X_test_tensor.to(self.device)
        y_test_tensor = y_test_tensor.to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=model_config['batch_size'], 
            shuffle=True
        )
        
        # Initialize model
        model = CarbonFootprintNeuralNetwork(
            input_size=X_train.shape[1],
            hidden_sizes=model_config['hidden_layers']
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=model_config['learning_rate']
        )
        
        # Training loop
        model.train()
        for epoch in range(model_config['epochs']):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{model_config['epochs']}, Loss: {total_loss/len(train_loader):.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            y_pred_train = model(X_train_tensor).cpu().numpy().flatten()
            y_pred_test = model(X_test_tensor).cpu().numpy().flatten()
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
        
        self.models['neural_network'] = model
        self.results['neural_network'] = metrics
        
        logger.info(f"Neural Network RMSE: {metrics['test_rmse']:.2f}")
        return metrics
    
    def _calculate_metrics(
        self, 
        y_train: pd.Series, 
        y_pred_train: np.ndarray,
        y_test: pd.Series, 
        y_pred_test: np.ndarray
    ) -> Dict[str, float]:
        """Calculate evaluation metrics.
        
        Args:
            y_train: Training targets
            y_pred_train: Training predictions
            y_test: Test targets
            y_pred_test: Test predictions
            
        Returns:
            Dictionary containing evaluation metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        def smape(y_true, y_pred):
            """Symmetric Mean Absolute Percentage Error."""
            return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
        
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_smape': smape(y_train, y_pred_train),
            'test_smape': smape(y_test, y_pred_test),
        }
        
        return metrics
    
    def train_all_models(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """Train all enabled models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary containing results for all models
        """
        logger.info("Training all enabled models")
        
        models_config = self.config['models']
        
        if models_config['linear_regression']['enabled']:
            self.train_linear_regression(X_train, y_train, X_test, y_test)
        
        if models_config['random_forest']['enabled']:
            self.train_random_forest(X_train, y_train, X_test, y_test)
        
        if models_config['xgboost']['enabled']:
            self.train_xgboost(X_train, y_train, X_test, y_test)
        
        if models_config['neural_network']['enabled']:
            self.train_neural_network(X_train, y_train, X_test, y_test)
        
        logger.info("All models trained successfully")
        return self.results
    
    def get_feature_importance(self, model_name: str) -> pd.Series:
        """Get feature importance for tree-based models.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Series containing feature importance scores
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            return pd.Series(
                model.feature_importances_,
                index=self.models[model_name].feature_names_in_ if hasattr(model, 'feature_names_in_') else None
            )
        else:
            raise ValueError(f"Model {model_name} does not support feature importance")
