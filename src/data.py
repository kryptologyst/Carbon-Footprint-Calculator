"""Data generation and preprocessing module for carbon footprint calculation."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class CarbonFootprintDataGenerator:
    """Generate synthetic carbon footprint data for training and testing."""
    
    def __init__(self, config: Dict[str, Any], random_seed: int = 42):
        """Initialize the data generator.
        
        Args:
            config: Configuration dictionary containing data parameters
            random_seed: Random seed for reproducibility
        """
        self.config = config
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        
        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        
    def generate_features(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic lifestyle features.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame containing generated features
        """
        logger.info(f"Generating {n_samples} samples of lifestyle data")
        
        # Extract feature parameters from config
        features_config = self.config['features']
        
        # Generate features based on config
        data = {
            'miles_driven': np.random.normal(
                features_config['miles_driven']['mean'],
                features_config['miles_driven']['std'],
                n_samples
            ),
            'electricity_usage': np.random.normal(
                features_config['electricity_usage']['mean'],
                features_config['electricity_usage']['std'],
                n_samples
            ),
            'meat_consumption': np.random.normal(
                features_config['meat_consumption']['mean'],
                features_config['meat_consumption']['std'],
                n_samples
            ),
            'flights': np.random.poisson(
                features_config['flights']['lambda'],
                n_samples
            ),
            'recycling_rate': np.random.uniform(
                features_config['recycling_rate']['min'],
                features_config['recycling_rate']['max'],
                n_samples
            )
        }
        
        df = pd.DataFrame(data)
        
        # Ensure non-negative values for realistic data
        df['miles_driven'] = np.maximum(df['miles_driven'], 0)
        df['electricity_usage'] = np.maximum(df['electricity_usage'], 0)
        df['meat_consumption'] = np.maximum(df['meat_consumption'], 0)
        df['flights'] = np.maximum(df['flights'], 0)
        
        logger.info("Feature generation completed")
        return df
    
    def generate_target(self, features: pd.DataFrame) -> pd.Series:
        """Generate carbon footprint targets based on features.
        
        Args:
            features: DataFrame containing lifestyle features
            
        Returns:
            Series containing carbon footprint values in kg CO₂e
        """
        logger.info("Generating carbon footprint targets")
        
        # Realistic carbon footprint calculation with noise
        carbon_footprint = (
            0.2 * features['miles_driven'] +      # Transportation emissions
            0.5 * features['electricity_usage'] +  # Energy consumption
            27 * features['meat_consumption'] +    # Diet emissions
            250 * features['flights'] -            # Flight emissions
            100 * features['recycling_rate'] +     # Recycling offset
            np.random.normal(0, 50, len(features))  # Random noise
        )
        
        # Ensure non-negative carbon footprint
        carbon_footprint = np.maximum(carbon_footprint, 0)
        
        logger.info("Target generation completed")
        return pd.Series(carbon_footprint, name='carbon_footprint')
    
    def create_dataset(self, n_samples: int) -> Tuple[pd.DataFrame, pd.Series]:
        """Create complete dataset with features and targets.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of (features_df, targets_series)
        """
        logger.info(f"Creating dataset with {n_samples} samples")
        
        features = self.generate_features(n_samples)
        targets = self.generate_target(features)
        
        logger.info("Dataset creation completed")
        return features, targets
    
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
        logger.info("Scaling features")
        
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        logger.info("Feature scaling completed")
        return X_train_scaled, X_test_scaled
    
    def get_feature_info(self) -> Dict[str, str]:
        """Get feature information and units.
        
        Returns:
            Dictionary mapping feature names to their units
        """
        return {
            feature: config['unit'] 
            for feature, config in self.config['features'].items()
        }
