"""Carbon Footprint Calculator package."""

from .data import CarbonFootprintDataGenerator
from .models import ModelTrainer, CarbonFootprintNeuralNetwork
from .evaluation import ModelEvaluator

__version__ = "1.0.0"
__author__ = "kryptologyst"

__all__ = [
    "CarbonFootprintDataGenerator",
    "ModelTrainer", 
    "CarbonFootprintNeuralNetwork",
    "ModelEvaluator"
]
