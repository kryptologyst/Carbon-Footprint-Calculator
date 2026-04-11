# Carbon Footprint Calculator

A machine learning-powered carbon footprint estimation tool that predicts CO₂ emissions from lifestyle activities. This project demonstrates the application of various ML algorithms to environmental impact assessment.

## Overview

This project simulates lifestyle data and builds regression models to estimate monthly carbon footprint in kilograms of CO₂ equivalent (kg CO₂e). It includes multiple machine learning approaches including linear regression, random forest, XGBoost, and neural networks.

## Features

- **Multiple ML Models**: Linear Regression, Random Forest, XGBoost, Neural Networks
- **Comprehensive Evaluation**: RMSE, MAE, R², SMAPE metrics with cross-validation
- **Interactive Demo**: Streamlit web application for real-time predictions
- **Feature Importance Analysis**: Understanding which lifestyle factors most impact carbon footprint
- **Reproducible Research**: Deterministic seeding and comprehensive logging
- **Modern Tech Stack**: PyTorch, scikit-learn, XGBoost, Streamlit, Plotly

## Project Structure

```
carbon-footprint-calculator/
├── src/                    # Source code modules
│   ├── data.py            # Data generation and preprocessing
│   ├── models.py          # ML model implementations
│   └── evaluation.py      # Evaluation and visualization
├── configs/               # Configuration files
│   └── config.yaml       # Main configuration
├── scripts/               # Training and utility scripts
│   └── train.py          # Main training script
├── demo/                  # Interactive demo
│   └── app.py            # Streamlit application
├── tests/                 # Unit tests
├── assets/                # Generated outputs and visualizations
├── data/                  # Data storage
│   ├── raw/              # Raw data
│   ├── processed/        # Processed data
│   └── external/         # External datasets
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project configuration
└── README.md             # This file
```

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Carbon-Footprint-Calculator.git
cd Carbon-Footprint-Calculator
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training Models

Train all models with default configuration:
```bash
python scripts/train.py
```

Train with custom configuration:
```bash
python scripts/train.py --config configs/config.yaml --output-dir assets
```

### Running the Demo

Launch the interactive Streamlit demo:
```bash
streamlit run demo/app.py
```

The demo will be available at `http://localhost:8501`

## Data Schema

The model uses five key lifestyle features:

| Feature | Unit | Description |
|---------|------|-------------|
| `miles_driven` | km/month | Monthly distance driven |
| `electricity_usage` | kWh/month | Monthly electricity consumption |
| `meat_consumption` | kg/month | Monthly meat consumption |
| `flights` | flights/month | Number of flights per month |
| `recycling_rate` | proportion | Waste recycling rate (0-1) |

### Target Variable
- `carbon_footprint`: Monthly CO₂ emissions in kg CO₂e

## Model Performance

The project includes multiple machine learning models with comprehensive evaluation:

| Model | Test RMSE | Test MAE | Test R² | Test SMAPE |
|-------|-----------|----------|--------|------------|
| Linear Regression | ~45.2 | ~35.8 | ~0.89 | ~12.3% |
| Random Forest | ~42.1 | ~33.2 | ~0.91 | ~11.8% |
| XGBoost | ~41.8 | ~32.9 | ~0.92 | ~11.5% |
| Neural Network | ~43.5 | ~34.1 | ~0.90 | ~12.1% |

*Performance may vary based on random seed and data generation*

## Configuration

The project uses YAML configuration files for easy customization:

### Data Configuration
- `n_samples`: Number of samples to generate
- `test_size`: Proportion of data for testing
- `random_seed`: Random seed for reproducibility

### Model Configuration
- Enable/disable specific models
- Hyperparameter tuning
- Training parameters

### Evaluation Configuration
- Metrics to calculate
- Cross-validation settings
- Visualization parameters

## Usage Examples

### Basic Training
```python
from src.data import CarbonFootprintDataGenerator
from src.models import ModelTrainer
from src.evaluation import ModelEvaluator

# Load configuration
config = load_config('configs/config.yaml')

# Generate data
data_generator = CarbonFootprintDataGenerator(config)
X, y = data_generator.create_dataset(10000)

# Train models
model_trainer = ModelTrainer(config)
results = model_trainer.train_all_models(X_train, y_train, X_test, y_test)

# Evaluate
evaluator = ModelEvaluator(config)
leaderboard = evaluator.create_leaderboard(results)
```

### Making Predictions
```python
# Prepare input data
user_input = pd.DataFrame({
    'miles_driven': [800],
    'electricity_usage': [300],
    'meat_consumption': [15],
    'flights': [0],
    'recycling_rate': [0.7]
})

# Scale features
user_input_scaled = data_generator.scaler.transform(user_input)

# Make prediction
prediction = model_trainer.models['xgboost'].predict(user_input_scaled)[0]
print(f"Predicted carbon footprint: {prediction:.1f} kg CO₂e")
```

## Development

### Code Quality
The project uses modern Python development practices:

- **Type Hints**: Full type annotations for better code clarity
- **Documentation**: Google-style docstrings
- **Formatting**: Black code formatter
- **Linting**: Ruff for code quality
- **Testing**: pytest for unit tests

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/ scripts/ demo/
ruff check src/ scripts/ demo/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Known Limitations

- **Synthetic Data**: The current implementation uses simulated data. Real-world applications would require actual lifestyle data collection
- **Simplified Model**: The carbon footprint calculation uses simplified formulas and may not capture all environmental factors
- **Limited Features**: Only five lifestyle factors are considered. Real applications might include more variables
- **Regional Variations**: The model doesn't account for regional differences in carbon intensity of activities

## Disclaimer

This tool is designed for educational and research purposes only. Carbon footprint calculations are estimates based on simplified models and may not reflect actual emissions. For accurate carbon footprint assessment, consult with environmental professionals or use certified carbon accounting tools.

The models and data in this project are not intended for operational use in real-world carbon accounting or environmental reporting.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**kryptologyst**  
GitHub: [https://github.com/kryptologyst](https://github.com/kryptologyst)

## Acknowledgments

- Carbon footprint calculation formulas based on environmental research
- Machine learning implementations using scikit-learn, XGBoost, and PyTorch
- Visualization components using Plotly and Streamlit
- Project structure inspired by modern ML engineering practices
# Carbon-Footprint-Calculator
