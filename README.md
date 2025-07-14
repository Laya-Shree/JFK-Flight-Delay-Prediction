# JFK Flight Delay Prediction

## Introduction
This project aims to predict flight delays at JFK Airport using machine learning techniques. The goal is to analyze historical flight and weather data to build models that can classify and/or regress the expected delay of a flight, helping airlines and passengers anticipate disruptions.

## Dataset
- **Source:** The dataset (`data.csv`) contains flight and weather information for flights departing from JFK.
- **Features include:**
  - Flight date, scheduled and actual departure/arrival times
  - Carrier, destination, distance
  - Weather features: temperature, humidity, wind speed/gust/direction, pressure, dew point, and weather condition
  - Delay information (departure delay in minutes)

## Exploratory Data Analysis (EDA)
- Analyzed delay distributions by hour of day and day of week
- Visualized delay distributions and outliers
- Explored relationships between delay and weather/flight features
- Outlier removal and feature engineering (e.g., converting time columns, creating categorical delay classes)

## Modeling Approaches
- **Classification:**
  - XGBoost Classifier to predict delay categories (early, on-time, short delay, long delay)
  - Target encoding for categorical variables
  - Addressed class imbalance with SMOTE oversampling
- **Regression:**
  - Random Forest Regressor and XGBoost Regressor to predict delay in minutes
  - LightGBM Regressor for comparison
  - Feature importance analysis

## Results
- **XGBoost Classifier:**
  - Achieved ~70% accuracy on test data for multi-class delay prediction
  - Best performance for predicting 'on-time' and 'early' classes
- **Random Forest Regressor:**
  - Mean Absolute Error (MAE): ~12.3 minutes
  - R² Score: ~0.40
- **XGBoost Regressor:**
  - R² Score: up to ~0.72 after outlier removal and feature engineering

## How to Run
1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run notebooks:**
   - Open the notebooks in the `notebooks/` directory (e.g., `eda.ipynb`, `XGB_classifier.ipynb`, `RF_code.ipynb`, `XGB_code.ipynb`) using Jupyter Notebook or JupyterLab.
   - Follow the cells for EDA, preprocessing, model training, and evaluation.
3. **Models:**
   - Pretrained model files (`.pkl`) are available in the `notebooks/` directory for XGBoost classifiers.

## Requirements
See `requirements.txt` for the full list. Key packages:
- pandas, numpy, matplotlib, seaborn, plotly
- scikit-learn, xgboost, lightgbm, imbalanced-learn, category_encoders
- jupyter, ipykernel

## Project Structure
- `data.csv` — Main dataset
- `notebooks/` — Jupyter notebooks for EDA and modeling
- `models/` — (Reserved for model files)
- `requirements.txt` — Python dependencies

## Acknowledgements
- Data and project inspired by real-world flight delay prediction challenges. 
