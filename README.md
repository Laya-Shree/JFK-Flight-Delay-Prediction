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
  - XGBoost Classifier to predict delay categories (on-time, short delay[<1hr], long delay[>1hr])
  - Target encoding for categorical variables
  - Addressed class imbalance with SMOTE oversampling
- **Regression:**
  - Random Forest Regressor and XGBoost Regressor to predict delay in minutes
  - LightGBM Regressor for comparison
  - Feature importance analysis

## Results
- **XGBoost Classifier:**
  - Achieved ~70% accuracy on test data for multi-class delay prediction
  - Best performance for predicting 'on-time' and 'long delay' classes
- **Random Forest Regressor:**
  - Mean Absolute Error (MAE): ~12.3 minutes
  - R² Score: ~0.40
- **XGBoost Regressor:**
  - R² Score: up to ~0.69 after outlier removal and feature engineering


## Project Structure
- `data.csv` — Main dataset
- `notebooks/` — Jupyter notebooks for EDA and modeling
- `models/` — (Reserved for model files)
- `requirements.txt` — Python dependencies

## Acknowledgements
- Data and project inspired by real-world flight delay prediction challenges. 
