# Crop Yield Prediction

A machine learning web app that predicts crop yield (hg/ha) from country, crop, year, rainfall, pesticides and temperature.

## Dataset
Crop Yield Prediction Dataset (Kaggle), about 28,000 rows. File: `datasets/yield_df.csv`

## Model
- Linear Regression: R2 = 0.727
- Random Forest: R2 = 0.984 (best, used in the app)

## How to run
pip install -r requirements.txt
python models/train_model.py
python -m streamlit run app.py

## Tech
Python, pandas, scikit-learn, Streamlit 