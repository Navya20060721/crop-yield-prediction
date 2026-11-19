import os
import joblib
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(base_dir, "models", "crop_model.pkl"))
columns = joblib.load(os.path.join(base_dir, "models", "columns.pkl"))

def predict_yield(area, item, year, rainfall, pesticides, temp):
    row = pd.DataFrame([{
        "Year": year,
        "average_rain_fall_mm_per_year": rainfall,
        "pesticides_tonnes": pesticides,
        "avg_temp": temp,
        "Area": area,
        "Item": item,
    }])
    row = pd.get_dummies(row, columns=["Area", "Item"])
    row = row.reindex(columns=columns, fill_value=0)
    return model.predict(row)[0]

if __name__ == "__main__":
    print(predict_yield("India", "Rice, paddy", 2013, 1083, 100000, 25)) 