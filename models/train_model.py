import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(base_dir, "datasets", "yield_df.csv"))
df = df.drop(columns=["Unnamed: 0"], errors="ignore")
df = df.drop_duplicates().dropna()
print("Shape:", df.shape)

df_encoded = pd.get_dummies(df, columns=["Area", "Item"])
X = df_encoded.drop("hg/ha_yield", axis=1)
y = df_encoded["hg/ha_yield"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
}

best_name, best_model, best_r2 = None, None, -float("inf")
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"{name}: R2={r2:.3f}  MAE={mae:.0f}  RMSE={rmse:.0f}")
    if r2 > best_r2:
        best_name, best_model, best_r2 = name, model, r2

print("Best model:", best_name)

os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
joblib.dump(best_model, os.path.join(base_dir, "models", "crop_model.pkl"))
joblib.dump(list(X.columns), os.path.join(base_dir, "models", "columns.pkl"))
joblib.dump(sorted(df["Area"].unique()), os.path.join(base_dir, "models", "areas.pkl"))
joblib.dump(sorted(df["Item"].unique()), os.path.join(base_dir, "models", "items.pkl"))
print("Model saved") 