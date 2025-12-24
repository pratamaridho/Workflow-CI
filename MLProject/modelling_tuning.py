# MLProject/modelling.py
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. SETUP ENV & MLFLOW ---
# Kita mengandalkan Environment Variables yang diset di GitHub Actions
# MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD
mlflow.set_experiment("CI_GDP_Prediction")

# --- 2. LOAD DATA ---
print("Memuat data...")
if not os.path.exists('data_gdp_asean_clean.csv'):
    raise FileNotFoundError("Dataset tidak ditemukan!")

df = pd.read_csv('data_gdp_asean_clean.csv')
features = ['Country_Code', 'Year', 'GDP_Prev_Year']
target = 'GDP_Scaled'
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. TRAINING ---
rf = RandomForestRegressor(random_state=42)
param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# --- 4. LOGGING KE MLFLOW ---
with mlflow.start_run() as run:
    # Simpan Run ID ke file text agar bisa dipakai step Docker nanti
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)

    mlflow.log_params(best_params)
    
    y_pred = best_model.predict(X_test)
    mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred))
    mlflow.log_metric("r2", r2_score(y_test, y_pred))
    
    # Log Model (PENTING: folder model dinamai 'model')
    mlflow.sklearn.log_model(best_model, "model")
    
    # Artefak Gambar
    plt.figure()
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.savefig("actual_vs_predicted.png")
    plt.close()
    mlflow.log_artifact("actual_vs_predicted.png")

    print(f"Training Selesai. Run ID: {run.info.run_id}")
