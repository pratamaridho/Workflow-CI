import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. KONFIGURASI MLFLOW (LOKAL) ---
# Tidak perlu DagsHub init atau set_tracking_uri
# Secara default akan membuat folder 'mlruns' di direktori saat ini
mlflow.set_experiment("Skilled_GDP_Prediction_Local")

# --- 2. LOAD DATA ---
print("Memuat data...")
# Pastikan file csv berada di folder yang sama
# Jika file tidak ditemukan, pastikan path-nya benar
if not os.path.exists('data_gdp_asean_clean.csv'):
    raise FileNotFoundError("File 'data_gdp_asean_clean.csv' tidak ditemukan!")

df = pd.read_csv('data_gdp_asean_clean.csv')

# Definisikan Fitur dan Target
features = ['Country_Code', 'Year', 'GDP_Prev_Year']
target = 'GDP_Scaled'

X = df[features]
y = df[target]

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data Loaded. Train: {X_train.shape}, Test: {X_test.shape}")

# --- 3. HYPERPARAMETER TUNING ---
print("Memulai Hyperparameter Tuning...")
rf = RandomForestRegressor(random_state=42)

# Parameter yang akan di-tuning
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

# Grid Search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print(f"Model Terbaik ditemukan: {best_params}")

# --- 4. MLFLOW RUN (MANUAL LOGGING - SYARAT SKILLED/ADVANCE) ---
with mlflow.start_run(run_name="Best_Model_Tuning_RF"):
    
    # A. Log Parameters
    mlflow.log_params(best_params)
    
    # Predict
    y_pred = best_model.predict(X_test)
    
    # B. Log Metrics (Manual)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Metrics -> MAE: {mae:.5f}, RMSE: {rmse:.5f}, R2: {r2:.5f}")
    
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)
    
    # C. Log Model
    mlflow.sklearn.log_model(best_model, "random_forest_model")
    
    # --- 5. ARTEFAK TAMBAHAN ---
    
    # Artefak 1: Plot Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='b')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual GDP Scaled")
    plt.ylabel("Predicted GDP Scaled")
    plt.title("Actual vs Predicted GDP")
    plt.savefig("actual_vs_predicted.png")
    plt.close()
    
    mlflow.log_artifact("actual_vs_predicted.png")
    
    # Artefak 2: Feature Importance Plot
    plt.figure(figsize=(10, 6))
    importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
    importances.sort_values().plot(kind='barh', color='teal')
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()
    
    mlflow.log_artifact("feature_importance.png")
    
    # Artefak 3: Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='purple')
    plt.title("Residuals Distribution")
    plt.xlabel("Error")
    plt.savefig("residuals_distribution.png")
    plt.close()
    
    mlflow.log_artifact("residuals_distribution.png")

    print("\n[SUCCESS] Training Selesai.")
    print(f"Jalankan 'mlflow ui' di terminal untuk melihat hasil tracking.")