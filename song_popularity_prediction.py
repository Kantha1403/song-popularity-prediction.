import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("data.csv")

# Display first rows
df.head()

# Basic info
print(df.info())

# Check for missing values
print(df.isnull().sum())
# Phase 2: Data Cleaning

# Remove duplicates based on 'id' (song uniqueness)
df.drop_duplicates(subset=['id'], inplace=True)

# Optional: Filter out unpopular tracks (popularity < 5)
df = df[df['popularity'] > 5]

# Optional: Keep only tracks after the year 2000
df = df[df['year'] >= 2000]

# Reset index
df.reset_index(drop=True, inplace=True)

print(f"✅ Cleaned dataset has {len(df)} rows after filtering")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create folder for plots
os.makedirs("plots", exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("data.csv")
print(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
features = ['danceability', 'valence', 'tempo']
plt.figure(figsize=(15, 5))
for i, feature in enumerate(features, 1):
    plt.subplot(1, 3, i)
    sns.histplot(df[feature], kde=True, bins=30, color='skyblue')
    plt.title(f"{feature.capitalize()} Distribution")
plt.tight_layout()
plt.savefig("plots/feature_distributions.png", dpi=300)
plt.show()
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(10, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.savefig("plots/correlation_heatmap.png", dpi=300)
plt.show()
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='energy', y='loudness', hue='popularity', palette='viridis', alpha=0.7)
plt.title("Energy vs Loudness")
plt.savefig("plots/energy_vs_loudness.png", dpi=300)
plt.show()
plt.figure(figsize=(10, 5))
sns.lineplot(
    data=df.groupby('year')['popularity'].mean().reset_index(),
    x='year', y='popularity', marker="o"
)
plt.title("Average Popularity Over Years")
plt.savefig("plots/popularity_by_year.png", dpi=300)
plt.show()
violin_features = ['danceability', 'valence', 'energy', 'tempo']
for feature in violin_features:
plt.figure(figsize=(8, 5))
sns.violinplot(x='explicit', y=feature, data=df, palette='muted')
plt.title(f"{feature.capitalize()} Distribution by Explicit Content")
plt.savefig(f"plots/violin_{feature}.png", dpi=300)
plt.show()
top_artists = (
    df.groupby('artists')['popularity']
    .mean()
    .sort_values(ascending=False)
    .head(15)
    .reset_index()
)
plt.figure(figsize=(10, 6))
sns.barplot(data=top_artists, y='artists', x='popularity', palette='coolwarm')
plt.title("Top 15 Artists by Average Popularity")
plt.savefig("plots/top_artists.png", dpi=300)
plt.show()
df['decade'] = (df['year'] // 10) * 10
plt.figure(figsize=(10, 6))
sns.boxplot(x='decade', y='tempo', data=df, palette='pastel')
plt.title("Tempo Distribution by Decade")
plt.savefig("plots/tempo_by_decade.png", dpi=300)
plt.show()
# ===== Phase 4 — Predicting Song Popularity =====
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ===== Load Cleaned Dataset =====
df = pd.read_csv("data.csv")  # use your cleaned file name

# ===== Features & Target =====
features = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'duration_ms'
]
target = 'popularity'

X = df[features]
y = df[target]

# ===== Train-Test Split =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== Train Model =====
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ===== Predictions =====
y_pred = model.predict(X_test)

# ===== Evaluation =====
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ R² Score: {r2:.2f}")

# ===== Feature Importance =====
importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# ===== Plot Feature Importance =====
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance for Predicting Popularity')
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load dataset
df = pd.read_csv("data.csv")
logging.info(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ===== Feature Engineering =====
df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
df["decade"] = (df["year"] // 10) * 10
df["energy_danceability"] = df["energy"] * df["danceability"]
df = df.dropna(subset=["year"])

features = [
    'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
    'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
    'speechiness', 'tempo', 'valence', 'year', 'decade', 'energy_danceability'
]
X = df[features]
y = df['popularity']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===== XGBoost Model =====
xgb_model = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
logging.info("🚀 Training XGBoost...")
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
rmse_xgb = mean_squared_error(y_test, y_pred_xgb, squared=False)
r2_xgb = r2_score(y_test, y_pred_xgb)

# ===== LightGBM Model =====
lgbm_model = LGBMRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
logging.info("🚀 Training LightGBM...")
lgbm_model.fit(X_train, y_train)

y_pred_lgbm = lgbm_model.predict(X_test)
rmse_lgbm = mean_squared_error(y_test, y_pred_lgbm, squared=False)
r2_lgbm = r2_score(y_test, y_pred_lgbm)

# ===== Results =====
logging.info(f"📊 XGBoost -> RMSE: {rmse_xgb:.2f}, R²: {r2_xgb:.2f}")
logging.info(f"📊 LightGBM -> RMSE: {rmse_lgbm:.2f}, R²: {r2_lgbm:.2f}")
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# ===== 1. Feature Importance Plots =====
def plot_feature_importance(model, model_name, feature_names, ax):
    importance = model.feature_importances_
    sorted_idx = importance.argsort()
    ax.barh(range(len(sorted_idx)), importance[sorted_idx], align="center")
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_title(f"{model_name} Feature Importance")
    ax.set_xlabel("Importance Score")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

plot_feature_importance(xgb_model, "XGBoost", X.columns, axes[0])
plot_feature_importance(lgbm_model, "LightGBM", X.columns, axes[1])

plt.tight_layout()
plt.show()

# ===== 2. Save Models =====
joblib.dump(xgb_model, "xgboost_model.pkl")
joblib.dump(lgbm_model, "lightgbm_model.pkl")
print("✅ Models saved as 'xgboost_model.pkl' and 'lightgbm_model.pkl'")

# ===== 3. Predict on Random Songs (fixed) =====
sample_df = df.sample(5, random_state=42)

# Keep only the features used during training
X_sample = sample_df[X.columns]  # ensures same feature set
y_sample = sample_df["popularity"]

xgb_preds = xgb_model.predict(X_sample)
lgbm_preds = lgbm_model.predict(X_sample)

results = pd.DataFrame({
    "Song": sample_df["name"].values,
    "Artists": sample_df["artists"].values,
    "Actual Popularity": y_sample.values,
    "XGBoost Pred": xgb_preds,
    "LightGBM Pred": lgbm_preds
})

print("\n🎯 Sample Predictions:")
print(results)
import pandas as pd
import numpy as np
import joblib
import json

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

RNG = 42
np.random.seed(RNG)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def summarize_scores(name, y_true, y_pred):
    print(f"📊 {name} -> RMSE: {rmse(y_true, y_pred):.2f}, R²: {r2_score(y_true, y_pred):.2f}")
# Load CSV
df = pd.read_csv("data.csv")

# Feature engineering
df["decade"] = (df["year"] // 10) * 10
df["energy_danceability"] = df["energy"] * df["danceability"]

TARGET = "popularity"
DROP_COLS = ["artists", "id", "name", "release_date"]

# Exact feature order — this will be saved for later
FEATURES = [
    "acousticness", "danceability", "duration_ms", "energy", "explicit",
    "instrumentalness", "key", "liveness", "loudness", "mode",
    "speechiness", "tempo", "valence", "year", "decade", "energy_danceability"
]

X = df.drop(columns=DROP_COLS + [TARGET])
X = X[FEATURES].copy()
y = df[TARGET].copy()

print(f"✅ Data ready: {X.shape}")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RNG
)
# Define models
xgb = XGBRegressor(
    n_estimators=400, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric="rmse",
    tree_method="hist", random_state=RNG, n_jobs=-1
)

lgbm = LGBMRegressor(
    n_estimators=600, num_leaves=63, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=RNG, n_jobs=-1
)

# Fit
xgb.fit(X_train, y_train)
lgbm.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
lgbm_preds = lgbm.predict(X_test)
ensemble_preds = (xgb_preds + lgbm_preds) / 2

summarize_scores("XGBoost", y_test, xgb_preds)
summarize_scores("LightGBM", y_test, lgbm_preds)
summarize_scores("Ensemble", y_test, ensemble_preds)
joblib.dump(xgb, "xgboost.pkl")
joblib.dump(lgbm, "lightgbm.pkl")

with open("features.json", "w") as f:
    json.dump(FEATURES, f)

print("💾 Saved models and feature list")
def load_features():
    with open("features.json", "r") as f:
        return json.load(f)

def prepare_data(df, features):
    df["decade"] = (df["year"] // 10) * 10
    df["energy_danceability"] = df["energy"] * df["danceability"]
    for col in ["artists", "id", "name", "release_date"]:
        if col in df.columns:
            df = df.drop(columns=col)
    return df[features]

# Example — reload and predict
features = load_features()
xgb_loaded = joblib.load("xgboost.pkl")
lgbm_loaded = joblib.load("lightgbm.pkl")

X_test_safe = prepare_data(df.iloc[:20].copy(), features)
p1 = xgb_loaded.predict(X_test_safe)
p2 = lgbm_loaded.predict(X_test_safe)
p_ens = (p1 + p2) / 2

print("✅ Predictions ready", p_ens[:5])
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Get feature names
feature_names = X_full.columns

# Extract importances and normalize to [0,1]
xgb_importances = xgb_model.feature_importances_
lgbm_importances = lgbm_model.feature_importances_

xgb_importances = xgb_importances / np.max(xgb_importances)
lgbm_importances = lgbm_importances / np.max(lgbm_importances)

# Create DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'XGBoost': xgb_importances,
    'LightGBM': lgbm_importances
})

# Get top 10 by average importance
importance_df['Average'] = (importance_df['XGBoost'] + importance_df['LightGBM']) / 2
importance_df = importance_df.sort_values(by='Average', ascending=False).head(10)

# Plot
plt.figure(figsize=(10,6))
bar_width = 0.35
indices = np.arange(len(importance_df))

plt.barh(indices, importance_df['XGBoost'], bar_width, label='XGBoost', alpha=0.7)
plt.barh(indices + bar_width, importance_df['LightGBM'], bar_width, label='LightGBM', alpha=0.7)

plt.yticks(indices + bar_width / 2, importance_df['Feature'])
plt.xlabel('Normalized Importance')
plt.title('Top 10 Feature Importances (Normalized)')
plt.legend()
plt.gca().invert_yaxis()
plt.show()
import pandas as pd
import numpy as np
import joblib

# ===== Load new data =====
new_df = pd.read_csv("data.csv")

# ===== Recreate missing engineered features =====
# Create 'decade' from 'year' if missing
if 'year' in new_df.columns and 'decade' not in new_df.columns:
    new_df['decade'] = (new_df['year'] // 10) * 10

# Create 'energy_danceability' if missing
if {'energy', 'danceability'}.issubset(new_df.columns) and 'energy_danceability' not in new_df.columns:
    new_df['energy_danceability'] = new_df['energy'] * new_df['danceability']

# ===== Load models =====
xgb_model = joblib.load("xgboost_model.pkl")
lgbm_model = joblib.load("lightgbm_model.pkl")

# ===== Match training feature order =====
X_new = new_df[X_full.columns]  # X_full must come from training step

# ===== Predict with each model =====
xgb_preds_new = xgb_model.predict(X_new)
lgbm_preds_new = lgbm_model.predict(X_new)

# ===== Ensemble prediction =====
ensemble_preds_new = (xgb_preds_new + lgbm_preds_new) / 2

# ===== Save predictions =====
pred_df = new_df.copy()
pred_df['Predicted_Popularity_XGBoost'] = xgb_preds_new
pred_df['Predicted_Popularity_LightGBM'] = lgbm_preds_new
pred_df['Predicted_Popularity_Ensemble'] = ensemble_preds_new

output_file = "predictions.csv"
pred_df.to_csv(output_file, index=False)

# ===== Output summary =====
print(f"✅ Predictions saved to {output_file}")
print(pred_df[['Predicted_Popularity_XGBoost', 
               'Predicted_Popularity_LightGBM', 
               'Predicted_Popularity_Ensemble']].head(10))

