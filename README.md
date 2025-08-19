 Song Popularity Prediction

This project predicts the popularity score of songs using **machine learning models** like **XGBoost** and **LightGBM**, with an **ensemble model** for improved accuracy.

---

##  Project Overview
The goal of this project is to predict how popular a song will be based on its audio features and metadata.
We trained multiple machine learning models and compared their performance, then used an ensemble of the models to get better results.

---

##  Features
- Data preprocessing (handling missing values, feature engineering)
- Feature importance analysis (XGBoost & LightGBM)
- Model training and evaluation
- Predictions using multiple models
- Saving predictions to CSV

---

##  Dataset
The dataset contains song metadata and audio features such as:
- `danceability`
- `energy`
- `acousticness`
- `tempo`
- `valence`
- plus additional engineered features like `decade` and `energy\_danceability`

---

##  Models Used
1. **XGBoost** – Gradient boosting model optimized for speed and performance
2. **LightGBM** – Fast, efficient gradient boosting framework
3. **Ensemble Model** – Average predictions from XGBoost and LightGBM for better accuracy

---

##  Results
| Model       | RMSE  | R²    |
|-------------|-------|-------|
| XGBoost     | 9.41  | 0.82  |
| LightGBM    | 9.42  | 0.81  |
| Ensemble    | **9.39**  | **0.82** |

---

## Folder Structure
