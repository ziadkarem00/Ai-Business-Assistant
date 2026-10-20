# Trained AI Models (`models/`)

This directory stores the trained Machine Learning and NLP models alongside their evaluation metrics:

- **`forecaster.joblib`** — Trained `RandomForestRegressor` model for daily sales forecasting
- **`sentiment.joblib`** — Trained `Pipeline(TfidfVectorizer + LogisticRegression)` model for bilingual customer review sentiment classification 
- **`metrics.json`** — Latest model evaluation benchmarks (`MAE`, `RMSE`, `R2` for sales forecasting, and `Accuracy`, `Macro F1-Score`, and `Confusion Matrix` for sentiment analysis).
