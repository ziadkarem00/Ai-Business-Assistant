"""FastAPI Backend Server for AI Business Assistant.

Run from the project root directory:
    uvicorn backend.main:app --reload --port 8000
Or directly:
    python backend/main.py
"""

import os
import sys
import json
from typing import Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root is in sys.path so `python backend/main.py` works from anywhere
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.data_prep import load_sales, load_reviews
from backend import analytics as A
from backend.forecasting import train_forecaster, forecast_future, feature_importance_df
from backend.sentiment import train_sentiment, predict_sentiment, top_complaints
from backend.assistant import answer, recommendations

app = FastAPI(
    title="AI Business Assistant API",
    description=(
        "REST API for analyzing retail sales, forecasting future demand with Random Forest, "
        "performing NLP sentiment analysis with TF-IDF & Logistic Regression, and answering "
        "business questions via a data-driven Chatbot."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    lang: str = "en"


class SentimentRequest(BaseModel):
    text: str


# Initialize data and models on server startup
STATE = {}


def init_engine():
    """Load data, train/save ML models, and cache results in memory."""
    sales = load_sales()
    reviews = load_reviews()
    rf_model, rf_metrics, rf_comp, feat_df = train_forecaster(sales, test_days=60, save_model=True)
    future_30 = forecast_future(rf_model, sales, days=30)
    sent_model, sent_metrics = train_sentiment(reviews, save_model=True)

    # Save metrics summary to models/metrics.json
    models_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    summary_metrics = {
        "forecasting": rf_metrics,
        "sentiment": {
            "accuracy": sent_metrics["accuracy"],
            "f1_macro": sent_metrics["f1_macro"],
            "labels": sent_metrics["labels"],
            "confusion": sent_metrics["confusion"],
        },
    }
    with open(os.path.join(models_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    STATE["sales"] = sales
    STATE["reviews"] = reviews
    STATE["rf_model"] = rf_model
    STATE["rf_metrics"] = rf_metrics
    STATE["rf_comp"] = rf_comp
    STATE["feat_cols"] = [c for c in feat_df.columns if c != "y"]
    STATE["future_30"] = future_30
    STATE["sent_model"] = sent_model
    STATE["sent_metrics"] = sent_metrics


@app.on_event("startup")
def on_startup():
    if not STATE:
        init_engine()


def get_state():
    if not STATE:
        init_engine()
    return STATE


@app.get("/")
def root():
    s = get_state()
    return {
        "project": "AI Business Assistant API",
        "status": "online",
        "sales_records": len(s["sales"]),
        "review_records": len(s["reviews"]),
        "docs_url": "/docs",
    }


@app.get("/api/kpis")
def get_kpis():
    s = get_state()
    sales = s["sales"]
    monthly = A.monthly_sales(sales).copy()
    monthly["Month"] = monthly["Month"].astype(str)
    return {
        "kpis": A.kpis(sales),
        "top_products": A.top_products(sales, n=10, ascending=False).to_dict(orient="records"),
        "low_products": A.top_products(sales, n=10, ascending=True).to_dict(orient="records"),
        "monthly_sales": monthly.to_dict(orient="records"),
        "by_country": A.by_country(sales, n=10).to_dict(orient="records"),
    }


@app.get("/api/forecast")
def get_forecast(days: int = Query(30, ge=7, le=90)):
    s = get_state()
    fut = s["future_30"] if days == 30 else forecast_future(s["rf_model"], s["sales"], days=days)
    fut_out = fut.copy()
    fut_out["Date"] = fut_out["Date"].astype(str)

    comp_out = s["rf_comp"].reset_index().rename(columns={"InvoiceDate": "Date"})
    comp_out["Date"] = comp_out["Date"].astype(str)

    feat_imp = feature_importance_df(s["rf_model"], s["feat_cols"])
    return {
        "metrics": s["rf_metrics"],
        "comparison": comp_out.to_dict(orient="records"),
        "forecast": fut_out.to_dict(orient="records"),
        "feature_importance": feat_imp.to_dict(orient="records"),
    }


@app.get("/api/sentiment")
def get_sentiment_overview():
    s = get_state()
    reviews = s["reviews"]
    dist = reviews["sentiment"].value_counts().to_dict()
    complaints = [{"phrase": p, "count": c} for p, c in top_complaints(reviews, n=8)]
    return {
        "metrics": {
            "accuracy": s["sent_metrics"]["accuracy"],
            "f1_macro": s["sent_metrics"]["f1_macro"],
            "labels": s["sent_metrics"]["labels"],
            "confusion": s["sent_metrics"]["confusion"],
        },
        "distribution": dist,
        "top_complaints": complaints,
    }


@app.post("/api/sentiment/predict")
def predict_review_sentiment(req: SentimentRequest):
    s = get_state()
    return predict_sentiment(s["sent_model"], req.text)


@app.post("/api/chat")
def chat_with_assistant(req: ChatRequest):
    s = get_state()
    reply = answer(req.question, s["sales"], s["reviews"], s["future_30"], lang=req.lang)
    return {"question": req.question, "lang": req.lang, "answer": reply}


@app.get("/api/recommendations")
def get_business_recommendations(lang: str = Query("en", regex="^(en|ar)$")):
    s = get_state()
    recs = recommendations(s["sales"], s["reviews"], s["future_30"], lang=lang)
    return {"lang": lang, "recommendations": recs}


if __name__ == "__main__":
    import uvicorn
    init_engine()
    print("Starting AI Business Assistant FastAPI Server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
