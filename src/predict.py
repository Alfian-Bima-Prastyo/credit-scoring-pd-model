import pandas as pd
import numpy as np
import joblib
import shap
from src.preprocessing import preprocess

# Load artefak sekali saja saat modul diimport
model           = joblib.load("model/final_model.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")
explainer       = shap.TreeExplainer(model)

FINAL_CUTOFF = 0.25  # sesuai Cell 104 di notebook

def predict_single(input_dict: dict) -> dict:
    """
    Input  : dict satu applicant, key = nama kolom raw
    Output : dict berisi pd_score, decision, top_shap_features
    """
    df_input = pd.DataFrame([input_dict])
    df_proc  = preprocess(df_input)

    pd_score = model.predict_proba(df_proc)[0, 1]
    decision = "APPROVE" if pd_score <= FINAL_CUTOFF else "REJECT"

    # SHAP — top 5 fitur paling berpengaruh
    shap_values = explainer.shap_values(df_proc)
    shap_series = pd.Series(shap_values[0], index=feature_columns)
    top_shap    = shap_series.abs().sort_values(ascending=False).head(5)
    top_features = [
        {"feature": feat, "shap_value": round(float(shap_series[feat]), 4)}
        for feat in top_shap.index
    ]

    return {
        "pd_score"        : round(float(pd_score), 4),
        "decision"        : decision,
        "top_shap_features": top_features
    }


def predict_batch(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Input  : DataFrame raw (dari CSV upload)
    Output : DataFrame original + kolom pd_score dan decision
    """
    df_proc   = preprocess(df_raw)
    pd_scores = model.predict_proba(df_proc)[:, 1]
    decisions = ["APPROVE" if s <= FINAL_CUTOFF else "REJECT" for s in pd_scores]

    result = df_raw.copy()
    result["pd_score"] = np.round(pd_scores, 4)
    result["decision"] = decisions

    return result