import pandas as pd
import numpy as np
import joblib
import shap
from src.preprocessing import preprocess

model           = joblib.load("model/final_model.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")
explainer       = shap.TreeExplainer(model)

FINAL_CUTOFF = 0.25  

def predict_single(input_dict: dict) -> dict:
    """
    Input  : dict satu applicant, key = nama kolom raw
    Output : dict berisi pd_score, decision, top_shap_features
    """
    df_input = pd.DataFrame([input_dict])
    df_proc  = preprocess(df_input)

    pd_score = model.predict_proba(df_proc)[0, 1]
    decision = "APPROVE" if pd_score <= FINAL_CUTOFF else "REJECT"

    shap_values = explainer.shap_values(df_proc)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    shap_series = pd.Series(sv[0], index=feature_columns)
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


def predict_batch_shap(df_raw: pd.DataFrame, detail_limit: int = 100) -> dict:
    """
    Input  : DataFrame raw + batas baris untuk detail SHAP
    Output : dict berisi:
             - df_result      : DataFrame hasil scoring (pd_score + decision)
             - shap_summary   : Series mean(|SHAP|) semua baris — untuk bar chart
             - shap_detail_df : DataFrame SHAP top-5 per baris (max detail_limit baris)
    """
    df_proc   = preprocess(df_raw)
    pd_scores = model.predict_proba(df_proc)[:, 1]
    decisions = ["APPROVE" if s <= FINAL_CUTOFF else "REJECT" for s in pd_scores]

    df_result = df_raw.copy()
    df_result["pd_score"] = np.round(pd_scores, 4)
    df_result["decision"] = decisions

    shap_values  = explainer.shap_values(df_proc)
    sv           = shap_values[1] if isinstance(shap_values, list) else shap_values
    shap_matrix  = pd.DataFrame(sv, columns=feature_columns)
    shap_summary = shap_matrix.abs().mean().sort_values(ascending=False)  

    subset       = shap_matrix.iloc[:detail_limit]
    detail_rows  = []
    for i, row in subset.iterrows():
        top5 = row.abs().sort_values(ascending=False).head(5)
        for feat in top5.index:
            detail_rows.append({
                "row"       : i,
                "feature"   : feat,
                "shap_value": round(float(row[feat]), 4),
                "direction" : "Menaikkan risiko" if row[feat] > 0 else "Menurunkan risiko",
            })
    shap_detail_df = pd.DataFrame(detail_rows)

    return {
        "df_result"     : df_result,
        "shap_summary"  : shap_summary,
        "shap_detail_df": shap_detail_df,
    }