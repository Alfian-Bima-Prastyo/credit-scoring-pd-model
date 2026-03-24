import pandas as pd
import numpy as np
import joblib
import re

def preprocess(df_input: pd.DataFrame) -> pd.DataFrame:
    df = df_input.copy()

    income_cap      = joblib.load("model/income_cap.pkl")
    feature_columns = joblib.load("model/feature_columns.pkl")
    ext_src_medians = joblib.load("model/ext_source_medians.pkl")

    df["AMT_INCOME_TOTAL"] = np.where(
        df["AMT_INCOME_TOTAL"] > income_cap, income_cap, df["AMT_INCOME_TOTAL"]
    )

    df["EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"]    = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    for col in ext_cols:
        if col not in df.columns:
            df[col] = np.nan
    for col in ext_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ext_cols:
        df[col + "_MISSING"] = df[col].isnull().astype(int)
    for col in ext_cols:
        df[col] = df[col].fillna(ext_src_medians[col])
    
    df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)

    df["CREDIT_TO_INCOME"]  = df["AMT_CREDIT"]  / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_TO_INCOME"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TO_ANNUITY"] = df["AMT_CREDIT"]  / df["AMT_ANNUITY"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    df["CREDIT_TO_AGE"]     = df["AMT_CREDIT"]  / (df["DAYS_BIRTH"] / -365)

    df = pd.get_dummies(df, drop_first=True)
    df.columns = [re.sub('[^A-Za-z0-9_]+', '_', col) for col in df.columns]

    df = df.reindex(columns=feature_columns, fill_value=0)

    return df