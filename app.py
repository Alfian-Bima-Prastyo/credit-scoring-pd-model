import streamlit as st
import pandas as pd
import joblib
from src.predict import predict_single, predict_batch

st.set_page_config(page_title="Credit Scoring", layout="wide")

feature_columns = joblib.load("model/feature_columns.pkl")

st.title("Credit Risk — Probability of Default")
st.caption("Portofolio project · Home Credit Dataset · LightGBM")

tab1, tab2 = st.tabs(["Single Prediction", "Batch Scoring"])

# ── TAB 1: SINGLE PREDICTION ─────────────────────────────────────────────────
with tab1:
    st.subheader("Input Applicant")

    col1, col2, col3 = st.columns(3)

    with col1:
        AMT_INCOME_TOTAL  = st.number_input("Annual income (IDR)",        value=270000.0, step=10000.0)
        AMT_CREDIT        = st.number_input("Loan amount",                 value=500000.0, step=10000.0)
        AMT_ANNUITY       = st.number_input("Monthly annuity",             value=25000.0,  step=1000.0)

    with col2:
        DAYS_BIRTH        = st.number_input("Age (years)",                 value=35, step=1) * -365
        DAYS_EMPLOYED     = st.number_input("Years employed",              value=5,  step=1) * -365
        CNT_FAM_MEMBERS   = st.number_input("Family members",              value=2,  step=1)

    with col3:
        EXT_SOURCE_1      = st.number_input("EXT_SOURCE_1 (0–1, opsional)", value=0.5, min_value=0.0, max_value=1.0)
        EXT_SOURCE_2      = st.number_input("EXT_SOURCE_2 (0–1, opsional)", value=0.5, min_value=0.0, max_value=1.0)
        EXT_SOURCE_3      = st.number_input("EXT_SOURCE_3 (0–1, opsional)", value=0.5, min_value=0.0, max_value=1.0)

    if st.button("Predict", type="primary"):
        input_dict = {
            "AMT_INCOME_TOTAL" : AMT_INCOME_TOTAL,
            "AMT_CREDIT"       : AMT_CREDIT,
            "AMT_ANNUITY"      : AMT_ANNUITY,
            "DAYS_BIRTH"       : DAYS_BIRTH,
            "DAYS_EMPLOYED"    : DAYS_EMPLOYED,
            "CNT_FAM_MEMBERS"  : CNT_FAM_MEMBERS,
            "EXT_SOURCE_1"     : EXT_SOURCE_1,
            "EXT_SOURCE_2"     : EXT_SOURCE_2,
            "EXT_SOURCE_3"     : EXT_SOURCE_3,
        }

        result = predict_single(input_dict)

        # Tampilkan hasil
        col_score, col_dec = st.columns(2)
        with col_score:
            st.metric("PD Score", f"{result['pd_score']:.4f}")
        with col_dec:
            color = "green" if result["decision"] == "APPROVE" else "red"
            st.markdown(f"**Decision:** :{color}[{result['decision']}]")

        # SHAP explanation
        st.subheader("Top 5 Feature Contributions")
        shap_df = pd.DataFrame(result["top_shap_features"])
        shap_df["direction"] = shap_df["shap_value"].apply(
            lambda x: "Menaikkan risiko" if x > 0 else "Menurunkan risiko"
        )
        st.dataframe(shap_df, use_container_width=True)

# ── TAB 2: BATCH SCORING ──────────────────────────────────────────────────────
with tab2:
    st.subheader("Upload CSV")
    st.caption("CSV harus memiliki kolom yang sama dengan raw input applicant")

    uploaded = st.file_uploader("Upload file CSV", type="csv")

    if uploaded:
        df_raw  = pd.read_csv(uploaded)
        st.write(f"Data diterima: {len(df_raw):,} baris")

        if st.button("Score All", type="primary"):
            with st.spinner("Memproses..."):
                df_result = predict_batch(df_raw)

            # Summary metrics
            approval_rate = (df_result["decision"] == "APPROVE").mean()
            avg_pd        = df_result["pd_score"].mean()

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Applicants", f"{len(df_result):,}")
            m2.metric("Approval Rate",    f"{approval_rate:.1%}")
            m3.metric("Avg PD Score",     f"{avg_pd:.4f}")

            st.dataframe(df_result[["pd_score", "decision"]].join(
                df_raw.select_dtypes("number").head()), 
                use_container_width=True
            )

            # Download hasil
            csv_out = df_result.to_csv(index=False).encode("utf-8")
            st.download_button("Download Hasil CSV", csv_out, "hasil_scoring.csv", "text/csv")