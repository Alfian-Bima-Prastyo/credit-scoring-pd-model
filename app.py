import streamlit as st
import pandas as pd
import joblib
from src.predict import predict_single, predict_batch_shap

st.set_page_config(page_title="Credit Scoring", layout="wide")

feature_columns = joblib.load("model/feature_columns.pkl")

st.title("Credit Risk — Probability of Default")
st.caption("Portofolio project · Home Credit Dataset · LightGBM")

tab1, tab2 = st.tabs(["Single Prediction", "Batch Scoring"])

with tab1:
    st.subheader("Input Applicant")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        AMT_INCOME_TOTAL  = st.number_input("Annual income (IDR)",         value=270000.0, step=10000.0)
        AMT_CREDIT        = st.number_input("Loan amount",                  value=500000.0, step=10000.0)
        AMT_ANNUITY       = st.number_input("Monthly annuity",              value=25000.0,  step=1000.0)
        AMT_GOODS_PRICE   = st.number_input("Goods price",                  value=450000.0, step=10000.0)

    with col2:
        DAYS_BIRTH        = st.number_input("Age (years)",                  value=35, step=1) * -365
        years_employed    = st.number_input("Years employed (0 = tidak bekerja)", value=5, min_value=0, step=1)
        DAYS_EMPLOYED     = 365243 if years_employed == 0 else years_employed * -365
        CNT_FAM_MEMBERS   = st.number_input("Family members",               value=2,  step=1)

    with col3:
        EXT_SOURCE_1      = st.number_input("EXT_SOURCE_1 (0–1, opsional)", value=0.5, min_value=0.0, max_value=1.0)
        EXT_SOURCE_2      = st.number_input("EXT_SOURCE_2 (0–1, opsional)", value=0.5, min_value=0.0, max_value=1.0)
        EXT_SOURCE_3      = st.number_input("EXT_SOURCE_3 (0–1, opsional)", value=0.5, min_value=0.0, max_value=1.0)

    with col4:
        years_registration = st.number_input("Years since registration",    value=10, min_value=0, step=1)
        DAYS_REGISTRATION  = years_registration * -365
        years_id_publish   = st.number_input("Years since ID publish",      value=5,  min_value=0, step=1)
        DAYS_ID_PUBLISH    = years_id_publish * -365
        st.caption("Fitur lain yang tidak ada di form diasumsikan nilai default (0). "
                   "Lihat SHAP untuk kontribusi lengkap semua fitur.")
        
    if st.button("Predict", type="primary"):

        input_dict = {
            "AMT_INCOME_TOTAL" : AMT_INCOME_TOTAL,
            "AMT_CREDIT"       : AMT_CREDIT,
            "AMT_ANNUITY"      : AMT_ANNUITY,
            "AMT_GOODS_PRICE"  : AMT_GOODS_PRICE,
            "DAYS_BIRTH"       : DAYS_BIRTH,
            "DAYS_EMPLOYED"    : DAYS_EMPLOYED,
            "CNT_FAM_MEMBERS"  : CNT_FAM_MEMBERS,
            "DAYS_REGISTRATION": DAYS_REGISTRATION,
            "DAYS_ID_PUBLISH"  : DAYS_ID_PUBLISH,
            "EXT_SOURCE_1"     : EXT_SOURCE_1,
            "EXT_SOURCE_2"     : EXT_SOURCE_2,
            "EXT_SOURCE_3"     : EXT_SOURCE_3,
        }

        result = predict_single(input_dict)

        col_score, col_dec = st.columns(2)
        with col_score:
            st.metric("PD Score", f"{result['pd_score']:.4f}")
        with col_dec:
            color = "green" if result["decision"] == "APPROVE" else "red"
            st.markdown(f"**Decision:** :{color}[{result['decision']}]")

        st.subheader("Top 5 Feature Contributions")
        shap_df = pd.DataFrame(result["top_shap_features"])
        shap_df["direction"] = shap_df["shap_value"].apply(
            lambda x: "Menaikkan risiko" if x > 0 else "Menurunkan risiko"
        )
        st.dataframe(shap_df, use_container_width=True)

        form_fields = list(input_dict.keys())
        derived_features = {
            "EXT_SOURCE_MEAN", "CREDIT_TO_INCOME", "ANNUITY_TO_INCOME",
            "CREDIT_TO_ANNUITY", "INCOME_PER_PERSON", "CREDIT_TO_AGE",
            "EXT_SOURCE_1_MISSING", "EXT_SOURCE_2_MISSING", "EXT_SOURCE_3_MISSING",
            "EMPLOYED_ANOMALY",
        }
        known_features = set(form_fields) | derived_features
        non_form = [r for r in result["top_shap_features"] if r["feature"] not in known_features]
        if non_form:
            st.caption(
                f" {len(non_form)} dari 5 fitur di atas berasal dari data bureau/internal "
                f"({', '.join(r['feature'] for r in non_form)}) — tidak tersedia di form, "
                f"diasumsikan nilai 0."
            )

with tab2:
    st.subheader("Upload CSV")
    st.caption("CSV harus memiliki kolom yang sama dengan raw input applicant")

    uploaded = st.file_uploader("Upload file CSV", type="csv")

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.write(f"Data diterima: {len(df_raw):,} baris")

        _max_detail  = min(200, len(df_raw))
        _min_detail  = min(1, _max_detail)
        _step_detail = 1 if _max_detail < 10 else 10
        detail_limit = st.slider(
            "Jumlah baris untuk SHAP detail (per-row explanation)",
            min_value=_min_detail, max_value=_max_detail,
            value=_max_detail, step=_step_detail
        )

        if st.button("Score All", type="primary"):
            with st.spinner("Memproses scoring + SHAP..."):
                batch_result = predict_batch_shap(df_raw, detail_limit=detail_limit)

            df_result      = batch_result["df_result"]
            shap_summary   = batch_result["shap_summary"]
            shap_detail_df = batch_result["shap_detail_df"]

            approval_rate = (df_result["decision"] == "APPROVE").mean()
            avg_pd        = df_result["pd_score"].mean()

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Applicants", f"{len(df_result):,}")
            m2.metric("Approval Rate",    f"{approval_rate:.1%}")
            m3.metric("Avg PD Score",     f"{avg_pd:.4f}")

            st.subheader("Hasil Scoring")
            st.dataframe(
                df_result[["pd_score", "decision"]].join(df_raw.select_dtypes("number")),
                use_container_width=True
            )

            st.subheader("SHAP Summary — Top 15 Feature Importance (Seluruh Batch)")
            st.caption("Rata-rata |SHAP value| dari seluruh baris. Semakin besar, semakin berpengaruh terhadap prediksi.")
            top15 = shap_summary.head(15).reset_index()
            top15.columns = ["feature", "mean_abs_shap"]
            top15["mean_abs_shap"] = top15["mean_abs_shap"].round(4)
            st.bar_chart(top15.set_index("feature")["mean_abs_shap"])

            with st.expander("Lihat tabel SHAP summary"):
                st.dataframe(top15, use_container_width=True)

            st.subheader(f"SHAP Detail — Top 5 Fitur per Baris (subset {detail_limit} baris pertama)")
            st.caption("Setiap baris menunjukkan 5 fitur paling berpengaruh untuk applicant tersebut.")

            if not shap_detail_df.empty:
                row_options = sorted(shap_detail_df["row"].unique())
                selected_row = st.selectbox("Pilih baris applicant:", row_options)
                row_shap = shap_detail_df[shap_detail_df["row"] == selected_row].drop(columns="row")
                st.dataframe(row_shap, use_container_width=True)

                with st.expander("Lihat semua SHAP detail (semua baris subset)"):
                    st.dataframe(shap_detail_df, use_container_width=True)

            csv_out = df_result.to_csv(index=False).encode("utf-8")
            st.download_button("Download Hasil CSV", csv_out, "hasil_scoring.csv", "text/csv")