import streamlit as st
import pandas as pd
from agents.leak_detector import detect_leaks
from agents.insight_agent import generate_summary

st.set_page_config(page_title="ZeroLeak.AI", layout="wide")
st.title("📉 ZeroLeak.AI – Revenue Leakage Analyzer")

uploaded_file = st.file_uploader("Upload Stripe / CRM CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview")
    st.dataframe(df.head())

    if st.button("🔍 Run Analysis"):
        flagged_df = detect_leaks(df)
        summary = generate_summary(flagged_df)

        st.success("Analysis Complete ✅")
        st.subheader("⚠️ Detected Revenue Issues")
        st.dataframe(flagged_df)

        st.subheader("🧠 AI Insights")
        st.markdown(summary)

        st.download_button("📥 Download Fix Report", flagged_df.to_csv(index=False), file_name="zeroleak_report.csv")