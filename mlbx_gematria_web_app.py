
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="MLBX LIVE", layout="wide")
st.title("🧠⚾ MLBX LIVE - AI + Gematria Baseball Decoder")

@st.cache_resource
def load_model():
    return joblib.load("MLBX_Gematria_Model_Logistic.pkl")

model = load_model()

uploaded_file = st.file_uploader("📤 Upload your MLBX + Gematria input CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_columns = [
        "MLBX_Win_Home_%", "Vegas_Win_Home_%", "Edge_%",
        "Home_Starter_ERA", "Away_Starter_ERA",
        "Home_Bullpen_ERA", "Away_Bullpen_ERA",
        "Gematria_Alignment_Score", "Narrative_Confidence_High"
    ]

    if not all(col in df.columns for col in required_columns):
        st.error("❌ Your file is missing required columns.")
    else:
        predictions = model.predict(df[required_columns])
        probabilities = model.predict_proba(df[required_columns])[:, 1]

        df["Predicted_Win"] = predictions
        df["Win_Probability"] = (probabilities * 100).round(2)

        st.success("✅ Predictions generated!")
        st.dataframe(df)

        st.subheader("📊 Win Probability Distribution")
        fig, ax = plt.subplots()
        df["Win_Probability"].plot(kind="hist", bins=10, ax=ax)
        ax.set_title("Win Probability Histogram")
        ax.set_xlabel("Win Probability (%)")
        st.pyplot(fig)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results as CSV", csv, "mlbx_predictions.csv", "text/csv")
