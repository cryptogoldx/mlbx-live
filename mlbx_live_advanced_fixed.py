
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="MLBX LIVE", layout="wide")

st.markdown("""
<style>
    .main {
        background-color: #0d1117;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🧠⚾ MLBX LIVE - AI + Gematria Baseball Decoder")
st.caption("Upload your CSV to see value picks, player props, and AI-driven parlay suggestions")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("MLBX_Gematria_Model_Logistic.pkl")

model = load_model()

# Upload
uploaded_file = st.file_uploader("📤 Upload your MLBX + Gematria input CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required = [
        "MLBX_Win_Home_%", "Vegas_Win_Home_%", "Edge_%",
        "Home_Starter_ERA", "Away_Starter_ERA",
        "Home_Bullpen_ERA", "Away_Bullpen_ERA",
        "Gematria_Alignment_Score", "Narrative_Confidence_High"
    ]

    if not all(col in df.columns for col in required):
        st.error("❌ Missing required columns in your file.")
    else:
        # Fill blanks and predict
        input_df = df.copy()
        input_df[required] = input_df[required].fillna(0)
        probs = model.predict_proba(input_df[required])[:, 1]
        preds = model.predict(input_df[required])

        input_df["Predicted_Win"] = preds
        input_df["Win_Probability"] = (probs * 100).round(2)

        # Best value pick highlighter
        input_df["Best_Value_Pick"] = input_df["Edge_%"].apply(
            lambda x: "✅" if float(x) > 7 else ""
        )

        st.success("✅ Predictions complete!")
        st.dataframe(input_df)

        # Win Probability Chart
        st.subheader("📊 Win Probability Distribution")
        fig, ax = plt.subplots()
        input_df["Win_Probability"].plot(kind="hist", bins=10, ax=ax)
        ax.set_title("Win Probability Histogram")
        ax.set_xlabel("Win Probability (%)")
        st.pyplot(fig)

        # Player Props
        st.subheader("🧩 Player Prop Suggestions")
        prop = input_df["Gematria_Alignment_Score"].apply(
            lambda x: "💥 HR Prop" if int(x) > 85 else "🧊 Under Hits"
        )
        props = pd.DataFrame({
            "Matchup": input_df.get("Matchup", pd.Series(["Game 1", "Game 2", "Game 3"])),
            "Prop": prop
        })
        st.table(props)

        # Parlay Builder
        st.subheader("🎯 Top Parlay Picks")
        top3 = input_df.sort_values(by="Win_Probability", ascending=False).head(3)
        parlay = top3[["Matchup", "Win_Probability"]].copy()
        parlay.columns = ["Matchup", "Confidence (%)"]
        st.table(parlay)

        # Download
        csv = input_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results", csv, "MLBX_LIVE_Results.csv", "text/csv")
