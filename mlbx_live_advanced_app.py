
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="MLBX LIVE - Baseball AI Decoder", layout="wide")

st.markdown("""
<style>
    .main {
        background-color: #0d1117;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        font-weight: bold;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠⚾ MLBX LIVE - AI + Gematria Baseball Decoder")
st.subheader("🔎 Upload your Matchups and Get AI Predictions, Props & Parlays")

@st.cache_resource
def load_model():
    return joblib.load("MLBX_Gematria_Model_Logistic.pkl")

model = load_model()

uploaded_file = st.file_uploader("📤 Upload your MLBX + Gematria CSV Template", type="csv")

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
        input_df = df.copy()
        input_df[required_columns] = input_df[required_columns].fillna(0)
        predictions = model.predict(input_df[required_columns])
        probabilities = model.predict_proba(input_df[required_columns])[:, 1]
        input_df["Predicted_Win"] = predictions
        input_df["Win_Probability"] = (probabilities * 100).round(2)

        # Highlight Best Value Picks
        input_df["Best_Value_Pick"] = input_df["Edge_%"].apply(lambda x: "✅" if x != "" and float(x) > 7 else "")

        st.success("✅ Predictions Generated!")
        st.dataframe(input_df)

        # Chart
        st.subheader("📊 Win Probability Distribution")
        fig, ax = plt.subplots()
        input_df["Win_Probability"].plot(kind="hist", bins=10, ax=ax)
        ax.set_title("Win Probability Histogram")
        ax.set_xlabel("Win Probability (%)")
        st.pyplot(fig)

        # Player Prop Recommendations
        st.subheader("🧩 Suggested Player Prop Picks")
        props = input_df["Gematria_Alignment_Score"].apply(lambda x: "💥 HR Pick" if int(x) > 85 else "🧊 Under Hits")
        prop_table = pd.DataFrame({
            "Matchup": input_df.get("Matchup", pd.Series(["Game 1", "Game 2", "Game 3"])),
            "Prop Suggestion": props
        })
        st.table(prop_table)

        # Parlay Builder
        st.subheader("🎯 Top 3 AI-Ranked Parlays")
        top_games = input_df.sort_values(by="Win_Probability", ascending=False).head(3)
        parlay_df = top_games[["Matchup", "Win_Probability"]].reset_index(drop=True)
        parlay_df.columns = ["Matchup", "Confidence (%)"]
        st.table(parlay_df)

        csv = input_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Full Results CSV", csv, "MLBX_LIVE_Results.csv", "text/csv")
