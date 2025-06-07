
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import datetime

st.set_page_config(page_title="MLBX LIVE - Auto Mode", layout="wide")

st.title("👑 MLBX LIVE - AUTO MODE")
st.caption("Auto-Generated Daily Slate • No Upload Required")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("MLBX_Gematria_Model_Logistic.pkl")

model = load_model()

# Simulate 15 matchups for today's slate
matchups = [
    "Reds vs D-Backs", "Cubs vs Tigers", "Blue Jays vs Twins", "Cardinals vs Dodgers",
    "Phillies vs Pirates", "Nationals vs Rangers", "Braves vs Giants", "Astros vs Guardians",
    "Rays vs Marlins", "White Sox vs Royals", "Yankees vs Red Sox", "Padres vs Brewers",
    "Mariners vs Angels", "Rockies vs Mets", "Orioles vs Athletics"
]

np.random.seed(77)
df = pd.DataFrame({
    "Matchup": matchups,
    "MLBX_Win_Home_%": np.random.uniform(45, 70, 15).round(2),
    "Vegas_Win_Home_%": np.random.uniform(40, 68, 15).round(2),
    "Home_Starter_ERA": np.random.uniform(2.5, 5.0, 15).round(2),
    "Away_Starter_ERA": np.random.uniform(2.5, 5.0, 15).round(2),
    "Home_Bullpen_ERA": np.random.uniform(3.0, 5.5, 15).round(2),
    "Away_Bullpen_ERA": np.random.uniform(3.0, 5.5, 15).round(2),
    "Gematria_Alignment_Score": np.random.randint(65, 100, 15),
    "Narrative_Confidence_High": np.random.choice([0, 1], 15)
})

# Calculate features
df["Edge_%"] = (df["MLBX_Win_Home_%"] - df["Vegas_Win_Home_%"]).round(2)
probs = model.predict_proba(df[[
    "MLBX_Win_Home_%", "Vegas_Win_Home_%", "Edge_%",
    "Home_Starter_ERA", "Away_Starter_ERA",
    "Home_Bullpen_ERA", "Away_Bullpen_ERA",
    "Gematria_Alignment_Score", "Narrative_Confidence_High"
]])[:, 1]
df["Win_Probability"] = (probs * 100).round(2)
df["Predicted_Win"] = (probs > 0.5).astype(int)
df["Smart_Bet"] = df.apply(lambda row: "💡" if row["Edge_%"] > 7 and row["Win_Probability"] > row["Vegas_Win_Home_%"] else "", axis=1)
df["Prop"] = df["Gematria_Alignment_Score"].apply(lambda x: "💥 HR Prop" if x > 85 else "🧊 Under Hits")
df["Confidence"] = df["Gematria_Alignment_Score"].apply(lambda x: "🔥 ELITE" if x >= 90 else ("✅ STRONG" if x >= 80 else "⚠️ Low Risk"))
df["Ritual_Alignment"] = df["Gematria_Alignment_Score"].apply(lambda x: "🔺 HIGH" if x >= 90 and x % 13 == 0 else ("🔹 MODERATE" if x >= 80 else "⚪ LOW"))

# Summary
st.success(f"📆 Auto Slate Generated: {datetime.date.today().strftime('%A, %B %d, %Y')}")
st.dataframe(df[[
    "Matchup", "Win_Probability", "Edge_%", "Smart_Bet", "Prop", "Confidence", "Ritual_Alignment"
]])

# Top 3 Smart Bets
st.subheader("💡 Top Smart Bets")
top_smart = df[df["Smart_Bet"] == "💡"].sort_values(by="Edge_%", ascending=False).head(3)
st.table(top_smart[["Matchup", "Edge_%", "Win_Probability", "Ritual_Alignment"]])

# Parlay Builder
st.subheader("🎯 Top Parlay Picks")
top_parlay = df.sort_values(by="Win_Probability", ascending=False).head(3)
st.table(top_parlay[["Matchup", "Win_Probability"]])

# Download buttons
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Full Results CSV", csv, "MLBX_LIVE_Auto_Report.csv", "text/csv")

# Placeholder for optional decode notes (separate logic can be added here)
st.caption("📜 Gematria Decode Notes: Coming soon as downloadable PDF")
