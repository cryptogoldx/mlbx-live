
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="MLBX LIVE - VIP Dashboard", layout="wide")

st.markdown("""
<style>
    .main {
        background-color: #0d1117;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

st.title("👑 MLBX LIVE - VIP DASHBOARD")
st.caption("Smart Bets, ROI, Props, and Prop Parlay Builder")

@st.cache_resource
def load_model():
    return joblib.load("MLBX_Gematria_Model_Logistic.pkl")

model = load_model()

uploaded_file = st.file_uploader("📤 Upload MLBX Input CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required = [
        "MLBX_Win_Home_%", "Vegas_Win_Home_%", "Edge_%",
        "Home_Starter_ERA", "Away_Starter_ERA",
        "Home_Bullpen_ERA", "Away_Bullpen_ERA",
        "Gematria_Alignment_Score", "Narrative_Confidence_High"
    ]

    if not all(col in df.columns for col in required):
        st.error("❌ Missing required columns.")
    else:
        df[required] = df[required].fillna(0)
        probs = model.predict_proba(df[required])[:, 1]
        preds = model.predict(df[required])

        df["Predicted_Win"] = preds
        df["Win_Probability"] = (probs * 100).round(2)
        df["Best_Value_Pick"] = df["Edge_%"].apply(lambda x: "✅" if float(x) > 7 else "")
        df["Smart_Bet"] = df.apply(lambda row: "💡" if float(row["Edge_%"]) > 7 and float(row["Win_Probability"]) > float(row["Vegas_Win_Home_%"]) else "", axis=1)

        # ROI
        st.success("✅ VIP Data Loaded")
        win_count = int(df["Predicted_Win"].sum())
        loss_count = len(df) - win_count
        st.metric("🧮 Win Count", win_count)
        st.metric("💥 Loss Count", loss_count)

        st.subheader("📈 ROI Tracker (1 Unit Flat Betting)")
        smart_bets = df[df["Smart_Bet"] == "💡"]
        smart_wins = smart_bets["Predicted_Win"].sum()
        smart_losses = len(smart_bets) - smart_wins
        total_units = smart_wins * 1 - smart_losses
        roi_percent = round((total_units / len(smart_bets)) * 100, 2) if len(smart_bets) > 0 else 0
        st.write(f"**Smart Bets Placed:** {len(smart_bets)}")
        st.write(f"**Wins:** {int(smart_wins)} | **Losses:** {int(smart_losses)}")
        st.write(f"**Units Won:** {total_units} | **ROI:** {roi_percent}%")

        # Smart Bets
        st.subheader("💡 Smart Bets")
        st.dataframe(smart_bets[["Matchup", "Edge_%", "Win_Probability", "Smart_Bet"]])

        # Props
        st.subheader("🔥 Player Prop Suggestions")
        df["Prop"] = df["Gematria_Alignment_Score"].apply(lambda x: "💥 HR Prop" if int(x) > 85 else "🧊 Under Hits")
        st.dataframe(df[["Matchup", "Gematria_Alignment_Score", "Prop"]])

        # Prop Parlay Builder
        st.subheader("🧩 Prop Parlay Builder")
        elite_props = df[df["Gematria_Alignment_Score"] >= 90].head(2)
        strong_props = df[(df["Gematria_Alignment_Score"] < 90) & (df["Gematria_Alignment_Score"] >= 80)].head(2)
        prop_parlay = pd.concat([elite_props, strong_props])[["Matchup", "Gematria_Alignment_Score", "Prop"]].copy()

        def rate_confidence(score):
            if score >= 90:
                return "🔥 ELITE"
            elif score >= 80:
                return "✅ STRONG"
            else:
                return "⚠️ Low Risk"

        prop_parlay["Confidence"] = prop_parlay["Gematria_Alignment_Score"].apply(rate_confidence)
        st.table(prop_parlay)

        # Parlay
        st.subheader("🎯 Top Parlay Builder")
        top_parlay = df.sort_values(by="Win_Probability", ascending=False).head(3)
        st.table(top_parlay[["Matchup", "Win_Probability"]].rename(columns={"Win_Probability": "Confidence (%)"}))

        # Download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download All Results", csv, "MLBX_VIP_Dashboard_Results.csv", "text/csv")
