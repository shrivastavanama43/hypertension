# streamlit_hypertension_watch_app.py
# Single-file Streamlit app that:
# - connects (or simulates) a smartwatch via Bluetooth (Bleak) to read vitals
# - displays live readings alongside normal/average values for comparison
# - asks symptom questions
# - trains/uses a simple ML model (RandomForest) on synthetic data to predict future hypertension risk
# - shows a horizontal risk bar that changes color from green->yellow->red by risk percentage
# - provides personalized lifestyle guidance based on risk
# - includes a placeholder function to look up recommended doctors using Google Places API (requires user API key)

# Requirements:
# pip install streamlit scikit-learn joblib pandas numpy bleak
# Note: Bleak works on Windows/Linux/macOS; run Streamlit on the same machine that has Bluetooth access.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import asyncio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from io import BytesIO

# Optional: bleak for Bluetooth; import only if available
try:
    from bleak import BleakClient, BleakScanner
    BLEAK_AVAILABLE = True
except Exception:
    BLEAK_AVAILABLE = False

# ----------------------------- Helper utilities -----------------------------
MODEL_PATH = "hypertension_rf_model.pkl"

NORMALS = {
    "systolic_bp": 120,
    "diastolic_bp": 80,
    "resting_hr": 70,
    "spo2": 98,
}

st.set_page_config(page_title="PulseGuard — Hypertension Watch", layout="wide")

# CSS for look and feel (red & white theme with professional style)
st.markdown(
    """
    <style>
    body {background-color: #ffffff;}
    .sidebar .sidebar-content {background-color: #fff9f9}
    .big-title{font-size:34px; font-weight:700; color:#b30000}
    .subtitle{color:#4b4b4b}
    .card{background: linear-gradient(90deg,#ffffff,#fff2f2); padding:16px; border-radius:12px; box-shadow: 0 6px 18px rgba(179,0,0,0.08)}
    .muted{color:#6b6b6b}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Synthetic model training -----------------------------

def generate_synthetic_dataset(n=3000, random_state=42):
    rng = np.random.RandomState(random_state)
    age = rng.randint(18, 80, size=n)
    bmi = rng.normal(25, 4, size=n).clip(16, 45)
    resting_hr = rng.normal(70, 10, size=n).clip(40, 120)
    systolic = (110 + (age - 30) * 0.5 + (bmi - 24) * 0.8 + (resting_hr - 70) * 0.3 + rng.normal(0, 8, n)).astype(int)
    diastolic = (72 + (age - 30) * 0.2 + (bmi - 24) * 0.4 + rng.normal(0,6,n)).astype(int)
    activity_level = rng.choice([0,1,2], size=n, p=[0.3,0.5,0.2])  # 0 low,1 medium,2 high
    sleep_hours = rng.normal(7,1.2,size=n).clip(3,10)
    stress_level = rng.choice([0,1,2], size=n, p=[0.25,0.5,0.25])
    smoker = rng.binomial(1, 0.15, size=n)
    alcohol = rng.binomial(1, 0.25, size=n)
    family_history = rng.binomial(1, 0.2, size=n)

    # Create a target: chance of developing HTN in 5 years (binary for training, but we'll use predicted probability)
    risk_score = (0.02*(age-40).clip(0) + 0.04*(bmi-24).clip(0) + 0.03*(systolic-120).clip(0) +
                  0.06*smoker + 0.05*alcohol + 0.06*family_history + 0.03*(stress_level))
    prob = 1/(1+np.exp(-risk_score))
    hypertension_future = (prob > 0.4).astype(int)

    df = pd.DataFrame({
        "age": age,
        "bmi": bmi,
        "resting_hr": resting_hr,
        "systolic": systolic,
        "diastolic": diastolic,
        "activity_level": activity_level,
        "sleep_hours": sleep_hours,
        "stress_level": stress_level,
        "smoker": smoker,
        "alcohol": alcohol,
        "family_history": family_history,
        "future_htn": hypertension_future,
    })
    return df


def train_and_save_model(path=MODEL_PATH, force_retrain=False):
    if os.path.exists(path) and not force_retrain:
        try:
            model = joblib.load(path)
            return model
        except Exception:
            pass
    df = generate_synthetic_dataset()
    X = df.drop(columns=["future_htn"])
    y = df["future_htn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    numeric_cols = ["age", "bmi", "resting_hr", "systolic", "diastolic", "sleep_hours"]
    cat_cols = ["activity_level", "stress_level", "smoker", "alcohol", "family_history"]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=120, random_state=42))
    ])

    pipeline.fit(X_train[numeric_cols + cat_cols], y_train)
    y_prob = pipeline.predict_proba(X_test[numeric_cols + cat_cols])[:,1]
    auc = roc_auc_score(y_test, y_prob)
    joblib.dump(pipeline, path)
    return pipeline

model = train_and_save_model()

# ----------------------------- Bluetooth (simulated) reader -----------------------------

class SimulatedWatch:
    def __init__(self):
        self.rng = np.random.RandomState(123)

    def read(self):
        # returns a dict of readings
        systolic = int(np.clip(self.rng.normal(120, 12), 90, 190))
        diastolic = int(np.clip(self.rng.normal(78, 8), 50, 120))
        hr = int(np.clip(self.rng.normal(72, 10), 45, 140))
        spo2 = int(np.clip(self.rng.normal(97,1.5), 88, 100))
        steps = int(np.clip(self.rng.normal(2000,1800), 0, 20000))
        temp = round(np.clip(self.rng.normal(36.5,0.4), 35.5, 38.5),1)
        return {
            "systolic": systolic,
            "diastolic": diastolic,
            "hr": hr,
            "spo2": spo2,
            "steps": steps,
            "skin_temp": temp,
        }

SIM_WATCH = SimulatedWatch()

async def read_from_ble(address=None, characteristic_uuid=None, timeout=5):
    # This is a placeholder. Actual characteristic parsing depends on the watch model.
    if not BLEAK_AVAILABLE:
        return None
    try:
        async with BleakClient(address) as client:
            # read characteristic (user must supply correct uuid)
            raw = await client.read_gatt_char(characteristic_uuid)
            # decode / parse raw bytes into values — left as an exercise; we will return None by default
            return None
    except Exception:
        return None

# ----------------------------- UI -----------------------------

st.title("PulseGuard — Hypertension monitor & predictor")

# Top visuals and connect button on first page
with st.container():
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("<div class='big-title'>Keep your heart in sight — Predict & prevent hypertension</div>", unsafe_allow_html=True)
        st.markdown("""
            <div class='subtitle'>Connect your smartwatch to stream vitals, ask a few quick questions, and get a personalized prediction of future hypertension risk — plus an actionable lifestyle plan.</div>
            """, unsafe_allow_html=True)
        st.write("---")
        connect_mode = st.radio("Connection mode", ["Simulated (demo)", "Bluetooth (real watch)"]) 
        if connect_mode == "Bluetooth (real watch)" and not BLEAK_AVAILABLE:
            st.warning("Bleak library is not available in this environment. To use Bluetooth, install 'bleak' and run locally.")

    with col2:
        st.image(
            "https://images.unsplash.com/photo-1511715280342-2590d3a8f8f6?auto=format&fit=crop&w=800&q=60",
            caption="Wearables make prevention possible",
            use_column_width=True,
        )

st.markdown("---")

# Sidebar for user profile inputs
with st.sidebar:
    st.header("User profile")
    age = st.number_input("Age", min_value=12, max_value=120, value=30)
    sex = st.selectbox("Sex", ["Male","Female","Other"])
    height_cm = st.number_input("Height (cm)", value=170)
    weight_kg = st.number_input("Weight (kg)", value=70)
    bmi = round(weight_kg / ((height_cm/100)**2),1) if height_cm>0 else 0
    st.markdown(f"**BMI:** {bmi}")
    st.markdown("---")
    st.header("Symptom quick-check")
    sym_polyuria = st.checkbox("Frequent urination")
    sym_headache = st.checkbox("Frequent headaches or blurred vision")
    sym_dizziness = st.checkbox("Dizziness or fainting episodes")
    sym_chest = st.checkbox("Chest discomfort or palpitations")
    family_history = st.checkbox("Family history of hypertension")

# Main tabs: Readings & Risk
tab1, tab2 = st.tabs(["Live Readings", "Risk & Guidance"]) 

# ---------- Live Readings Tab ----------
with tab1:
    st.header("Live readings from your watch")
    colA, colB = st.columns([2,1])
    with colA:
        placeholder = st.empty()
        # Read values (simulated or Bluetooth)
        if connect_mode == "Simulated (demo)":
            readings = SIM_WATCH.read()
        else:
            # try to read from BLE; synchronous calls to async function
            readings = None
            st.info("Attempting Bluetooth read — provide watch address/characteristic in the code. If none, simulation will be used.")
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                ble_res = loop.run_until_complete(read_from_ble(address=None, characteristic_uuid=None))
                readings = ble_res if ble_res else SIM_WATCH.read()
            except Exception:
                readings = SIM_WATCH.read()

        # display cards
        with placeholder.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            rcol1, rcol2, rcol3 = st.columns(3)
            rcol1.metric("Systolic BP (mmHg)", readings["systolic"], NORMALS["systolic_bp"], delta=None)
            rcol2.metric("Diastolic BP (mmHg)", readings["diastolic"], NORMALS["diastolic_bp"], delta=None)
            rcol3.metric("Resting HR (bpm)", readings["hr"], NORMALS["resting_hr"], delta=None)
            rcol1.metric("SpO₂ (%)", readings["spo2"], NORMALS["spo2"], delta=None)
            rcol2.metric("Skin Temp (°C)", readings["skin_temp"], "~36.6")
            rcol3.metric("Steps (today)", readings["steps"], "—")
            st.markdown('</div>', unsafe_allow_html=True)

    with colB:
        st.subheader("Comparison: average person")
        comp_df = pd.DataFrame({
            "Measure": ["Systolic (mmHg)", "Diastolic (mmHg)", "Resting HR (bpm)", "SpO2 (%)"],
            "You": [readings["systolic"], readings["diastolic"], readings["hr"], readings["spo2"]],
            "Average": [NORMALS["systolic_bp"], NORMALS["diastolic_bp"], NORMALS["resting_hr"], NORMALS["spo2"]],
        })
        st.table(comp_df.set_index("Measure"))

# store the latest readings in session state for use in risk tab
st.session_state["latest_readings"] = readings

# ---------- Risk & Guidance Tab ----------
with tab2:
    st.header("Risk prediction & personalized guidance")
    st.write("We use your latest watch readings plus short questionnaire answers to estimate future hypertension risk.")

    # Create feature vector
    latest = st.session_state.get("latest_readings", SIM_WATCH.read())
    features = {
        "age": age,
        "bmi": bmi,
        "resting_hr": latest["hr"],
        "systolic": latest["systolic"],
        "diastolic": latest["diastolic"],
        "activity_level": 0 if st.checkbox("I get very little exercise", key="act0") else (1 if st.checkbox("I exercise moderately", key="act1") else 2),
        "sleep_hours": float(st.slider("Average sleep hours per night", 3.0, 10.0, 7.0)),
        "stress_level": 2 if st.checkbox("I feel often stressed", key="stress2") else (1 if st.checkbox("Some stress", key="stress1") else 0),
        "smoker": 1 if st.checkbox("I smoke occasionally/regularly", key="smk") else 0,
        "alcohol": 1 if st.checkbox("I consume alcohol", key="alc") else 0,
        "family_history": 1 if family_history else 0,
    }

    feat_df = pd.DataFrame([features])
    st.subheader("Input summary")
    st.table(feat_df.T)

    prob = model.predict_proba(feat_df)[0][1]
    risk_pct = int(prob * 100)

    st.subheader(f"Estimated risk of developing hypertension in the next 5 years: {risk_pct}%")

    # Horizontal color-changing bar using HTML/CSS
    def render_risk_bar(pct):
        # gradient transitions: 0-40 green, 40-70 yellow, 70-100 red
        if pct < 40:
            color = "linear-gradient(90deg, #4caf50, #8bc34a)"
        elif pct < 70:
            color = "linear-gradient(90deg, #ffeb3b, #ff9800)"
        else:
            color = "linear-gradient(90deg, #ff7043, #d32f2f)"
        html = f"""
        <div style='width:100%; background:#f2f2f2; border-radius:12px; padding:6px;'>
          <div style='width:{pct}%; background: {color}; height:28px; border-radius:8px; text-align:center; font-weight:600;'>
            {pct}%
          </div>
        </div>
        """
        st.components.v1.html(html, height=50)

    render_risk_bar(risk_pct)

    # Explanation of risk drivers (simple feature importances from RandomForest)
    st.markdown("**Top contributing factors (model estimate):**")
    try:
        importances = model.named_steps['rf'].feature_importances_
        cols = ["age","bmi","resting_hr","systolic","diastolic","sleep_hours","activity_level","stress_level","smoker","alcohol","family_history"]
        imp_df = pd.DataFrame({"feature": cols, "importance": importances}).sort_values("importance", ascending=False).head(6)
        st.table(imp_df.set_index("feature"))
    except Exception:
        st.write("(unable to compute feature importances)")

    st.markdown("---")
    st.subheader("Personalized lifestyle recommendations")

    # Tailored guidance based on risk tier
    if risk_pct < 30:
        st.markdown("**Low risk — maintain healthy habits**")
        st.write("- Keep regular moderate-intensity exercise (150 min/week).\n- Maintain balanced diet: fruits, vegetables, whole grains, lean protein.\n- Limit sodium intake (aim <2.3g sodium/day).\n- Keep weight in healthy range, monitor blood pressure yearly.")
    elif risk_pct < 60:
        st.markdown("**Moderate risk — take action now**")
        st.write("- Increase aerobic exercise: 30–60 minutes most days.\n- Adopt DASH-style diet: rich in fruits, vegetables, low-fat dairy, reduced saturated fat.\n- Reduce sodium and processed foods; avoid sugary drinks.\n- Practice stress reduction: mindfulness, deep breathing, 7–8 hours sleep.\n- Monitor BP at home weekly and record readings.")
    else:
        st.markdown("**High risk — urgent lifestyle changes & clinical follow-up**")
        st.write("- See a physician for formal evaluation and possible medication.\n- Immediate dietary changes: DASH diet, near-zero processed food and fried food.\n- Aim for daily moderate physical activity; incorporate resistance training 2x/week.\n- Stop smoking; limit/avoid alcohol.\n- Begin home BP monitoring twice daily and bring readings to your doctor.")

    st.markdown("---")
    st.subheader("Symptom notes")
    if sym_chest or sym_dizziness:
        st.warning("You reported serious symptoms (chest discomfort or dizziness). Please seek urgent medical attention if symptoms are severe.")
    else:
        st.write("No immediate alarming symptom boxes checked — but consult your healthcare provider for persistent or worsening symptoms.")

    st.markdown("---")
    st.subheader("Find recommended doctors (optional)")
    st.info("This app does not ship with prepopulated doctor lists. To fetch local recommended cardiologists or hypertension specialists you can: 1) use Google Maps / Google Places with your city, or 2) provide a Google Places API key in the code to search programmatically.")
    st.markdown("**Programmatic placeholder (in code):** `lookup_doctors_nearby(api_key, 'New Delhi, India', 'cardiologist')` — implement with Google Places API key.")

    # Offer export of a short report
    if st.button("Export summary report (PDF)"):
        report = BytesIO()
        txt = f"PulseGuard report\nAge: {age}\nBMI: {bmi}\nLatest systolic/diastolic: {latest['systolic']}/{latest['diastolic']}\nRisk: {risk_pct}%\n"
        report.write(txt.encode())
        report.seek(0)
        st.download_button("Download report.txt", data=report, file_name="pulseguard_report.txt")

st.markdown("---")
st.caption("Note: This tool is for educational/demo purposes and does NOT replace clinical judgment. Always consult a licensed healthcare professional before making medical decisions.")

# ----------------------------- Optional helpers for advanced users -----------------------------

def lookup_doctors_nearby(api_key, location_string, query_term="cardiologist", radius=20000):
    """
    Placeholder function: if you provide a Google Places API key, this function can query 'Places Text Search'
    to return a list of doctors (name, address, phone, place_id). We intentionally do NOT call any web APIs
    in the default app — the user must supply the API key and run locally.
    """
    import requests
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": f"{query_term} in {location_string}",
        "key": api_key,
    }
    r = requests.get(url, params=params)
    return r.json()


# End of file
