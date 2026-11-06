import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="SCADA Anomaly Detection", layout="wide")

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_my_model():
    return load_model("scada_model.h5", compile=False)

model = load_my_model()

st.title("🔍 SCADA Anomaly Detection Dashboard")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("⚙️ Settings")

time_steps = st.sidebar.slider("Time Steps", 10, 50, 20)
threshold_percentile = st.sidebar.slider("Threshold (%)", 90, 99, 95)
speed = st.sidebar.slider("Live Speed", 0.01, 0.2, 0.05)

mode = st.sidebar.radio("Mode", ["Static Analysis", "Live Monitoring"])

# -------------------------------
# Session State
# -------------------------------
if "running" not in st.session_state:
    st.session_state.running = False

if "last_anomaly" not in st.session_state:
    st.session_state.last_anomaly = None

# -------------------------------
# Upload
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload SCADA CSV File")

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Data")
    st.write(df.head())

    # Preprocess
    df = df.select_dtypes(include=[np.number])
    df = df.dropna()

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)

    # Sequence
    def create_sequences(data, time_steps):
        X = []
        for i in range(len(data) - time_steps):
            X.append(data[i:i + time_steps])
        return np.array(X)

    X = create_sequences(data_scaled, time_steps)

    # Prediction
    X_pred = model.predict(X)
    mse = np.mean(np.power(X - X_pred, 2), axis=(1,2))

    threshold = np.percentile(mse, threshold_percentile)
    anomalies = mse > threshold

    # ===============================
    # 🟢 STATIC MODE
    # ===============================
    if mode == "Static Analysis":

        st.subheader("📉 Anomaly Detection Graph")

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(mse, label="Reconstruction Error")
        ax.axhline(threshold, color='r', label="Threshold")

        ax.scatter(np.where(anomalies)[0], mse[anomalies],
                   color='red', label='Anomalies')

        ax.legend()
        st.pyplot(fig)

        st.subheader("🚨 Alerts")
        st.write(f"Total Anomalies: {np.sum(anomalies)}")

    # ===============================
    # 🔴 LIVE MODE
    # ===============================
    elif mode == "Live Monitoring":

        st.subheader("📡 Live Anomaly Detection")

        colA, colB = st.columns(2)

        if colA.button("▶️ Start"):
            st.session_state.running = True

        if colB.button("⏹ Stop"):
            st.session_state.running = False

        chart = st.line_chart()
        status_box = st.empty()
        metric_box = st.empty()
        last_status_box = st.empty()
        last_anomaly_box = st.empty()

        if st.session_state.running:

            for i in range(len(mse)):

                if not st.session_state.running:
                    status_box.warning("⏹ Monitoring Stopped")
                    break

                value = mse[i]
                is_anomaly = value > threshold

                # Update chart
                chart.add_rows(pd.DataFrame([value]))

                # Current metric
                metric_box.metric(
                    "Current Error",
                    round(value, 6),
                    "🚨 Anomaly" if is_anomaly else "Normal"
                )

                # Alerts
                if is_anomaly:
                    status_box.error(f"🚨 Anomaly at step {i}")

                    # Save last anomaly
                    st.session_state.last_anomaly = (i, value)

                    # Alarm sound
                    st.markdown("""
                        <audio autoplay>
                        <source src="https://www.soundjay.com/button/beep-07.wav" type="audio/wav">
                        </audio>
                    """, unsafe_allow_html=True)

                else:
                    status_box.success(f"✅ Normal at step {i}")

                # Last status
                last_status_box.info(
                    f"Last Status → {'🚨 Anomaly' if is_anomaly else '✅ Normal'} | Step: {i}"
                )

                # 🔥 LAST DETECTED ANOMALY DISPLAY
                if st.session_state.last_anomaly:
                    step, val = st.session_state.last_anomaly
                    last_anomaly_box.warning(
                        f"⚠️ Last Anomaly → Step: {step} | Error: {round(val,6)}"
                    )
                else:
                    last_anomaly_box.success("No anomaly detected yet")

                time.sleep(speed)

else:
    st.info("👆 Upload a CSV file to start") 