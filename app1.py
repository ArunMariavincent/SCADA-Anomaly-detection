import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Page config must be set before any other Streamlit calls
st.set_page_config(page_title="SCADA Anomaly Detection", layout="wide")

# -------------------------------
# Load Model (SAFE)
# -------------------------------
@st.cache_resource
def load_my_model():
    # load the model file present in workspace
    return load_model("scada_model.h5", compile=False)

model = load_my_model()

st.title("🔍 SCADA Anomaly Detection Dashboard")

# -------------------------------
# Sidebar Settings
# -------------------------------
st.sidebar.header("⚙️ Settings")

time_steps = st.sidebar.slider("Time Steps", 10, 50, 20)
threshold_percentile = st.sidebar.slider("Threshold (%)", 90, 99, 95)
speed = st.sidebar.slider("Live Speed", 0.01, 0.2, 0.05)

mode = st.sidebar.radio("Mode", ["Static Analysis", "Live Monitoring"])

# -------------------------------
# Upload File
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload SCADA CSV File")

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Data")
    st.write(df.head())

    # -------------------------------
    # Preprocessing
    # -------------------------------
    df = df.select_dtypes(include=[np.number])
    df = df.dropna()

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)

    # -------------------------------
    # Create Sequences
    # -------------------------------
    def create_sequences(data, time_steps):
        X = []
        for i in range(len(data) - time_steps):
            X.append(data[i:i + time_steps])
        return np.array(X)

    X = create_sequences(data_scaled, time_steps)

    # -------------------------------
    # Prediction
    # -------------------------------
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

        ax.scatter(
            np.where(anomalies)[0],
            mse[anomalies],
            color='red',
            label='Anomalies'
        )

        ax.legend()
        st.pyplot(fig)

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Points", len(mse))
        col2.metric("Anomalies Detected", int(np.sum(anomalies)))
        col3.metric("Threshold", round(threshold, 6))

        # Alerts
        st.subheader("🚨 Alerts")
        if np.sum(anomalies) > 0:
            st.error(f"⚠️ {np.sum(anomalies)} anomalies detected!")
        else:
            st.success("✅ System Normal")

        # Results Table
        st.subheader("📋 Results")
        results = pd.DataFrame({
            "Error": mse,
            "Anomaly": anomalies
        })
        st.dataframe(results.head(100))

        # Download
        st.download_button(
            "📥 Download Results",
            results.to_csv(index=False),
            "results.csv"
        )

    # ===============================
    # 🔴 LIVE MODE
    # ===============================
    elif mode == "Live Monitoring":

        st.subheader("📡 Live Anomaly Detection")

        if st.button("▶️ Start Monitoring"):

            chart = st.line_chart()
            status = st.empty()
            metric_box = st.empty()

            live_data = []

            for i in range(len(mse)):

                value = mse[i]
                live_data.append(value)

                # Update chart
                chart.add_rows(pd.DataFrame([value]))

                # Update metrics
                metric_box.metric(
                    "Current Error",
                    round(value, 6),
                    "Anomaly" if value > threshold else "Normal"
                )

                # Alerts
                if value > threshold:
                    status.error(f"🚨 Anomaly at step {i}")
                else:
                    status.success(f"✅ Normal at step {i}")

                time.sleep(speed)

else:
    st.info("👆 Upload a CSV file to start")