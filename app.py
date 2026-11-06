import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Page config must be set before any other Streamlit commands
st.set_page_config(page_title="SCADA Anomaly Detection", layout="wide")

# -------------------------------
# Load Model (FIXED)
# -------------------------------
@st.cache_resource
def load_my_model():
    return load_model("scada_model.h5", compile=False)

model = load_my_model()

# -------------------------------
# Page Config
# -------------------------------
st.title("🔍 SCADA Anomaly Detection Dashboard")

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("⚙️ Settings")

time_steps = st.sidebar.slider("Time Steps", 10, 50, 20)
threshold_percentile = st.sidebar.slider("Threshold (%)", 90, 99, 95)

# -------------------------------
# File Upload
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

    # -------------------------------
    # Metrics Display
    # -------------------------------
    total_points = len(mse)
    total_anomalies = np.sum(anomalies)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Points", total_points)
    col2.metric("Anomalies Detected", int(total_anomalies))
    col3.metric("Threshold", round(threshold, 6))

    # -------------------------------
    # Graph
    # -------------------------------
    st.subheader("📉 Anomaly Detection Graph")

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(mse, label="Reconstruction Error")
    ax.axhline(threshold, color='r', label="Threshold")

    # Highlight anomalies
    ax.scatter(
        np.where(anomalies)[0],
        mse[anomalies],
        color='red',
        label='Anomalies'
    )

    ax.legend()
    st.pyplot(fig)

    # -------------------------------
    # Alerts
    # -------------------------------
    st.subheader("🚨 Alerts")

    if total_anomalies > 0:
        st.error(f"⚠️ {total_anomalies} anomalies detected!")
    else:
        st.success("✅ System Normal")

    # -------------------------------
    # Results Table
    # -------------------------------
    st.subheader("📋 Results")

    results = pd.DataFrame({
        "Error": mse,
        "Anomaly": anomalies
    })

    st.dataframe(results.head(100))

    # -------------------------------
    # Download Results
    # -------------------------------
    csv = results.to_csv(index=False)

    st.download_button(
        label="📥 Download Results",
        data=csv,
        file_name="anomaly_results.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Upload a CSV file to start anomaly detection")