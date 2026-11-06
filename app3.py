from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


st.set_page_config(
    page_title="SCADA Sentinel",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: #f7f9fc;
        border: 1px solid #e6eaf0;
        padding: 1rem;
        border-radius: .75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_my_model():
    model_path = Path(__file__).with_name("scada_model.h5")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load_model(model_path, compile=False)


def create_sequences(data: np.ndarray, time_steps: int) -> np.ndarray:
    if len(data) <= time_steps:
        return np.empty((0, time_steps, data.shape[1]), dtype=np.float32)
    return np.array(
        [data[i : i + time_steps] for i in range(len(data) - time_steps)],
        dtype=np.float32,
    )


def build_results(
    frame: pd.DataFrame, time_steps: int, threshold_percentile: int
) -> tuple[pd.DataFrame, float, np.ndarray]:
    numeric = frame.select_dtypes(include=[np.number]).dropna()
    if numeric.empty:
        raise ValueError("The CSV must contain numeric columns with valid values.")
    if len(numeric) <= time_steps:
        raise ValueError(
            f"The CSV needs more than {time_steps} valid rows; "
            f"only {len(numeric)} were found."
        )

    scaled = MinMaxScaler().fit_transform(numeric)
    sequences = create_sequences(scaled, time_steps)
    predictions = model.predict(sequences, verbose=0)
    errors = np.mean(np.square(sequences - predictions), axis=(1, 2))
    threshold = float(np.percentile(errors, threshold_percentile))
    anomalies = errors > threshold

    results = pd.DataFrame(
        {
            "Sequence": np.arange(len(errors)),
            "Reconstruction Error": errors,
            "Anomaly": anomalies,
        }
    )
    return results, threshold, numeric


model = load_my_model()

st.title(" SCADA Sentinel")
st.caption("Real-time reconstruction-error monitoring for industrial sensor data")

with st.sidebar:
    st.header("Controls")
    time_steps = st.slider("Window size", 10, 50, 20)
    threshold_percentile = st.slider("Alert threshold percentile", 90, 99, 95)
    mode = st.radio("Dashboard mode", ["Static Analysis", "Live Monitoring"])
    speed = st.slider("Live interval (seconds)", 0.01, 0.20, 0.05)
    uploaded_file = st.file_uploader("Upload SCADA CSV", type=["csv"])

if "running" not in st.session_state:
    st.session_state.running = False
if "last_anomaly" not in st.session_state:
    st.session_state.last_anomaly = None

if not uploaded_file:
    st.info("Upload a CSV file from the sidebar to begin anomaly detection.")
    st.stop()

try:
    raw_df = pd.read_csv(uploaded_file)
    results, threshold, numeric_df = build_results(
        raw_df, time_steps, threshold_percentile
    )
except (pd.errors.ParserError, UnicodeDecodeError) as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()
except ValueError as exc:
    st.warning(str(exc))
    st.stop()

anomalies = results["Anomaly"].to_numpy(dtype=bool)
errors = results["Reconstruction Error"].to_numpy()
anomaly_count = int(anomalies.sum())
anomaly_rate = anomaly_count / len(results) * 100

overview, data_tab, results_tab = st.tabs(["Overview", "Sensor Data", "Results"])

with overview:
    st.subheader("System health")
    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Windows analyzed", f"{len(results):,}")
    metric_2.metric("Anomalies", f"{anomaly_count:,}")
    metric_3.metric("Anomaly rate", f"{anomaly_rate:.2f}%")
    metric_4.metric("Alert threshold", f"{threshold:.6f}")

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(errors, color="#2563eb", linewidth=1.5, label="Reconstruction error")
    ax.axhline(
        threshold,
        color="#f59e0b",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold ({threshold_percentile}th percentile)",
    )
    anomaly_indices = np.flatnonzero(anomalies)
    ax.scatter(
        anomaly_indices,
        errors[anomalies],
        color="#dc2626",
        s=32,
        label="Anomaly",
        zorder=3,
    )
    ax.set_xlabel("Sequence")
    ax.set_ylabel("Mean squared error")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    if anomaly_count:
        st.error(f"⚠️ {anomaly_count} anomalous windows require attention.")
    else:
        st.success("✅ No anomalous windows detected.")

with data_tab:
    st.subheader("Uploaded sensor data")
    st.caption(
        f"{len(raw_df):,} source rows · {len(numeric_df.columns)} numeric signals · "
        f"{len(numeric_df):,} valid rows used"
    )
    st.dataframe(raw_df.head(100), use_container_width=True)

with results_tab:
    st.subheader("Detection results")
    st.dataframe(
        results.sort_values("Reconstruction Error", ascending=False).head(100),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download detection results",
        results.to_csv(index=False),
        "scada_anomaly_results.csv",
        "text/csv",
    )

if mode == "Live Monitoring":
    st.divider()
    st.subheader("📶 Live monitoring replay")
    start_col, stop_col, status_col = st.columns([1, 1, 3])
    if start_col.button("▶ Start", use_container_width=True):
        st.session_state.running = True
    if stop_col.button("⏹ Stop", use_container_width=True):
        st.session_state.running = False
    status_col.caption(
        "Monitoring is running." if st.session_state.running else "Monitoring is paused."
    )

    chart_placeholder = st.empty()
    metric_placeholder = st.empty()
    alert_placeholder = st.empty()

    if st.session_state.running:
        live_errors = []
        live_anomalies = []
        for index, value in enumerate(errors):
            if not st.session_state.running:
                break
            is_anomaly = bool(anomalies[index])
            live_errors.append(value)
            live_anomalies.append(is_anomaly)

            live_frame = pd.DataFrame(
                {"Reconstruction error": live_errors, "Threshold": threshold}
            )
            chart_placeholder.line_chart(live_frame, use_container_width=True)
            metric_placeholder.metric(
                "Current error", f"{value:.6f}", "Anomaly" if is_anomaly else "Normal"
            )
            if is_anomaly:
                st.session_state.last_anomaly = (index, float(value))
                alert_placeholder.error(f"🚨 Anomaly at sequence {index}")
            else:
                alert_placeholder.success(f"✅ Normal at sequence {index}")
            time.sleep(speed)

    if st.session_state.last_anomaly:
        step, value = st.session_state.last_anomaly
        st.caption(f"Last anomaly: sequence {step}, error {value:.6f}")
