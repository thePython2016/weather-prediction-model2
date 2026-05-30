import pickle
import pandas as pd
import streamlit as st

# Load model and encoders
model = pickle.load(open('model.pkl', 'rb'))
label = pickle.load(open('label.pkl', 'rb'))

st.set_page_config(page_title="Weather Predictor", layout="wide")
st.markdown("""
    <style>
    div.stButton > button {
        background-color: #d4750d !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        background-color: #1558b0 !important;
        box-shadow: 0 4px 12px rgba(26,115,232,0.4) !important;
        transform: translateY(-1px) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Centered title via markdown
st.markdown(
    "<h1 style='text-align: center;'>Weather Prediction App</h1>",
    unsafe_allow_html=True,
)
st.markdown("---")

# Init session state
if "batch_success" not in st.session_state:
    st.session_state["batch_success"] = False
if "batch_data" not in st.session_state:
    st.session_state["batch_data"] = None


def highlight_row(row):
    """Highlight only rows where weather is rainy; leave all others transparent."""
    val_lower = str(row["Predicted Weather"]).lower().strip()
    is_rain = val_lower == "rainy" or val_lower == "rain" or (
        "rain" in val_lower and "no rain" not in val_lower
    )
    if is_rain:
        return ["background-color: #cce5ff; color: #004085;"] * len(row)
    else:
        return [""] * len(row)


def show_styled(df):
    styled = df.style.apply(highlight_row, axis=1)
    st.dataframe(styled, use_container_width=True)


# Two-column layout
col_left, col_right = st.columns(2)

# ── LEFT: Manual Input Form ──────────────────────────────────────────────────
with col_left:
    st.subheader("Manual Input Form")

    temperature = st.number_input("Temperature", key="temp", min_value=0, step=1, format="%d")
    humidity    = st.number_input("Humidity",    key="hum",  min_value=0, step=1, format="%d")
    wind_speed  = st.number_input("Wind Speed",  key="ws",   min_value=0, step=1, format="%d")
    cloud_cover = st.number_input("Cloud Cover", key="cc",   min_value=0, step=1, format="%d")
    pressure    = st.number_input("Pressure",    key="pres", min_value=0, step=1, format="%d")

    if st.button("Predict Weather", use_container_width=True):
        data = pd.DataFrame({
            "Temperature": [temperature],
            "Humidity":    [humidity],
            "Wind_Speed":  [wind_speed],
            "Cloud_Cover": [cloud_cover],
            "Pressure":    [pressure],
        })

        predict = model.predict(data)
        target  = label.inverse_transform(predict)
        data["Predicted Weather"] = target
        st.success("Prediction complete!")
        show_styled(data)

# ── RIGHT: CSV Upload ────────────────────────────────────────────────────────
with col_right:
    st.subheader("Batch Prediction via CSV Upload")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if st.button("Predict from File", use_container_width=True):
        if uploaded_file is not None:
            data    = pd.read_csv(uploaded_file)
            predict = model.predict(data)
            target  = label.inverse_transform(predict)
            data["Predicted Weather"] = target
            st.session_state["batch_success"] = True
            st.session_state["batch_data"] = data
        else:
            st.warning("Please upload a CSV file before predicting.")

    # Dismissible success banner
    if st.session_state["batch_success"]:
        msg_col, btn_col = st.columns([9, 1])
        with msg_col:
            st.success("Batch prediction complete!")
        with btn_col:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("✕", key="dismiss"):
                st.session_state["batch_success"] = False
                st.rerun()

    if st.session_state["batch_data"] is not None:
        show_styled(st.session_state["batch_data"])