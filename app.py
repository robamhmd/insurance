"""Streamlit app: predicts yearly medical insurance charges."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from features import add_features

MODELS = Path(__file__).parent / "models"

st.set_page_config(page_title="Insurance Charge Predictor", page_icon="🏥")


@st.cache_resource
def load():
    return (joblib.load(MODELS / "preprocessor.pkl"),
            joblib.load(MODELS / "model.pkl"),
            joblib.load(MODELS / "metrics.pkl"))


preprocessor, model, info = load()

st.title("🏥 Medical Insurance Charge Predictor")
st.write("Enter a person's details to estimate their yearly medical insurance charge (USD).")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=64, value=35, step=1)
    sex = st.selectbox("Sex", ["female", "male"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=28.0, step=0.1)
with col2:
    children = st.number_input("Children covered", min_value=0, max_value=10, value=0, step=1)
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

if st.button("Predict", type="primary"):
    row = pd.DataFrame([{"age": age, "sex": sex, "bmi": bmi, "children": children,
                         "smoker": smoker, "region": region}])
    X = preprocessor.transform(add_features(row))
    pred = float(model.predict(X)[0])
    st.success(f"Estimated yearly charge: **${pred:,.2f}**")
    mae = info["test"]["MAE"]
    st.caption(f"Typical error on unseen data is about ±${mae:,.0f} (test MAE), "
               f"so treat this as an estimate, not a quote.")

with st.expander("About the model"):
    st.write(f"Gradient Boosting Regressor (log-target), tuned with grid search. "
             f"Best params: `{info['best_params']}`. "
             f"Test R² = {info['test']['R2']:.3f}, test MAE = ${info['test']['MAE']:,.0f}.")
