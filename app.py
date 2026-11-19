import os
import joblib
import streamlit as st
from predict import predict_yield

base_dir = os.path.dirname(os.path.abspath(__file__))
areas = joblib.load(os.path.join(base_dir, "models", "areas.pkl"))
items = joblib.load(os.path.join(base_dir, "models", "items.pkl"))

st.title("Crop Yield Prediction")

area = st.selectbox("Country", areas)
item = st.selectbox("Crop", items)
year = st.number_input("Year", 1990, 2030, 2013)
rain = st.number_input("Average rainfall (mm/year)", 0, 4000, 1000)
pest = st.number_input("Pesticides (tonnes)", 0, 400000, 1000)
temp = st.number_input("Average temperature (°C)", 0, 40, 25)

if st.button("Predict Yield"):
    result = predict_yield(area, item, year, rain, pest, temp)
    st.success(f"Predicted yield: {result:,.0f} hg/ha ({result/10:,.0f} kg/ha)") 