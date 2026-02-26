import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

st.title("Power Plant Energy Output Predictor")

# Load dataset
dataset = pd.read_csv("Model_Selection_Project.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    random_state=0
)
model.fit(X_scaled, y)

# User Inputs
AT = st.number_input("Ambient Temperature")
V = st.number_input("Exhaust Vacuum")
AP = st.number_input("Ambient Pressure")
RH = st.number_input("Relative Humidity")

if st.button("Predict"):
    input_data = np.array([[AT, V, AP, RH]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Power Output: {prediction[0]:.2f}")