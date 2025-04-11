import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Housing Price Predictor", layout="centered")

st.title("🏡 Housing Price Predictor")
st.write("This app predicts Median House Value based on Total Rooms using Linear Regression.")

# Load and cache the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("housing.csv")
    return df.dropna(subset=['total_rooms', 'median_house_value'])

df = load_data()

# Show raw data
if st.checkbox("📂 Show raw data"):
    st.write(df.head())

# Features and target
X = df[['total_rooms']]
y = df['median_house_value']

# Train model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Sort for smoother line plot
sorted_df = df.sort_values('total_rooms')
sorted_X = sorted_df[['total_rooms']]
sorted_y_pred = model.predict(sorted_X)

# Plot
fig, ax = plt.subplots()
ax.scatter(X, y, color='blue', alpha=0.5, label='Actual')
ax.plot(sorted_X, sorted_y_pred, color='red', label='Prediction')
ax.set_xlabel("Total Rooms")
ax.set_ylabel("Median House Value")
ax.set_title("Linear Regression: Total Rooms vs Median House Value")
ax.legend()
st.pyplot(fig)

# Evaluation metrics
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

st.subheader("📊 Model Evaluation Metrics")
st.write(f"*Mean Absolute Error (MAE):* ${mae:,.2f}")
st.write(f"*Root Mean Squared Error (RMSE):* ${rmse:,.2f}")
st.write(f"*R² Score:* {r2:.4f}")

# Prediction
st.subheader("🔮 Predict House Value")
rooms_input = st.number_input("Enter total rooms:", min_value=1, value=5000, step=100)
st.caption("Note: Typical 'total_rooms' values range from around 100 to 50,000.")

if st.button("Predict"):
    prediction = model.predict([[rooms_input]])
    st.success(f"Predicted Median House Value: *${prediction[0]:,.2f}*")