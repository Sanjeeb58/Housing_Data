### 🏠 Housing Price Prediction App (Streamlit)

This Streamlit web app predicts median housing prices in California based on various housing features. It uses a Random Forest Regressor trained on a cleaned version of the California housing dataset.

### 🔍 Features
- Interactive housing price prediction
- Trained using the classic California Housing dataset
- Includes:
  - Data preprocessing and missing value handling
  - Label encoding for categorical variables
  - Evaluation metrics: MAE, RMSE, R² Score
  - Accuracy check within a margin of ±$50,000
  - Deployed on [Streamlit Community Cloud](https://sanjeeb58-housing-data-app-ux8rto.streamlit.app/)

### 📁 Files
- `app.py`: Main Streamlit app
- `housing.csv`: Dataset used for training (should be in the same directory)
- `requirements.txt`: Required Python packages
