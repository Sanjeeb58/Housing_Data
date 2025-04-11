import pandas as pd

file_path = "housing.csv"  
df = pd.read_csv(file_path)

df.head(), df.info()

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

print("Step 1: Load and prepare the dataset...")
# Fill missing values
df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)

# Encode the categorical feature
le = LabelEncoder()
df['ocean_proximity'] = le.fit_transform(df['ocean_proximity'])

# Feature columns and target
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity']
target = 'median_house_value'

X = df[features]
y = df[target]

print("Step 2: Split into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Step 3: Train the Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Step 4: Make predictions...")
y_pred = model.predict(X_test)

print("Step 5: Evaluate the model...")
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Check margin of error (±50000)
within_margin = np.abs(y_pred - y_test) <= 50000
accuracy_margin = np.mean(within_margin)

evaluation = {
    "Mean Absolute Error": round(mae, 2),
    "Root Mean Squared Error": round(rmse, 2),
    "R² Score": round(r2, 2),
    "Accuracy within ±$50,000": round(accuracy_margin * 100, 2)
}

# Sample predictions for sanity check
comparison = pd.DataFrame({
    'Actual Value': y_test.iloc[:10].values,
    'Predicted Value': y_pred[:10].round(2),
    'Error': (y_pred[:10] - y_test.iloc[:10]).round(2)
})

evaluation, comparison
