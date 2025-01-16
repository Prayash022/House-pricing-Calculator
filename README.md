
# House Price Prediction Using Linear Regression

## Overview
This project implements a **Linear Regression** model to predict house prices based on features such as square footage, number of bedrooms, and number of bathrooms. The dataset is preprocessed and analyzed to create a robust model for predicting house prices, allowing users to input custom house details for predictions.

---

## Requirements
To run the code, ensure you have the following:  

### Libraries:
- Python 3.8 or above
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  

### Tools:
- Google Colab or any Python IDE (like Jupyter Notebook, VS Code)  
- Dataset: `house-prices.csv` (placed in Google Drive)

### Installation of Libraries:
Run the following commands to install the required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn

### Variables:
# Columns in 'house-prices.csv' Dataset
- SqFt: Square footage of the house
- Bedrooms: Number of bedrooms
- Bathrooms: Number of bathrooms
- Price: Actual price of the house (target variable)

### Methodology:
# Step 1: Mount Google Drive and Import Libraries
from google.colab import drive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

drive.mount('/content/drive')  # Mount Google Drive

# Step 2: Load the Dataset
df = pd.read_csv('/content/drive/MyDrive/house-prices.csv')

# Step 3: Inspect and Clean the Data
print(df.info())  # Inspect structure
print(df.head())  # Preview dataset
df.drop_duplicates(inplace=True)  # Remove duplicate rows
df.fillna(0, inplace=True)  # Replace NaN values with 0

# Step 4: Feature Selection
X = df[['SqFt', 'Bedrooms', 'Bathrooms']]  # Replace with actual column names
y = df['Price']

# Step 5: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Step 8: Predict New House Price
square_feet = float(input("Enter the square footage: "))
bedrooms = int(input("Enter the number of bedrooms: "))
bathrooms = int(input("Enter the number of bathrooms: "))
new_house = np.array([[square_feet, bedrooms, bathrooms]])
predicted_price = model.predict(new_house)
print(f"Predicted price for the house: ${predicted_price[0]:,.2f}")

# Evaluation Metrics:
- Mean Squared Error (MSE): Measures average squared difference between actual and predicted prices.
- R-squared (R²): Explains how well the model captures the variance in the data.

### Results:
# Example Prediction:
Input:
   Enter the square footage: 2000
   Enter the number of bedrooms: 3
   Enter the number of bathrooms: 2
Output:
   Predicted price for the house: $350,000.00

