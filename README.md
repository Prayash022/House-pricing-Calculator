# House Price Prediction Using Linear Regression

## Overview
This project focuses on building a predictive model to estimate house prices using **Linear Regression**. The model utilizes key features such as square footage, number of bedrooms, and number of bathrooms to predict the price of a house. By preprocessing the dataset and training the model, we achieve meaningful predictions and provide a tool for interactive user input.

---

## Requirements

### Libraries
The following Python libraries are required for this project:  
- **pandas**: For data manipulation and analysis.  
- **numpy**: For numerical computations.  
- **matplotlib** and **seaborn**: For data visualization.  
- **scikit-learn**: For model training and evaluation.

### Tools
- **Google Colab** or a Python IDE such as Jupyter Notebook or VS Code.  
- **Dataset**: A CSV file (`house-prices.csv`) containing house data, including features like square footage, number of bedrooms, number of bathrooms, and their corresponding prices.  

---

## Dataset Description
The dataset used for this project should include the following columns:  
- **SqFt**: The square footage of the house.  
- **Bedrooms**: The number of bedrooms in the house.  
- **Bathrooms**: The number of bathrooms in the house.  
- **Price**: The target variable, representing the actual price of the house.  

---

## Methodology

### 1. Data Loading and Exploration
The dataset is loaded into a DataFrame, and its structure is explored to understand its characteristics. This includes checking for missing values, identifying duplicates, and summarizing the data.

### 2. Data Cleaning
To ensure the model performs optimally, the dataset is cleaned by removing duplicate records and handling missing values. Missing values are filled with appropriate default values (e.g., zero), and unnecessary columns (if any) are dropped.

### 3. Feature Selection
Relevant features such as square footage, number of bedrooms, and bathrooms are chosen as input variables. The target variable is the house price, which the model is trained to predict.

### 4. Splitting the Dataset
The dataset is split into training and testing subsets. Typically, 70% of the data is used for training the model, while the remaining 30% is reserved for testing its performance.

### 5. Training the Model
A **Linear Regression** model is trained using the training subset. This involves finding the optimal relationship between the input features and the target variable, minimizing the error in predictions.

### 6. Model Evaluation
The model is evaluated using metrics such as:  
- **Mean Squared Error (MSE):** Measures the average squared difference between actual and predicted prices.  
- **R-squared (R²):** Indicates the proportion of variance in the target variable explained by the model.  

These metrics help assess how well the model generalizes to unseen data.

### 7. Interactive Prediction
The trained model is used to predict the price of a house based on user input. Users provide details like square footage, number of bedrooms, and bathrooms, and the model outputs the estimated price.

---

## Outputs
- **Mean Squared Error (MSE):** Represents the average prediction error in squared units. A lower value indicates better performance.  
- **R-squared (R²):** Indicates the goodness-of-fit of the model. Values closer to 1 signify a better fit.  
- **Predicted Price:** An estimated house price based on the provided features.

### Example
When users input details such as square footage, number of bedrooms, and bathrooms, the model predicts a reasonable house price based on the trained relationships in the dataset.

---

## Notes
- The column names in the dataset should align with the expected features (`SqFt`, `Bedrooms`, `Bathrooms`, `Price`). If not, update the code accordingly.  
- Ensure the dataset is clean and free of inconsistencies to maximize model accuracy.  
- This project assumes that the relationship between the features and the target variable is linear, which might not always hold true for all datasets. Consider evaluating other models if performance is insufficient.

---

## How to Use
1. Prepare the dataset (`house-prices.csv`) with the required features and upload it to a location accessible by the script (e.g., Google Drive for Colab).  
2. Load the script into your preferred Python IDE or Google Colab environment.  
3. Follow the steps outlined in the methodology to preprocess, train, and evaluate the model.  
4. Input custom house details to predict prices interactively.  

---

## Future Enhancements
- **Feature Engineering:** Incorporate additional features such as location, age of the house, and neighborhood quality to improve predictions.  
- **Non-Linear Models:** Explore other algorithms (e.g., Decision Trees, Random Forest, or Neural Networks) to handle non-linear relationships.  
- **Advanced Cleaning:** Use more sophisticated techniques for handling missing or outlier data points.  
- **User Interface:** Create a web-based or graphical user interface for easier interaction with the model.

---

## Conclusion
This project provides a simple yet effective way to predict house prices using Linear Regression. While the current approach assumes a linear relationship between features and price, future enhancements can address its limitations and improve accuracy further.
