import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Load and preprocess the dataset
# ... (Your data loading and preprocessing code here)

# Define features and target
X = mxmh_data.drop(columns=['Timestamp', 'Permissions'], errors='ignore')  # Replace 'target_column' with the actual target column name
y = mxmh_data['target_column']  # Replace 'target_column' with the actual target column name

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'model.pkl')
