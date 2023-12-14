#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import numpy as np
from joblib import dump

# Load the dataset
file_path = 'S:/ML Course/Capstone 1/mxmh_survey_results.csv'
mxmh_data = pd.read_csv(file_path)

# Data Overview
print("Data Overview:")
print(mxmh_data.info())

# Separate features and the target variable before dropping non-feature columns
y = mxmh_data['Age']

# Drop non-feature columns
mxmh_data = mxmh_data.drop(columns=['Timestamp', 'Permissions', 'Age'], errors='ignore')

# Separate features and the target variable
X = mxmh_data

# Define categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()


# In[41]:


# Statistical Summary
print("\nStatistical Summary:")
descriptive_stats = mxmh_data.describe()
print(descriptive_stats)


# In[42]:


# Inspect the columns related to frequency of music genres and psychological factors
music_genre_columns = [col for col in mxmh_data.columns if 'Frequency' in col]
psychological_factors_columns = ['Anxiety', 'Depression', 'Insomnia', 'OCD']

# Create a new DataFrame with just the relevant columns for correlation analysis
analysis_data = mxmh_data[music_genre_columns + psychological_factors_columns]

# Check the data types and unique values for the music genre columns to see how they are encoded
music_genre_data_types = analysis_data[music_genre_columns].dtypes
music_genre_unique_values = {col: analysis_data[col].unique() for col in music_genre_columns}

(music_genre_data_types, music_genre_unique_values)

# Mapping for converting categorical frequency data to numerical
frequency_mapping = {
    'Never': 0,
    'Rarely': 1,
    'Sometimes': 2,
    'Very frequently': 3
}

# Apply the mapping to the music genre columns
for col in music_genre_columns:
    analysis_data.loc[:, col] = analysis_data[col].map(frequency_mapping)


# Now, calculate the correlation matrix for the relevant columns
correlation_matrix = analysis_data.corr()

# Display the correlation matrix for the psychological factors and music genres
psychological_factors_correlation = correlation_matrix[psychological_factors_columns].loc[music_genre_columns]
psychological_factors_correlation


# In[43]:


# Calculate the mean frequency scores for each music genre to determine popularity
genre_popularity = analysis_data[music_genre_columns].mean().sort_values(ascending=False)

genre_popularity


# In[44]:


# Missing Values Analysis
print("\nMissing Values Analysis:")
missing_values = mxmh_data.isnull().sum()
print(missing_values)


# In[45]:


# Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(y, bins=30, kde=True)  # Use 'y' instead of mxmh_data['Age']
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[46]:


# Hours per day spent on music
plt.figure(figsize=(10, 6))
sns.histplot(mxmh_data['Hours per day'], bins=30, kde=True)
plt.title('Distribution of Hours per Day Spent on Music')
plt.xlabel('Hours per Day')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[47]:


# Mental health indicators distributions
mental_health_columns = ['Anxiety', 'Depression', 'Insomnia', 'OCD']
plt.figure(figsize=(15, 10))
for i, col in enumerate(mental_health_columns):
    plt.subplot(2, 2, i+1)
    sns.histplot(mxmh_data[col], bins=30, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)
plt.tight_layout()
plt.show()


# In[50]:


# Reloading the csv
mxmh_data = pd.read_csv('S:/ML Course/Capstone 1/mxmh_survey_results.csv')

# Define transformers
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


# Fill NaN values with the mean of the 'Age' column
mxmh_data['Age'] = mxmh_data['Age'].fillna(mxmh_data['Age'].mean())

# Extract the target variable 'Age' before dropping it from the features DataFrame
y = mxmh_data['Age']
X = mxmh_data.drop(columns=['Age'])

# Define categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Define transformers
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Create the preprocessing pipeline for both numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Create a Linear Regression pipeline
linear_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('regressor', LinearRegression())])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the Linear Regression pipeline
linear_pipeline.fit(X_train, y_train)

# Predict and evaluate the Linear Regression model
y_pred = linear_pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Linear Regression Model MSE: {mse}')


# In[51]:


# Define categorical and numerical columns
categorical_cols = mxmh_data.select_dtypes(include=['object']).columns
numerical_cols = mxmh_data.select_dtypes(include=['float64', 'int64']).columns
numerical_cols = numerical_cols.drop('Anxiety')  # Assuming 'Anxiety' is the target

# Imputing and encoding in a pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define the target variable
y = mxmh_data['Anxiety'].values  # Replace 'Anxiety' with your target variable

# Split the dataset into features and target variable, and then into training and testing sets
X = mxmh_data.drop(columns=['Anxiety'])  # Drop the target variable to isolate features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing and training pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(random_state=42))])

# Define a hyperparameter grid
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5]
}

# Create a GridSearchCV object
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=3)

# Execute the grid search
grid_search.fit(X_train, y_train)

# Best parameters and best score
print(f'Best parameters: {grid_search.best_params_}')
best_mse = -grid_search.best_score_
print(f'Best model performance (MSE): {best_mse}')

# Predict and evaluate with the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Test Mean Squared Error: {mse}')


# In[52]:


get_ipython().run_cell_magic('writefile', 'train.py', "import pandas as pd\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.model_selection import train_test_split\nimport joblib\n\n# Load and preprocess the dataset\n# ... (Your data loading and preprocessing code here)\n\n# Define features and target\nX = mxmh_data.drop(columns=['Timestamp', 'Permissions'], errors='ignore')  # Replace 'target_column' with the actual target column name\ny = mxmh_data['Anxiety']  # Replace 'target_column' with the actual target column name\n\n# Split the dataset into training and testing sets\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\n# Initialize and train the model\nmodel = LinearRegression()\nmodel.fit(X_train, y_train)\n\n# Save the model to a file\njoblib.dump(model, 'model.pkl')\n")


# In[53]:


get_ipython().run_cell_magic('writefile', 'predict.py', "from flask import Flask, request, jsonify\nimport joblib\n\napp = Flask(__name__)\n\n# Load the model\nmodel = joblib.load('model.pkl')\n\n@app.route('/predict', methods=['POST'])\ndef predict():\n    data = request.get_json(force=True)\n    prediction = model.predict([data['features']])\n    return jsonify(prediction.tolist())\n\nif __name__ == '__main__':\n    app.run(debug=True)\n")


# In[54]:


from flask import Flask, request, jsonify
from joblib import load
from threading import Thread

app = Flask(__name__)

# Load the trained model
model = load('model.joblib')

@app.route('/')
def home():
    return "Welcome to the model prediction service!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Preprocess the input data as required, similar to how you did in the notebook
    # For example, if you expect a single feature called 'feature_input'
    input_data = [data['feature_input']]
    # Use the model to make a prediction
    prediction = model.predict([input_data])
    return jsonify({'prediction': prediction.tolist()})

# Define the function that will run the Flask app
def run_app():
    # Set the threaded argument to True to handle each request in a separate thread.
    app.run(port=6969, debug=True, use_reloader=False, threaded=True)

# Run the Flask app in a separate thread to avoid blocking the notebook
flask_thread = Thread(target=run_app)
flask_thread.start()


# In[4]:


pip freeze > requirements.txt


# In[3]:


import json
import re

# Load the current notebook
with open('Capstone1.ipynb', 'r') as f:
    nb = json.load(f)

# Extract all code cells
code_cells = [cell for cell in nb['cells'] if cell['cell_type'] == 'code']

# Extract all lines of code from code cells
code_lines = '\n'.join(['\n'.join(cell['source']) for cell in code_cells])

# Find all package import statements
imports = set(re.findall(r'^\s*(?:import|from)\s+(\S+)', code_lines, re.MULTILINE))

# Filter out Python standard library modules and submodules
# For a more comprehensive list, use `stdlibs` from `stdlib_list` package
stdlibs = set(['sys', 'os', 're', 'json'])
project_imports = imports - stdlibs

# Write to requirements.txt
with open('requirements.txt', 'w') as f:
    for imp in sorted(project_imports):
        f.write(imp + '\n')

print('Requirements written to requirements.txt')

