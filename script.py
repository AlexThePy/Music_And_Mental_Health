#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# Load the dataset
file_path = 'S:/ML Course/Capstone 1/mxmh_survey_results.csv'
mxmh_data = pd.read_csv(file_path)

# Data Overview
print("Data Overview:")
print(mxmh_data.info())

# Drop non-feature columns
mxmh_data = mxmh_data.drop(columns=['Timestamp', 'Permissions'], errors='ignore')

# Separate features and the target variable
X = mxmh_data.drop('Age', axis=1)
y = mxmh_data['Age']

# Define categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()


# In[3]:


# Statistical Summary
print("\nStatistical Summary:")
descriptive_stats = mxmh_data.describe()
print(descriptive_stats)


# In[4]:


# Missing Values Analysis
print("\nMissing Values Analysis:")
missing_values = mxmh_data.isnull().sum()
print(missing_values)


# In[5]:


# Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(mxmh_data['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[6]:


# Hours per day spent on music
plt.figure(figsize=(10, 6))
sns.histplot(mxmh_data['Hours per day'], bins=30, kde=True)
plt.title('Distribution of Hours per Day Spent on Music')
plt.xlabel('Hours per Day')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[7]:


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


# In[8]:


# Define transformers
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


# Fill NaN values with the mean (or median, or mode) of the column
mxmh_data['Age'] = mxmh_data['Age'].fillna(mxmh_data['Age'].mean())
X = mxmh_data.drop(columns=['Age'])
y = mxmh_data['Age']


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


# In[9]:


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


# In[ ]:




