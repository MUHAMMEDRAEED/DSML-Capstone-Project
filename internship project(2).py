#!/usr/bin/env python
# coding: utf-8

# # Title: Cancer Grade Prediction using Machine Learning
# 
# ## Author: MUHAMMED RAEED MK
# 
# ## Date : 01/07/2024
# 
# 
# 
# # Table of Contents
# 
#     1. Title Page
#     2. Table of Contents
#     3. Overview of Problem Statement
#     4. Objective
#     5. Data Loading and Preprocessing
#     6. Exploratory Data Analysis (EDA)
#     7. Feature Engineering
#     8. Model Training and Evaluation
#     9. Hyperparameter Tuning
#     10. Results and Conclusion
#     11. Model Deployment
# 
# 
# # Overview of Problem Statement
# 
# 
# The problem of cancer grade prediction is a critical issue in the field of cancer research.
# Cancer grading is a process of determining the severity of cancer based on various factors such as tumor size, lymph node involvement, and metastasis.
# Accurate cancer grade prediction can aid in diagnosis, treatment planning, and patient prognosis.
# However, cancer grade prediction is a complex task due to the involvement of multiple factors and the lack of a clear understanding of the underlying mechanisms.
# 
# 
# # Objective
# 
# 
# The objective of this project is to develop a machine learning model that accurately predicts cancer grades based on various features, including mutation statuses and patient information.
# The expected outcomes of this project are:
#      To develop a model that accurately predicts cancer grades with high accuracy and low error rates.
#      To identify the most important features that contribute to cancer grade prediction.
#      To provide insights into the relationships between mutation statuses and cancer grades.
# 

# In[36]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import joblib


# In[37]:


# Correct file paths without [1] suffix
mutations_file_path = 'TCGA_GBM_LGG_Mutations_all.csv'
info_file_path = 'TCGA_InfoWithGrade.csv'


# In[38]:


# Load the datasets
try:
    mutations_df = pd.read_csv(mutations_file_path)
    info_df = pd.read_csv(info_file_path)
    print("Files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}")


# In[39]:


# Proceed with further processing if files are loaded successfully
if 'mutations_df' in locals() and 'info_df' in locals():
    # Remove duplicate row from info dataset
    info_df_cleaned = info_df.drop_duplicates()


# In[40]:


# Convert categorical data to numerical in mutations dataset
mutations_df_cleaned = mutations_df.copy()
mutation_columns = mutations_df_cleaned.columns[7:]  # Columns with mutation status start from the 7th column

for col in mutation_columns:
    mutations_df_cleaned[col] = mutations_df_cleaned[col].map({'MUTATED': 1, 'NOT_MUTATED': 0})


# In[41]:


# EDA: Visualizations
sns.set(style="whitegrid")


# In[42]:


# Distribution of the Grade in the info dataset
plt.figure(figsize=(10, 6))
sns.countplot(x='Grade', data=info_df_cleaned)
plt.title('Distribution of Cancer Grades')
plt.xlabel('Grade')
plt.ylabel('Count')
plt.show()


# In[43]:


# Age distribution based on Grade
plt.figure(figsize=(10, 6))
sns.histplot(data=info_df_cleaned, x='Age_at_diagnosis', hue='Grade', kde=True, multiple="stack")
plt.title('Age Distribution by Cancer Grade')
plt.xlabel('Age at Diagnosis')
plt.ylabel('Count')
plt.show()


# In[44]:


# Relationship between mutation statuses and Grade
plt.figure(figsize=(10, 6))
sns.countplot(x='IDH1', hue='Grade', data=info_df_cleaned)
plt.title('IDH1 Mutation Status by Grade')
plt.xlabel('IDH1 Mutation Status')
plt.ylabel('Count')
plt.show()


# In[45]:


plt.figure(figsize=(10, 6))
sns.countplot(x='TP53', hue='Grade', data=info_df_cleaned)
plt.title('TP53 Mutation Status by Grade')
plt.xlabel('TP53 Mutation Status')
plt.ylabel('Count')
plt.show()


# In[46]:


plt.figure(figsize=(10, 6))
sns.countplot(x='ATRX', hue='Grade', data=info_df_cleaned)
plt.title('ATRX Mutation Status by Grade')
plt.xlabel('ATRX Mutation Status')
plt.ylabel('Count')
plt.show()


# In[47]:


# Correlation heatmap of numerical features in the info dataset
plt.figure(figsize=(14, 10))
correlation_matrix = info_df_cleaned.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[48]:


# Data Preprocessing: Standardize numerical features in the info dataset
scaler = StandardScaler()
info_df_cleaned[['Age_at_diagnosis']] = scaler.fit_transform(info_df_cleaned[['Age_at_diagnosis']])


# In[49]:


# Define the feature matrix (X) and target vector (y)
X = info_df_cleaned.drop(columns=['Grade'])
y = info_df_cleaned['Grade']


# In[50]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[51]:


# Feature Engineering: Example of creating a new feature (if applicable)
# For example, let's create a feature that is the square of the age
X_train['Age_squared'] = X_train['Age_at_diagnosis'] ** 2
X_test['Age_squared'] = X_test['Age_at_diagnosis'] ** 2


# In[52]:


# Model Training: Initialize the models
models = {
    'Logistic Regression': LogisticRegression(),
    'SVC': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'MLP Classifier': MLPClassifier(max_iter=300)
}


# In[53]:


# Train the models
for model_name, model in models.items():
    model.fit(X_train, y_train)


# In[54]:


# Model Evaluation
results = {}
for model_name, model in models.items():
    y_pred = model.predict(X_test)
    results[model_name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted')
    }


# In[55]:


# Display the evaluation results
results_df = pd.DataFrame(results).T
print(results_df)


# In[56]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Hyperparameter Tuning (Optional): Example for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Displaying the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)



# In[57]:


# Best parameters and score from GridSearchCV
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)


# In[58]:


# Results and Conclusion: Find the best model based on evaluation results
best_model_name = results_df['F1 Score'].idxmax()
best_model_performance = results_df.loc[best_model_name]

print("Best Model:", best_model_name)
print("Performance:", best_model_performance)


# In[59]:


# Model Deployment: Save the best model
best_model = models[best_model_name]
joblib.dump(best_model, 'best_model.pkl')


# In[60]:


import joblib

try:
    # Load the model and test with unseen data
    loaded_model = joblib.load('best_model.pkl')
    unseen_data_predictions = loaded_model.predict(X_test)
    print("Predictions on Unseen Data:", unseen_data_predictions)
except FileNotFoundError:
    print("Error loading files. Please check the file paths and try again.")
except Exception as e:
    print("An error occurred:", e)


# In[ ]:




