# DSML-Capstone-Project
Cancer Grade Classification with Machine Learning

This project involves classifying cancer grades using various machine learning algorithms. The data used for this project is sourced from TCGA (The Cancer Genome Atlas) and contains information on mutations and other clinical details.


Table of Contents

 Overview
 Installation
 Data Preparation
 Exploratory Data Analysis (EDA)
 Data Preprocessing
 Model Training
 Model Evaluation
 Hyperparameter Tuning
 Conclusion
 Model Deployment


Objective

The primary objective of this project is to develop a robust machine learning pipeline to classify cancer grades using clinical and mutation data from TCGA (The Cancer Genome Atlas). By achieving this objective, we aim to:

    Data Acquisition and Integration:
        Load and integrate mutation and clinical data from CSV files.
        Clean and preprocess the data to ensure it is suitable for analysis and modeling.

    Exploratory Data Analysis (EDA):
        Conduct a thorough EDA to understand the distribution of cancer grades, and the relationships between clinical features and mutation statuses.
        Visualize these relationships using plots to gain insights into the data.

    Data Preprocessing:
        Convert categorical data into numerical format where necessary.
        Standardize numerical features to ensure uniformity and improve model performance.
        Split the data into training and testing sets to evaluate model performance accurately.

    Feature Engineering:
        Create new features that could potentially enhance the model's predictive power.
        For example, derive new features such as the square of the age at diagnosis to capture non-linear relationships.

    Model Training and Evaluation:
        Implement and train various machine learning models, including Logistic Regression, SVC, Decision Tree, Random Forest, and MLP Classifier.
        Evaluate the performance of these models using metrics such as accuracy, precision, recall, and F1 score.
        Identify the best performing model based on these evaluation metrics.

    Hyperparameter Tuning:
        Perform hyperparameter tuning using GridSearchCV to optimize the parameters of the selected model(s) and enhance their performance.

    Model Selection and Conclusion:
        Determine the best model based on evaluation results and hyperparameter tuning.
        Summarize the performance and findings, providing insights into which model performs best for cancer grade classification.

    Model Deployment:
        Save the best performing model using joblib for future deployment and predictions.
        Load and test the saved model with unseen data to validate its effectiveness and reliability.

    Future Work and Improvements:
        Discuss potential improvements and extensions to the project, such as incorporating additional features, exploring more advanced models, or applying different data preprocessing techniques.
        
By following these steps, we aim to develop a comprehensive machine learning solution for cancer grade classification that can aid in clinical decision-making and improve patient outcomes.

