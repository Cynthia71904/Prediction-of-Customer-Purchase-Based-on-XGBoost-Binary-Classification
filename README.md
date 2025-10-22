# Prediction-of-Customer-Purchase-Based-on-XGBoost-Binary-Classification

# Project Overview

This project demonstrates how to use XGBoost for binary classification. The goal is to predict whether a customer will make a purchase based on various features like age, income, and purchase history.

# Requirements

Before running the code, you'll need to install the following Python packages:

'xgboost'

'scikit-learn'

'matplotlib'

'seaborn'

You can install them with this command:

'''pip install xgboost scikit-learn matplotlib seaborn'''

# Features

Data Generation: A synthetic dataset is created using 'sklearn.datasets.make_classification'.

Model Training: The dataset is split into training and testing sets. XGBoost is then used to train a binary classification model.

Evaluation: The model's performance is evaluated using accuracy and visualized with confusion matrices and learning curves.

# How to Run the Project

Clone the repository to your local machine.

Run the main script with the command 'python CBpredi.py'.

Visualizations Included

Confusion Matrix: Visualizes how well the model is predicting the correct classes.

Learning Curves: Plots the model's log-loss over training iterations to monitor overfitting or underfitting.

# License

This project is licensed under the MIT License.
