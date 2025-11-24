Iris Species Classification Project
Project Overview
This project implements a machine learning model to classify iris flower species based on their sepal and petal measurements. It uses the classic Iris dataset and applies logistic regression for classification. The project includes data exploration, preprocessing, model training, evaluation, and a command-line interface for making predictions on new data samples.

Dataset
The Iris dataset consists of 150 samples with 4 features each: sepal length, sepal width, petal length, and petal width. There are three target species classes:

Setosa

Versicolor

Virginica

Key Steps
Data Loading and Exploration

Loading the Iris dataset from scikit-learn

Creating a DataFrame for easier visualization and analysis

Displaying dataset summary statistics and info

Visualizing feature relationships with pairplots and correlation heatmap

Data Preprocessing

Splitting the dataset into training (80%) and testing (20%) sets with stratification to maintain class distribution

Standardizing features using StandardScaler for better model performance

Model Training

Training a Logistic Regression classifier on the scaled training data

Model hyperparameter: max_iter=200 to ensure convergence

Model Evaluation

Predicting species on the test set

Computing accuracy, classification report (precision, recall, F1-score), and displaying the confusion matrix visually using a heatmap

Prediction Function

Defining a function to predict species given new sepal and petal measurements

Provides an example prediction to demonstrate usage

Command Line Interface

CLI for user to input flower measurements and get a species prediction interactively

How to Run
Ensure the necessary Python packages are installed: numpy, pandas, matplotlib, seaborn, scikit-learn

Run the script: python iris_classifier.py

Follow the CLI prompts to predict the iris species of a flower by entering sepal and petal dimensions

Results
The classifier achieves high accuracy on the test set (typically >95%)

The confusion matrix and classification report provide insights into model performance for each species

Learning Outcomes
Hands-on experience with a classic dataset and supervised learning

Understanding data preprocessing techniques like train-test split and feature scaling

Implementing logistic regression and evaluating model metrics

Visualizing data for exploratory analysis

Writing reusable prediction functions and user-friendly CLI interaction

Future Improvements
Experiment with other classification algorithms (e.g., SVM, Random Forest)

Optimize hyperparameters with grid search or cross-validation

Deploy the model with a web interface for easier accessibility

