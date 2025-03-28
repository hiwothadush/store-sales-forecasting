import sys
import os
from os import P_ALL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import seaborn as sns
import logging
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
from scipy.stats import zscore
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
import importlib

def load_data(file_path):
    """Loads data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        print("data loaded successfully.")
        return data
    except Exception as e:
        print(f"error loading data: {e}")
        return None

def summarize_data(data):
    """Prints basic summary statistics of the dataset."""
    print("\n Dataset Info:")
    print(data.info())

    print("\n First 5 Rows:")
    print(data.head())

    print("\n Summary Statistics:")
    print(data.describe())

    print("\n Missing Values:")
    print(data.isnull().sum())

def visualize_missing_values(data):
    """Plots missing values using a heatmap."""
    plt.figure(figsize=(8, 5))
    sns.heatmap(data.isnull(), cmap='viridis', cbar=False)
    plt.title("Missing Values Heatmap")
    plt.show()

def handle_missing_values(data, strategy="drop", fill_value=None):
    """
    Handles missing values in a dataset.
    
    Parameters:
    - data (pd.DataFrame): The input DataFrame.
    - strategy (str): The strategy to handle missing values ("drop", "mean", "median", "mode", "fill").
    - fill_value: Custom value for filling missing data (used when strategy="fill").
    
    Returns:
    - pd.DataFrame: Processed DataFrame with missing values handled.
    """
    if strategy == "drop":
        data = data.dropna()  # Remove rows with missing values
    elif strategy == "mean":
        data = data.fillna(data.mean())  # Fill with column mean
    elif strategy == "median":
        data = data.fillna(data.median())  # Fill with column median
    elif strategy == "mode":
        data = data.fillna(data.mode().iloc[0])  # Fill with most frequent value
    elif strategy == "fill":
        if fill_value is not None:
            data = data.fillna(fill_value)  # Fill with a custom value
        else:
            raise ValueError("Please provide a fill_value when using 'fill' strategy.")
    else:
        raise ValueError("Invalid strategy! Choose from 'drop', 'mean', 'median', 'mode', or 'fill'.")
    
    print("Missing values handled successfully.")
    return data

def handle_skewness(data, column):
    """
    Checks skewness and applies transformation if data is highly skewed.

    Parameters:
    - data (pd.DataFrame): The dataset.
    - column (str): The numerical column to check and transform.

    Returns:
    - pd.Series: Transformed column if skewed, otherwise original.
    """
    original_data = data[column].dropna()
    skewness = skew(original_data)

    # Determine skewness type
    if -0.5 <= skewness <= 0.5:
        print(f"Skewness ({skewness:.4f}) is within normal range. No transformation needed.")
        return data[column]

    elif skewness > 0.5:
        print(f"Right-Skewed ({skewness:.4f}) detected. Applying log transformation.")
        transformed_data = np.log1p(original_data)  # log1p(x) = log(1 + x)

    elif skewness < -0.5:
        print(f"Left-Skewed ({skewness:.4f}) detected. Applying square root transformation.")
        transformed_data = np.sqrt(original_data)

    # If still highly skewed, apply Power Transformation
    if abs(skew(transformed_data)) > 1:
        print(f"Data is still highly skewed. Applying Power Transformation.")
        pt = PowerTransformer(method='yeo-johnson')
        transformed_data = pt.fit_transform(transformed_data.values.reshape(-1, 1)).flatten()

    # Plot before & after
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(original_data, kde=True, bins=30, ax=axes[0], color="red")
    axes[0].set_title(f"Original {column} Distribution")

    sns.histplot(transformed_data, kde=True, bins=30, ax=axes[1], color="green")
    axes[1].set_title(f"Transformed {column} Distribution")

    plt.show()

    return pd.Series(transformed_data, index=data.index)

# remove outliers using z-score
def remove_outliers(data, column, threshold=3):
    z_scores = zscore(data[column])
    abs_z_scores = np.abs(z_scores)
    return data[abs_z_scores < threshold]


def plot_correlation_matrix(data):
    """
    Plots the correlation matrix of the given data.

    Args:
        data (pd.DataFrame): The input data.
    """
    # Select only numeric features for correlation analysis
    numeric_data = data.select_dtypes(include=np.number)

    # Calculate the correlation matrix
    correlation_matrix = numeric_data.corr()

    # Plot the correlation matrix
    plt.figure(figsize=(10, 8))  # Adjust figure size if needed
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

def plot_categorical_counts(data):
    """Plots count plots for categorical variables."""
    cat_cols = data.select_dtypes(include=['object']).columns

    for col in cat_cols:
        plt.figure(figsize=(6, 4))
        sns.countplot(y=data[col], order=data[col].value_counts().index, palette="coolwarm")
        plt.title(f"Count Plot of {col}")
        plt.xlabel("Count")
        plt.ylabel(col)
        plt.show()