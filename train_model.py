# Loan Approval Decision Tree Training Script
# This script trains a Decision Tree classifier and saves it as a pickle file

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

def load_and_preprocess_data():
    """
    Load the loan data and preprocess it for training
    """
    # Load data
    df = pd.read_csv('loan_data.csv')

    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"\nApproval rate: {(df['Loan_Status'] == 'Y').mean():.2%}")

    # Handle missing values if any
    df.fillna(df.mode().iloc[0], inplace=True)

    # Create label encoders for categorical variables
    label_encoders = {}
    categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 
                         'Self_Employed', 'Property_Area', 'Loan_Status']

    for column in categorical_columns:
        le = LabelEncoder()
        df[column + '_Encoded'] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Select features for training
    feature_columns = ['Gender_Encoded', 'Married_Encoded', 'Dependents_Encoded',
                      'Education_Encoded', 'Self_Employed_Encoded', 'ApplicantIncome',
                      'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                      'Credit_History', 'Property_Area_Encoded']

    X = df[feature_columns]
    y = df['Loan_Status_Encoded']

    return X, y, label_encoders, df

def train_model(X, y):
    """
    Train the Decision Tree model
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Create and train the Decision Tree model
    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, 
                              target_names=['Rejected', 'Approved']))

    # Feature importance
    feature_names = X.columns
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nFeature Importance:")
    print(importance)

    return model, X_test, y_test, y_test_pred

def save_model_and_encoders(model, label_encoders):
    """
    Save the trained model and label encoders
    """
    # Save the model
    with open('loan_approval_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Save the label encoders
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)

    print("\nModel and encoders saved successfully!")
    print("- loan_approval_model.pkl")
    print("- label_encoders.pkl")

def main():
    """
    Main training function
    """
    print("="*50)
    print("LOAN APPROVAL DECISION TREE TRAINING")
    print("="*50)

    # Load and preprocess data
    X, y, label_encoders, df = load_and_preprocess_data()

    # Train model
    model, X_test, y_test, y_test_pred = train_model(X, y)

    # Save model and encoders
    save_model_and_encoders(model, label_encoders)

    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("\nFiles generated:")
    print("1. loan_approval_model.pkl - Trained Decision Tree model")
    print("2. label_encoders.pkl - Label encoders for categorical variables")
    print("3. loan_data.csv - Training dataset")

    return model, label_encoders

if __name__ == "__main__":
    model, encoders = main()