#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder


# In[32]:


# Load data
data = pd.read_csv('patients_data_with_RVoutcomes.csv')

# Clean data
data = data.dropna(subset=['Birthday', 'RV Dysfunction'])
data = data[data['RV Dysfunction'] != '0']

# Select features
columns_to_exclude = ['patid', 'patkey', 'rhcId','RHCDate', 'TTEDate', 'AVr_str', 'PVr_str', 'TVr_str', 'MVr_str']
data = data.drop(columns=columns_to_exclude)


# In[ ]:


def calculate_cardiac_indices(df):
    """
    Calculate cardiac indices based on available variables
    
    Parameters:
    df (pandas.DataFrame): Input dataframe with RHC and TTE variables
    
    Returns:
    pandas.DataFrame: Dataframe with additional cardiac indices columns
    """
    # Calculate mean blood pressure (assuming using non-invasive or systemic arterial measurements)
    df['mean_BP'] = (df['NIBPs_vitals'] + 2 * df['NIBPd_vitals']) / 3
    
    # Estimate Body Surface Area (BSA) using Mosteller formula
    # Requires Height in cm and Weight in kg
    df['BSA'] = np.sqrt((df['Height'] * df['Weight']) / 3600)
    
    # 1. Left Ventricle Stroke Work Index (LVSWI)
    # Using mean blood pressure and PCW as LAP approximation
    df['LVSWI'] = (df['mean_BP'] - df['PCW']) * 0.0136 / df['BSA']
    
    # 2. Right Ventricle Stroke Work Index (RVSWI)
    # Using Right Atrial Mean Pressure (RAm) and PCW
    df['RVSWI'] = (df['RAm'] - df['PCW']) * 0.0136 / df['BSA']
    
    # 3. LV Stiffness 
    # Using available LV diameter measurements
    # Note: This is an approximation and might need clinical validation
    df['LV_stiffness'] = (df['PCW'] * (df['LVIDd'] / 2)) / ((df['LVIDs'] - df['LVIDd']) / df['LVIDd'])
    
    # 4. Passive Cardiac Index
    # Using Cardiac Output and PCW
    df['Passive_Cardiac_Index'] = df['CO_td'] / (df['PCW'] * df['BSA'])
    
    return df
data = calculate_cardiac_indices(data)


# In[33]:


missing_percentages = data.isnull().mean() * 100

# Identify columns with more than 20% missing data
columns_to_drop = missing_percentages[missing_percentages > 20].index

# Drop these columns from the dataset
data = data.drop(columns=columns_to_drop)

# Display the columns dropped
print(f"Columns dropped: {columns_to_drop.tolist()}")


# In[34]:


#drop patients where more than 20% missing
missing_percentage = data.isnull().mean(axis=1)

# Keep only the rows where missing percentage is <= 0.2 (20%)
df_cleaned = data[missing_percentage <= 0.2]

# Reset index if desired
data = df_cleaned.reset_index(drop=True)


# In[35]:


# Convert 'Birthday' to Age
if 'Birthday' in data.columns:
    data['Birthday'] = pd.to_datetime(data['Birthday'], format="%d-%b-%y", errors='coerce')
    data = data.dropna(subset=['Birthday'])  # Drop rows where Birthday conversion failed

    today = pd.to_datetime('today')
    data['Age'] = (today - data['Birthday']).dt.days / 365.25  

data = data.drop(columns=['Birthday'], errors='ignore')

# Select numerical features for X
X = data.select_dtypes(include=[np.number]).drop(columns=['RV Dysfunction'], errors='ignore')


# In[36]:


from sklearn.impute import KNNImputer
# Initialize the KNN imputer
# You can adjust n_neighbors as needed
knn_imputer = KNNImputer(n_neighbors=5)

# Get the feature names for later use
feature_names = X.columns

# Apply KNN imputation to your features
X_imputed = knn_imputer.fit_transform(X)

# Convert back to DataFrame to preserve column names
X = pd.DataFrame(X_imputed, columns=feature_names)

# Then continue with standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[37]:


# Encode target variable into binary labels
Y = data['RV Dysfunction'].replace({
    'Moderate': 'High Dysfunction',
    'Severe': 'High Dysfunction',
    'Normal': 'Low Dysfunction',
    'Mild': 'Low Dysfunction'
})

label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_encoded, test_size=0.2, random_state=42)


# In[38]:


print("Class distribution:", np.bincount(y_train))
print("Classes",np.unique(Y) )


# In[39]:


def train_and_evaluate(model, model_name, X_train, Y_train, X_test, Y_test):
    """
    Trains and evaluates a classification model.
    Prints classification report, confusion matrix, and calculates ROC AUC.
    """
    print(f"\n--- {model_name} ---")

    # Train the model
    model.fit(X_train, Y_train)

    # Make predictions
    Y_pred = model.predict(X_test)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(Y_test, Y_pred))

    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(Y_test, Y_pred)
    cm_df = pd.DataFrame(cm)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='YlGnBu')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # Calculate ROC AUC
    if hasattr(model, "predict_proba"):  # Ensure the model supports probability predictions
        Y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1
        roc_auc = roc_auc_score(Y_test, Y_pred_proba)

        print(f"ROC AUC: {roc_auc:.4f}")

        # Plot ROC Curve
        fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()
    
    return model


# In[40]:


def tune_hyperparameters(X_train, y_train, n_iter=20, cv=5, random_state=42):
    """
    Tunes hyperparameters for a Random Forest classifier using RandomizedSearchCV.
    
    Parameters:
    - X_train: Training feature matrix.
    - y_train: Training labels.
    - n_iter: Number of random parameter sets to try (default: 20).
    - cv: Number of cross-validation folds (default: 5).
    - random_state: Random seed for reproducibility (default: 42).
    
    Returns:
    - best_model: The trained Random Forest model with the best hyperparameters.
    - best_params: The best hyperparameter combination found.
    """
    
    # Define the hyperparameter search space
    param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'min_samples_split': [2, 3, 5, 7, 10, 15, 20],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'max_features': [None, 'sqrt', "log2"],
    'bootstrap': [True, False],
    }

    
    # Initialize the Random Forest classifier
    rf = RandomForestClassifier(random_state=random_state)
    
    # Perform Randomized Search with Cross-Validation
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='roc_auc',
        random_state=random_state,
        n_jobs=-1,
        error_score='raise'
    )
    
    # Fit the model
    random_search.fit(X_train, y_train)
    
    # Extract the best model and parameters
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    return best_model, best_params


# In[41]:


# Train and evaluate the default Random Forest model
default_model = RandomForestClassifier(random_state=42)
default_model = train_and_evaluate(default_model, "Default Random Forest", X_train, y_train, X_test, y_test)


# In[45]:


# Tune hyperparameters for the Random Forest model
best_model, best_params = tune_hyperparameters(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")


# In[46]:


# Train and evaluate the Random Forest model with tuned hyperparameters
tuned_model = train_and_evaluate(best_model, "Tuned Random Forest", X_train, y_train, X_test, y_test)


# In[47]:


def plot_feature_importance(model, feature_names, title="Feature Importance", top_n=20, top_n_true=False):
    """
    Plot feature importance from a Random Forest model.
    
    Parameters:
    - model: Trained Random Forest model
    - feature_names: List of feature names
    - title: Plot title
    - top_n: Number of top features to display
    """
    # Get feature importance from the model
    importances = model.feature_importances_
    print(max(importances))
    # Create a DataFrame for better visualization
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    
    plot_df = feature_importance_df
    if top_n_true:
        plot_df = feature_importance_df.head(top_n)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=plot_df)
    plt.title(title)
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()
    
    return feature_importance_df



# In[48]:


print("\nFeature Importance for Default Random Forest:")
importance_df_default = plot_feature_importance(default_model, X.columns, "Default Random Forest Feature Importance", False)

# After training the tuned model
print("\nFeature Importance for Tuned Random Forest:")
importance_df_tuned = plot_feature_importance(best_model, X.columns, "Tuned Random Forest Feature Importance", False) 


# In[49]:


print("\nFeature Importance for Default Random Forest:")
importance_df_default = plot_feature_importance(default_model, X.columns, "Default Random Forest Top 20 Feature Importance", 20, True)

# After training the tuned model
print("\nFeature Importance for Tuned Random Forest:")
importance_df_tuned = plot_feature_importance(best_model, X.columns, "Tuned Random Forest Top 20 Feature Importance",20, True) 


# In[ ]:




