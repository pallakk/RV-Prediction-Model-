import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer

# Load and preprocess data
data = pd.read_csv('patients_data_with_RVoutcomes.csv')
data = data.dropna(subset=['Birthday'])

# Prepare features and target
X = data.select_dtypes(exclude=['object', 'string'])
feature_names = X.columns
Y = data['RV Dysfunction']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=feature_names)

# Group RV Dysfunction
Y_grouped = Y.replace({
    'Moderate': 'High Dysfunction',
    'Severe': 'High Dysfunction',
    'Normal': 'Low Dysfunction',
    'Mild': 'Low Dysfunction'
})

# Encode the grouped labels
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y_grouped)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform Cross-Validation
def perform_cross_validation(X, y, model, cv=5):
    # Use StratifiedKFold to maintain class distribution
    stratified_kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Compute cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=stratified_kfold, scoring='accuracy')
    
    # Print cross-validation results
    print(f"\nCross-Validation Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_scores

# Initialize the model
log_reg = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=500, random_state=42)

# Perform cross-validation
cv_scores = perform_cross_validation(X_scaled, Y_encoded, log_reg)

# Visualize Cross-Validation Scores
plt.figure(figsize=(10, 6))
plt.boxplot(cv_scores)
plt.title('Cross-Validation Accuracy Scores')
plt.ylabel('Accuracy')
plt.show()

# Detailed Cross-Validation with Per-Fold Metrics
def detailed_cross_validation(X, y, model, cv=5):
    stratified_kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    fold_metrics = []
    
    for fold, (train_index, val_index) in enumerate(stratified_kfold.split(X, y), 1):
        # Split data
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Check class distribution in train and validation sets
        print(f"\nFold {fold}:")
        print("Training Labels Distribution:", np.bincount(y_train))
        print("Validation Labels Distribution:", np.bincount(y_val))
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict and compute metrics
        y_pred = model.predict(X_val)
        
        # Check prediction class distribution
        print("Predicted Labels Distribution:", np.bincount(y_pred))
        
        accuracy = accuracy_score(y_val, y_pred)
        conf_matrix = confusion_matrix(y_val, y_pred)
        
        # Store metrics
        fold_metrics.append({
            'Fold': fold,
            'Accuracy': accuracy,
            'Confusion Matrix': conf_matrix
        })
        
        print(f"Accuracy: {accuracy}")
        print("Confusion Matrix:")
        print(conf_matrix)
    
    return fold_metrics

# Run detailed cross-validation
detailed_metrics = detailed_cross_validation(X_scaled, Y_encoded, log_reg)

# Compute and display feature importances
def compute_feature_importance(X, y, model, cv=5):
    # Use StratifiedKFold to maintain class distribution
    stratified_kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    feature_importances = []
    
    for train_index, _ in stratified_kfold.split(X, y):
        # Extract training data
        X_train, y_train = X[train_index], y[train_index]
        
        # Fit model and extract coefficients
        model.fit(X_train, y_train)
        feature_importances.append(model.coef_[0])
    
    # Compute average feature importance
    avg_importances = np.mean(feature_importances, axis=0)
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Average Importance': np.abs(avg_importances)
    }).sort_values('Average Importance', ascending=False)
    
    return feature_importance_df

# Compute and display feature importances
feature_importances = compute_feature_importance(X_scaled, Y_encoded, log_reg)
print("\nAverage Feature Importances Across Folds:")
print(feature_importances)

# Visualize Feature Importances
plt.figure(figsize=(12, 6))
feature_importances.plot(kind='bar', x='Feature', y='Average Importance')
plt.title('Average Feature Importance Across Cross-Validation Folds')
plt.xlabel('Features')
plt.ylabel('Average Absolute Coefficient')
plt.tight_layout()
plt.show()
