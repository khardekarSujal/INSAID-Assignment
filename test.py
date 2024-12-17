import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv('Fraud.csv')

# Display dataset structure and basic info
print(data.info())
print(data.head())

# Step 1: Data Cleaning
# Check for missing values
print("Missing values:\n", data.isnull().sum())

# No missing values detected. Proceed with outlier analysis.

# Check for outliers in 'amount'
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['amount'])
plt.title('Boxplot for Transaction Amount')
plt.show()

# Log-transform to reduce skewness
data['amount_log'] = np.log1p(data['amount'])

# Encode categorical variables
data = pd.get_dummies(data, columns=['type'], drop_first=True)

# Step 2: Feature Engineering
# Add features for balance discrepancies
data['orig_balance_diff'] = data['oldbalanceOrg'] - data['newbalanceOrig']
data['dest_balance_diff'] = data['newbalanceDest'] - data['oldbalanceDest']
data['flag_large_transfer'] = (data['amount'] > 200000).astype(int)

# Drop irrelevant columns
columns_to_drop = ['nameOrig', 'nameDest']
data.drop(columns=columns_to_drop, inplace=True)

# Correlation analysis to detect multi-collinearity
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Step 3: Splitting Data into Training and Testing
X = data.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y = data['isFraud']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Step 4: Model Building (Random Forest)
rf_model = RandomForestClassifier(random_state=42)
params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# GridSearch for hyperparameter tuning
grid_search = GridSearchCV(estimator=rf_model, param_grid=params, scoring='roc_auc', cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best model
best_rf_model = grid_search.best_estimator_
print("Best Parameters:\n", grid_search.best_params_)

# Step 5: Model Evaluation
y_pred = best_rf_model.predict(X_test)
y_pred_proba = best_rf_model.predict_proba(X_test)[:, 1]

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Key Predictors
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': best_rf_model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
print("Key Predictors:\n", feature_importances.head())

# Step 6: Recommendations for Prevention
print("\nRecommendations:")
print("1. Implement real-time anomaly detection systems.")
print("2. Enhance authentication mechanisms for large transactions.")
print("3. Educate users about secure transaction practices.")

# Step 7: Monitoring Effectiveness
print("\nTo measure effectiveness, monitor:")
print("- Detection rates and reduction in fraudulent activities.")
print("- Customer feedback and false positive rates.")
