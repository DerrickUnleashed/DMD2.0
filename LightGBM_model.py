import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, jaccard_score, precision_recall_curve, matthews_corrcoef, cohen_kappa_score, log_loss, auc
import lightgbm as lgb

# Load the dataset
df = pd.read_csv("DMD_combined_dataset.csv", index_col=False)

# Step 1: Load the features and labels
X = df.iloc[:, :-1]
X = X.drop(columns=["Unnamed: 0"], axis=1)
y = df.iloc[:, -1]

print("Features shape:", X.shape)
print("Labels shape:", y.shape)
print("\nLabel distribution:\n", y.value_counts())

# Step 2: Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nAfter split:")
print("  X_train shape:", X_train.shape)
print("  X_test shape:", X_test.shape)
print("  y_train distribution:\n", y_train.value_counts())
print("  y_test distribution:\n", y_test.value_counts())

# Step 3: Standardize features (fit on training set, transform both)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), 
    columns=X_train.columns, 
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), 
    columns=X_test.columns, 
    index=X_test.index
)

# Step 4: Define LightGBM classifier with best parameters
model = lgb.LGBMClassifier(
    colsample_bytree=0.8,
    learning_rate=0.1,
    max_depth=-1,
    n_estimators=100,
    num_leaves=15,
    reg_alpha=0,
    reg_lambda=0,
    subsample=0.8,
    random_state=42,
    verbosity=-1
)

# Step 5: Train the model with best parameters
model.fit(X_train_scaled, y_train)

# Step 6: Predict on test set using the best model
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Step 7: Evaluate the best model
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["NORMAL", "DMD"])
cm = confusion_matrix(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_prob)
jaccard = jaccard_score(y_test, y_pred)
precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)
mcc = matthews_corrcoef(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
loss = log_loss(y_test, y_prob)

print("\nLightGBM Classification Results (Best Model):")
print("Accuracy:", acc)
print("\nClassification Report:\n", report)
print("Confusion Matrix:\n", cm)
print("AUC-ROC Score:", auc_score)
print("Jaccard Score:", jaccard)
print("Precision-Recall AUC:", pr_auc)
print("Matthews Correlation Coefficient:", mcc)
print("Cohen's Kappa:", kappa)
print("Log Loss:", loss)

feature_importance = model.feature_importances_
feature_names = X.columns

print("\nTop 10 Most Important Features:")
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df.head(10).to_string(index=False))