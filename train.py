import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --- Configuration ---
DATA_PATH = os.path.join('data', 'students_sample.csv')
MODEL_DIR = 'models'
OUTPUT_DIR = 'output'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Load Data ---
print("1. Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

# --- 2. Feature Engineering & Preprocessing ---
print("\n2. Performing feature engineering and preprocessing...")

df.fillna(df.median(numeric_only=True), inplace=True)

weights = {'attendance': 0.2, 'assignment_score': 0.3, 'internal_marks': 0.4, 'participation': 0.1}
df['engagement_score'] = (df['attendance'] * weights['attendance'] + 
                        df['assignment_score'] * weights['assignment_score'] + 
                        df['internal_marks'] * weights['internal_marks'] + 
                        df['participation'] * 10 * weights['participation'])

le = LabelEncoder()
df['label'] = le.fit_transform(df['previous_result'])
y = df['label']

min_class_count = pd.Series(y).value_counts().min()

if min_class_count < 5:
    print("\n" + "="*60)
    warnings.warn(
        f"WARNING: The smallest class has only {min_class_count} members. "
        "Results will NOT be reliable and are for code-testing purposes only."
    )
    print("="*60 + "\n")
    cv_folds = max(2, min_class_count) 
else:
    cv_folds = 5

base_feature_cols = ['attendance', 'assignment_score', 'internal_marks', 'participation', 'engagement_score']
X_base = df[base_feature_cols]

poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(X_base)
poly_feature_names = poly.get_feature_names_out(base_feature_cols)
X = pd.DataFrame(X_poly, columns=poly_feature_names)
print(f"Expanded features from {X_base.shape[1]} to {X.shape[1]} using PolynomialFeatures.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 3. Model Pipeline ---
print("\n3. Building the model pipeline...")

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', XGBClassifier(
        random_state=42, 
        use_label_encoder=False, # Deprecated parameter
        eval_metric='mlogloss',
        # Set some reasonable default hyperparameters
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1
    ))
])

# Cross-validation is still useful for a rough performance estimate
print(f"Performing {cv_folds}-fold cross-validation...")
try:
    cv_scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring='f1_macro')
    print(f"Cross-Validation F1 Macro Scores: {np.round(cv_scores, 3)}")
    print(f"Average CV F1 Score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
except Exception as e:
    print(f"Could not perform cross-validation due to data limitations: {e}")

# --- 4. Train the Model ---
# ✅ ALTERATION: Removed GridSearchCV. We train the pipeline directly on the training data.
print("\n4. Training the model...")
pipeline.fit(X_train, y_train)
print("Model training complete.")

# --- 5. Final Evaluation ---
print("\n5. Evaluating the model on the test set...")
y_pred = pipeline.predict(X_test)

print('\nClassification Report:\n')
print(classification_report(
    y_test,
    y_pred,
    labels=np.arange(len(le.classes_)),
    target_names=le.classes_,
    zero_division=0
))

print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# --- 6. Feature Importance Analysis ---
print("\n6. Analyzing feature importances...")
xgb_model = pipeline.named_steps['classifier']
importances = xgb_model.feature_importances_

feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
plt.title('Top 15 Feature Importances')
plt.tight_layout()
importance_plot_path = os.path.join(OUTPUT_DIR, 'feature_importance.png')
plt.savefig(importance_plot_path)
print(f"Feature importance plot saved to {importance_plot_path}")

# --- 7. Save Model Artifacts ---
print("\n7. Saving final model artifacts...")
joblib.dump({
    'model_pipeline': pipeline, # Save the directly trained pipeline
    'label_encoder': le,
    'polynomial_features': poly,
    'base_feature_columns': base_feature_cols
}, os.path.join(MODEL_DIR, 'student_model_advanced.joblib'))

print(f"Saved advanced model to {os.path.join(MODEL_DIR, 'student_model_advanced.joblib')}")