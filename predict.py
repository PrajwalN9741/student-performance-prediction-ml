import joblib
import pandas as pd
import numpy as np

# --- 1. Load the Saved Artifacts ---
# The path must point to where your joblib file is saved.
MODEL_PATH = 'models/student_model_advanced.joblib' 
try:
    artifacts = joblib.load(MODEL_PATH)
    print("✅ Model artifacts loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: Model file not found at '{MODEL_PATH}'.")
    print("Please make sure you have run the 'train.py' script first.")
    exit()

# Extract the individual components
model_pipeline = artifacts['model_pipeline']
label_encoder = artifacts['label_encoder']
poly_features = artifacts['polynomial_features']
base_feature_cols = artifacts['base_feature_columns']

print("\nLoaded Components:")
print(f"-> Model Pipeline: {model_pipeline}")
print(f"-> Label Encoder Classes: {label_encoder.classes_}")


# --- 2. Prepare New Data for Prediction ---
# Create a sample of a new student to predict.
# The keys must match the original feature names.
new_student_data = {
    'attendance': [92.5],
    'assignment_score': [88.0],
    'internal_marks': [91.0],
    'participation': [8.5]
}

# Convert to a pandas DataFrame
new_student_df = pd.DataFrame(new_student_data)
print("\nNew student data:")
print(new_student_df)


# --- 3. Preprocess the New Data (Must be IDENTICAL to training) ---
print("\nProcessing new data...")
# a. Create the composite 'engagement_score'
weights = {'attendance': 0.2, 'assignment_score': 0.3, 'internal_marks': 0.4, 'participation': 0.1}
new_student_df['engagement_score'] = (new_student_df['attendance'] * weights['attendance'] + 
                                      new_student_df['assignment_score'] * weights['assignment_score'] + 
                                      new_student_df['internal_marks'] * weights['internal_marks'] + 
                                      new_student_df['participation'] * 10 * weights['participation'])

# Ensure the columns are in the correct order before applying polynomial features
new_student_df_base = new_student_df[base_feature_cols]

# b. Apply the same polynomial feature transformation
new_student_poly = poly_features.transform(new_student_df_base)
poly_feature_names = poly_features.get_feature_names_out(base_feature_cols)
new_student_df_poly = pd.DataFrame(new_student_poly, columns=poly_feature_names)

print("Data processed successfully!")


# --- 4. Make the Prediction ---
# The pipeline handles scaling and predicting automatically.
prediction_encoded = model_pipeline.predict(new_student_df_poly)
prediction_proba = model_pipeline.predict_proba(new_student_df_poly)

# --- 5. Decode the Prediction ---
# Convert the numeric prediction (e.g., 1) back to its original label (e.g., 'Good').
prediction_label = label_encoder.inverse_transform(prediction_encoded)

print("\n" + "="*30)
print("       PREDICTION RESULT")
print("="*30)
print(f"🎓 Predicted Performance: '{prediction_label[0]}'")

# Show probabilities for each class
print("\nConfidence Scores:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"  - {class_name}: {prediction_proba[0][i]*100:.2f}%")