import pandas as pd
import numpy as np
import os

# --- Configuration ---
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "students_sample.csv")

np.random.seed(42)  # reproducibility

# --- Parameters ---
n_samples = 500
classes = ["AtRisk", "Fail", "Pass", "High"]

# --- Feature Generation ---
attendance = np.random.randint(40, 101, size=n_samples)        # 40% - 100%
assignment_score = np.random.randint(20, 101, size=n_samples)  # 20 - 100
internal_marks = np.random.randint(10, 101, size=n_samples)    # 10 - 100
participation = np.random.randint(0, 11, size=n_samples)       # 0 - 10

# Engagement score (weighted formula, noisy to simulate variation)
engagement = (0.2 * attendance +
              0.3 * assignment_score +
              0.4 * internal_marks +
              0.1 * participation * 10 +
              np.random.normal(0, 5, n_samples))  # add noise

# --- Label Assignment Rules ---
labels = []
for eng in engagement:
    if eng < 40:
        labels.append("AtRisk")
    elif eng < 55:
        labels.append("Fail")
    elif eng < 75:
        labels.append("Pass")
    else:
        labels.append("High")

# --- Build DataFrame ---
df = pd.DataFrame({
    "attendance": attendance,
    "assignment_score": assignment_score,
    "internal_marks": internal_marks,
    "participation": participation,
    "previous_result": labels
})

# --- Save Dataset ---
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Synthetic dataset generated: {OUTPUT_PATH}")
print(df.head())
print("\nClass distribution:")
print(df["previous_result"].value_counts())
