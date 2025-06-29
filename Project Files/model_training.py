# model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier# model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
df = pd.read_excel("data/HealthCareData.xlsx")

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Optional: Strip strings and lowercase for consistency
df = df.applymap(lambda x: str(x).strip().lower() if isinstance(x, str) else x)

# Select useful features and target
selected_columns = [
    "age", "gender", "duration of alcohol consumption(years)",
    "quantity of alcohol consumption (quarters/day)", "diabetes result",
    "blood pressure (mmhg)", "obesity", "hemoglobin  (g/dl)",
    "pcv  (%)", "sgot/ast      (u/l)", "sgpt/alt (u/l)", "albumin   (g/dl)",
    "predicted value(out come-patient suffering from liver  cirrosis or not)"
]
df = df[selected_columns].copy()

# Rename for simplicity
df.columns = [
    "age", "gender", "alcohol_years", "alcohol_quantity",
    "diabetes", "bp", "obesity", "hb", "pcv",
    "sgot", "sgpt", "albumin", "target"
]

# Remove rows with missing values
df.replace("", np.nan, inplace=True)
df.dropna(inplace=True)

# Convert target and categorical columns to numeric
mapping = {"yes": 1, "no": 0, "male": 1, "female": 0}
df["gender"] = df["gender"].map(mapping)
df["diabetes"] = df["diabetes"].map(mapping)
df["obesity"] = df["obesity"].map(mapping)
df["target"] = df["target"].map(mapping)

# Remove rows that became NaN due to unmapped strings
df.dropna(inplace=True)

# Extract systolic BP if format is "120/80"
def extract_systolic(bp_str):
    try:
        return float(str(bp_str).split("/")[0])
    except:
        return np.nan

df["bp"] = df["bp"].apply(extract_systolic)
df.dropna(inplace=True)

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE to balance
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_scaled, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Model trained and saved successfully")

from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
df = pd.read_excel("data/HealthCareData.xlsx")

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Optional: Strip strings and lowercase for consistency
df = df.applymap(lambda x: str(x).strip().lower() if isinstance(x, str) else x)

# Select useful features and target
selected_columns = [
    "age", "gender", "duration of alcohol consumption(years)",
    "quantity of alcohol consumption (quarters/day)", "diabetes result",
    "blood pressure (mmhg)", "obesity", "hemoglobin  (g/dl)",
    "pcv  (%)", "sgot/ast      (u/l)", "sgpt/alt (u/l)", "albumin   (g/dl)",
    "predicted value(out come-patient suffering from liver  cirrosis or not)"
]
df = df[selected_columns].copy()

# Rename for simplicity
df.columns = [
    "age", "gender", "alcohol_years", "alcohol_quantity",
    "diabetes", "bp", "obesity", "hb", "pcv",
    "sgot", "sgpt", "albumin", "target"
]

# Remove rows with missing values
df.replace("", np.nan, inplace=True)
df.dropna(inplace=True)

# Convert target and categorical columns to numeric
mapping = {"yes": 1, "no": 0, "male": 1, "female": 0}
df["gender"] = df["gender"].map(mapping)
df["diabetes"] = df["diabetes"].map(mapping)
df["obesity"] = df["obesity"].map(mapping)
df["target"] = df["target"].map(mapping)

# Remove rows that became NaN due to unmapped strings
df.dropna(inplace=True)

# Extract systolic BP if format is "120/80"
def extract_systolic(bp_str):
    try:
        return float(str(bp_str).split("/")[0])
    except:
        return np.nan

df["bp"] = df["bp"].apply(extract_systolic)
df.dropna(inplace=True)

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE to balance
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_scaled, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Model trained and saved successfully")
