import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import joblib

# Load data
df = pd.read_csv("parkinson.csv")

features = ["spread1", "PPE"]
X = df[features]
y = df["status"]

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline !!!
model = Pipeline([
    ("scaler", MinMaxScaler()),
    ("svc", SVC(kernel="rbf", C=10, gamma="scale"))
])

# Train
model.fit(X_train, y_train)

# Save
joblib.dump(model, "parkinson_model.joblib")
