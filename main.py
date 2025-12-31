import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

df = pd.read_csv("parkinson.csv")

features = ["MDVP:Fo(Hz)", "MDVP:Jitter(%)"]
X = df[features]
y = df["status"]

model = Pipeline([
    ("scaler", MinMaxScaler()),
    ("svc", SVC(C=10))
])

model.fit(X, y)

joblib.dump(model, "parkinson_model.joblib")
