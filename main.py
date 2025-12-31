import pandas as pd
import joblib
import yaml

x = pd.read_csv("parkinsons.csv")
df = pd.DataFrame(x)
df.head()
featuresq = ["spread1", "PPE"]
targetq = "status"
from sklearn.preprocessing import MinMaxScaler

# Select the two input features as a DataFrame
X = df[["spread1", "PPE"]]
from sklearn.model_selection import train_test_split

# Target variable
y = df["status"]

# Split the data into training and validation sets (X_train and X_val remain DataFrames)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=342
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# The features are already selected in X_train and X_val
# The target is already selected in y_train and y_val

model = Pipeline([
    ("scaler", MinMaxScaler()),
    ("clf", LogisticRegression(C=1, max_iter=200))
])

# Fit the model using only the training data
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

# חיזוי על סט הבדיקה
y_pred = model.predict(X_val)

# חישוב הדיוק
accuracy = accuracy_score(y_val, y_pred)

print("Test accuracy:", accuracy)
import joblib

joblib.dump(model, 'my_model.joblib')
