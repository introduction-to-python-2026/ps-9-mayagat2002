# Download the data from your GitHub repository
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv
!wget https://raw.githubusercontent.com/yotam-biu/python_utils/main/lab_setup_do_not_edit.py -O /content/lab_setup_do_not_edit.py
import lab_setup_do_not_edit
import pandas as pd

# Load the dataset
df = pd.read_csv("/content/parkinsons.csv")

# Display first rows to verify
df.head()
features = ["spread1", "PPE"]
X = df[features]
y = df["status"]

model.fit(X, y)
joblib.dump(model, "parkinson_model.joblib")
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
