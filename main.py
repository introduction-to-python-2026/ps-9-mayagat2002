import pandas as pd
import joblib
import yaml

def load_model_and_predict():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_file = config["path"]
    features = config["features"]

    model = joblib.load(model_file)

    df = pd.read_csv("parkinson.csv")

    X = df[features]
    predictions = model.predict(X)

    return predictions

if __name__ == "__main__":
    load_model_and_predict()
