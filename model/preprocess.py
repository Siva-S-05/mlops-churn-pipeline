import pandas as pd

def preprocess_data(file_path):
    # Load data
    data = pd.read_csv(file_path)

    # Drop unnecessary column
    data = data.drop("customerID", axis=1)

    # Convert TotalCharges to numeric
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    # Drop missing values
    data = data.dropna()

    # Convert target column
    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

    # One-hot encoding
    data = pd.get_dummies(data, drop_first=True)

    return data