from preprocess import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load and preprocess data
data = preprocess_data("data/churn.csv")

# Split features and target
X = data.drop("Churn", axis=1)
y = data["Churn"]

joblib.dump(list(X.columns), "model/columns.pkl")
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/model.pkl")

print("Model trained and saved successfully")