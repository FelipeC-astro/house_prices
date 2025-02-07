from src.preprocessing import load_data, preprocess_data
from src.modeling import train_and_evaluate
from src.prediction import make_predictions

# File paths
train_path = "data/train.csv"
test_path = "data/test.csv"
output_path = "data/submission.csv"

# Define features and target variable
features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
target = "SalePrice"

# Load and preprocess data
train_df, test_df = load_data(train_path, test_path)
X_train, X_valid, y_train, y_valid, test_df = preprocess_data(train_df, test_df, features, target)

# Train and evaluate models
best_model = train_and_evaluate(X_train, X_valid, y_train, y_valid)

# Make predictions and save results
make_predictions(best_model, test_df, features, output_path)
