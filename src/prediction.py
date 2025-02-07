import pandas as pd

def make_predictions(model, test_df, features, output_path):
    """
    Uses the best model to make predictions on the test set and saves results.
    """
    predictions = model.predict(test_df[features])
    output = pd.DataFrame({"Id": test_df.index, "SalePrice": predictions})
    output.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")