from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def get_mock_data():
    # Generate a synthetic binary classification dataset
    X, y = make_classification(n_samples=1000,  # Total number of samples
                            n_features=10,   # Number of features
                            n_informative=8, # Number of informative features
                            n_redundant=2,   # Number of redundant features
                            n_classes=2,     # Number of classes (binary classification)
                            random_state=42) # Seed for reproducibility

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test