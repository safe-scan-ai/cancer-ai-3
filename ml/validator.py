import argparse
import wandb
import os

from dotenv import load_dotenv

from utils import load_model, evaluate_model, log_results_to_wandb
from mockdata import get_mock_data

# Parse command line arguments to choose the model
parser = argparse.ArgumentParser(description="Choose which model to test.")
parser.add_argument("--model", type=str, choices=["sklearn", "tensorflow", "pytorch"], required=True, help="The model to test (sklearn, tensorflow, pytorch).")
args = parser.parse_args()

# Load the environment variables
load_dotenv()

# Access the API key from the environment variable
wandb_api_key = os.getenv('WANDB_API_KEY')


# Log in to wandb using the API key
wandb.login(key=wandb_api_key)

# Load data
X_train, X_test, y_train, y_test = get_mock_data()

# To be replaced with the official SafescanAI account and added in cofig.py
PROJECT_NAME = "model-validation"
ENTITY_NAME = "urbaniak-bruno-safescanai"

# Load and evaluate models based on the chosen model
if args.model == "sklearn":
    sklearn_model = load_model("models/sklearn_model.pkl", "sklearn")
    sklearn_results = evaluate_model(sklearn_model, "sklearn", X_test, y_test)
    log_results_to_wandb(PROJECT_NAME, ENTITY_NAME, "hotkey_sklearn", sklearn_results)

elif args.model == "tensorflow":
    tensorflow_model = load_model("models/tensorflow_model.keras", "tensorflow")
    tensorflow_results = evaluate_model(tensorflow_model, "tensorflow", X_test, y_test)
    log_results_to_wandb(PROJECT_NAME, ENTITY_NAME, "hotkey_tensorflow", tensorflow_results)

elif args.model == "pytorch":
    pytorch_model = load_model("models/pytorch_model.pth", "pytorch")
    pytorch_results = evaluate_model(pytorch_model, "pytorch", X_test, y_test)
    log_results_to_wandb(PROJECT_NAME, ENTITY_NAME, "hotkey_pytorch", pytorch_results)

# Finish the WandB run
wandb.finish()
