import joblib
# import tensorflow as tf
# import torch
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
import time



def load_model(model_path, model_format):
    if model_format == "sklearn":
        return joblib.load(model_path)
    # elif model_format == "tensorflow":
    #     return tf.keras.models.load_model(model_path)
    # elif model_format == "pytorch":
    #     return torch.load(model_path)
    else:
        raise ValueError(f"Unsupported model format: {model_format}")

def evaluate_model(model, model_format, X_test, y_test):
    start_time = time.time()

    if model_format == "sklearn":
        y_pred = model.predict(X_test)
    # elif model_format == "tensorflow":
    #     y_pred = model.predict(X_test).round()
    # elif model_format == "pytorch":
    #     model.eval()
    #     with torch.no_grad():
    #         X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    #         y_pred = model(X_test_tensor).round().numpy()
    else:
        raise ValueError(f"Unsupported model format: {model_format}")


    run_time = time.time() - start_time
    tested_entries = len(y_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    return accuracy, precision, recall, conf_matrix, fpr, tpr, roc_auc, run_time, tested_entries

def log_results_to_wandb(hotkey, accuracy, precision, recall, conf_matrix, fpr, tpr, roc_auc, run_time, tested_entries):
    wandb.init(project="model-validation", entity="urbaniak-bruno-safescanai") # TODO do zmiany

    wandb.log({
        # "hotkey": hotkey,
        "tested_entries": tested_entries,
        "model_test_run_time": run_time,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": conf_matrix.tolist(),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "roc_auc": roc_auc
    })

    wandb.finish()
    return