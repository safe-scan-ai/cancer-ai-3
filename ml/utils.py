import joblib
# import tensorflow as tf
# import torch
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
import time
from dataclasses import dataclass

@dataclass
class ModelEvaluationResult:
    accuracy: float
    precision: float
    recall: float
    confusion_matrix: any
    fpr: any
    tpr: any
    roc_auc: float
    run_time: float
    tested_entries: int

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

    return ModelEvaluationResult(
        tested_entries=tested_entries,
        run_time=run_time,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        confusion_matrix=conf_matrix,
        fpr=fpr,
        tpr=tpr,
        roc_auc=roc_auc,
    )

def log_results_to_wandb(project, entity, hotkey, evaluation_result: ModelEvaluationResult):
    wandb.init(project=project, entity=entity)  # TODO: Update this line as needed

    wandb.log({
        "hotkey": hotkey,
        "tested_entries": evaluation_result.tested_entries,
        "model_test_run_time": evaluation_result.run_time,
        "accuracy": evaluation_result.accuracy,
        "precision": evaluation_result.precision,
        "recall": evaluation_result.recall,
        "confusion_matrix": evaluation_result.confusion_matrix.tolist(),
        "roc_curve": {
            "fpr": evaluation_result.fpr.tolist(),
            "tpr": evaluation_result.tpr.tolist()
        },
        "roc_auc": evaluation_result.roc_auc
    })

    wandb.finish()
    return