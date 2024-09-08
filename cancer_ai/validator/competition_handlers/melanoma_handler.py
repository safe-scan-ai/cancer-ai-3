from .base_handler import BaseCompetitionHandler
from .base_handler import ModelEvaluationResult

from typing import List
from PIL import Image
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    fbeta_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc,
)


class MelanomaCompetitionHandler(BaseCompetitionHandler):
    """Handler for melanoma competition"""

    def __init__(self, X_test, y_test, weight_fbeta=0.6, weight_accuracy=0.3, weight_auc=0.1) -> None:
        super().__init__(X_test, y_test)
        self.weight_fbeta = weight_fbeta
        self.weight_accuracy = weight_accuracy
        self.weight_auc = weight_auc


    def prepare_y_pred(self, y_pred: np.ndarray) -> np.ndarray:
        return [1 if y == "True" else 0 for y in self.y_test]

    def calculate_score(self, fbeta: float, accuracy: float, roc_auc: float) -> float:
        return fbeta * self.weight_fbeta + accuracy * self.weight_accuracy + roc_auc * self.weight_auc
    
    def get_model_result(
        self, y_test: List[float], y_pred: np.ndarray, run_time_s: float
    ) -> ModelEvaluationResult:
        y_pred_binary = [1 if y > 0.5 else 0 for y in y_pred]
        tested_entries = len(y_test)
        accuracy = accuracy_score(y_test, y_pred_binary)
        precision = precision_score(y_test, y_pred_binary, zero_division=0)
        fbeta = fbeta_score(y_test, y_pred_binary, beta=2, zero_division=0)
        recall = recall_score(y_test, y_pred_binary, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred_binary)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        score = self.calculate_score(fbeta, accuracy, roc_auc)

        return ModelEvaluationResult(
            tested_entries=tested_entries,
            run_time_s=run_time_s,
            accuracy=accuracy,
            precision=precision,
            fbeta=fbeta,
            recall=recall,
            confusion_matrix=conf_matrix,
            fpr=fpr,
            tpr=tpr,
            roc_auc=roc_auc,
            score=score
        )
