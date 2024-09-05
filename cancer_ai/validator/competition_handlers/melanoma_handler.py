from .base_handler import BaseCompetitionHandler
from .base_handler import ModelEvaluationResult

from typing import List
from PIL import Image
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc,
)


class MelanomaCompetitionHandler(BaseCompetitionHandler):
    """Handler for melanoma competition"""

    def __init__(self, X_test, y_test) -> None:
        super().__init__(X_test, y_test)

    def prepare_y_pred(self, y_pred: np.ndarray) -> np.ndarray:
        return [1 if y == "True" else 0 for y in self.y_test]

    def get_model_result(
        self, y_test: List[float], y_pred: np.ndarray, run_time_s: float
    ) -> ModelEvaluationResult:
        y_pred_binary = [1 if y > 0.5 else 0 for y in y_pred]
        tested_entries = len(y_test)
        accuracy = accuracy_score(y_test, y_pred_binary)
        precision = precision_score(y_test, y_pred_binary)
        recall = recall_score(y_test, y_pred_binary)
        conf_matrix = confusion_matrix(y_test, y_pred_binary)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        return ModelEvaluationResult(
            tested_entries=tested_entries,
            run_time_s=run_time_s,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            confusion_matrix=conf_matrix,
            fpr=fpr,
            tpr=tpr,
            roc_auc=roc_auc,
        )
