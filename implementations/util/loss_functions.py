from typing import List, Tuple, Callable
from enum import Enum, auto
import math

class LossFunctionType(Enum):
    MSE = auto()
    MAE = auto()
    BINARY_CROSS_ENTROPY = auto()
    CATEGORICAL_CROSS_ENTROPY = auto()

class LossFunction:
    def __init__(self, loss_type: LossFunctionType):
        self.loss_type = loss_type
        self.loss_function, self.loss_derivative_function = self._get_loss_functions(loss_type)

    def _get_loss_functions(self, loss_type: LossFunctionType) -> Tuple[Callable, Callable]:
        if loss_type == LossFunctionType.MSE:
            return self._mse_loss, self._mse_loss_derivative
        elif loss_type == LossFunctionType.MAE:
            return self._mae_loss, self._mae_loss_derivative
        elif loss_type == LossFunctionType.BINARY_CROSS_ENTROPY:
            return self._binary_cross_entropy, self._binary_cross_entropy_derivative
        elif loss_type == LossFunctionType.CATEGORICAL_CROSS_ENTROPY:
            return self._categorical_cross_entropy, self._categorical_cross_entropy_derivative
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def loss(self, y_true: List[float], y_pred: List[float]) -> float:
        return self.loss_function(y_true, y_pred)

    def loss_derivative(self, y_true: List[float], y_pred: List[float]) -> List[float]:
        return self.loss_derivative_function(y_true, y_pred)

    def _mse_loss(self, y_true: List[float], y_pred: List[float]) -> float:
        n = len(y_true)
        return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n

    def _mse_loss_derivative(self, y_true: List[float], y_pred: List[float]) -> List[float]:
        n = len(y_true)
        return [2 * (yp - yt) / n for yt, yp in zip(y_true, y_pred)]

    def _mae_loss(self, y_true: List[float], y_pred: List[float]) -> float:
        n = len(y_true)
        return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n

    def _mae_loss_derivative(self, y_true: List[float], y_pred: List[float]) -> List[float]:
        n = len(y_true)
        return [(1 if yp > yt else -1) / n for yt, yp in zip(y_true, y_pred)]

    def _binary_cross_entropy(self, y_true: List[float], y_pred: List[float], epsilon: float = 1e-10) -> float:
        n = len(y_true)
        return -sum(
            yt * math.log(max(yp, epsilon)) + (1 - yt) * math.log(max(1 - yp, epsilon))
            for yt, yp in zip(y_true, y_pred)
        ) / n

    def _binary_cross_entropy_derivative(self, y_true: List[float], y_pred: List[float], epsilon: float = 1e-10) -> List[float]:
        return [
            (yp - yt) / max(yp * (1 - yp), epsilon) 
            for yt, yp in zip(y_true, y_pred)
        ]

    def _categorical_cross_entropy(self, y_true: List[List[float]], y_pred: List[List[float]], epsilon: float = 1e-10) -> float:
        n = len(y_true)
        return -sum(
            sum(yt * math.log(max(yp, epsilon)) for yt, yp in zip(yt_row, yp_row))
            for yt_row, yp_row in zip(y_true, y_pred)
        ) / n

    def _categorical_cross_entropy_derivative(self, y_true: List[List[float]], y_pred: List[List[float]], epsilon: float = 1e-10) -> List[List[float]]:
        return [
            [-yt / max(yp, epsilon) for yt, yp in zip(yt_row, yp_row)]
            for yt_row, yp_row in zip(y_true, y_pred)
        ]
