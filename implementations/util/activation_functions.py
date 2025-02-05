from typing import List, Tuple, Callable, Union
from enum import Enum, auto
from .tensor import Tensor
import math

class ActivationFunctionType(Enum):
    RELU = auto()
    LEAKY_RELU = auto()
    SIGMOID = auto()
    TANH = auto()
    SOFTMAX = auto()

class ActivationFunction:
    def __init__(self, function_type: ActivationFunctionType, alpha: float = 0.01):
        self.alpha = alpha
        self.function_type = function_type
        self.activation_function, self.activation_derivative_function = self._get_activation_functions(function_type)

    def _get_activation_functions(self, function_type: ActivationFunctionType) -> Tuple[Callable, Callable]:
        match function_type:
            case ActivationFunctionType.RELU:
                return self._relu, self._relu_derivative
            case ActivationFunctionType.LEAKY_RELU:
                return self._leaky_relu, self._leaky_relu_derivative
            case ActivationFunctionType.SIGMOID:
                return self._sigmoid, self._sigmoid_derivative
            case ActivationFunctionType.TANH:
                return self._tanh, self._tanh_derivative
            case ActivationFunctionType.SOFTMAX:
                return self._softmax, self._softmax_derivative
            case _:
                raise ValueError(f"Unsupported activation function: {function_type}")

    def activate(self, x: float) -> float:
        return self.activation_function(x)

    def activate_derivative(self, x: float) -> float:
        return self.activation_derivative_function(x)

    def _relu(self, x: Union[Tensor, float]) -> Union[Tensor, float]:
        if isinstance(x, Tensor):
            return x.map(self._relu)
        return max(0, x)

    def _relu_derivative(self, x: Union[Tensor, float]) -> Union[Tensor, float]:
        if isinstance(x, Tensor):
            return x.map(self._relu_derivative)
        return 1 if x > 0 else 0

    def _leaky_relu(self, x: Union[Tensor, float]) -> Union[Tensor, float]:
        if isinstance(x, Tensor):
            return x.map(self._leaky_relu)
        return x if x > 0 else self.alpha * x

    def _leaky_relu_derivative(self, x: Union[Tensor, float]) -> Union[Tensor, float]:
        if isinstance(x, Tensor):
            return x.map(self._leaky_relu_derivative)
        return 1 if x > 0 else self.alpha

    def _sigmoid(self, x: Union[Tensor, float]) -> Union[Tensor, float]:
        if isinstance(x, Tensor):
            return x.map(self._sigmoid)
        return 1 / (1 + math.exp(-x))

    def _sigmoid_derivative(self, x: Union[Tensor, float]) -> Union[Tensor, float]:
        if isinstance(x, Tensor):
            return x.map(self._sigmoid_derivative)
        sig = self._sigmoid(x)
        return sig * (1 - sig)

    def _tanh(self, x: Union[Tensor, float]) -> Union[Tensor, float]:
        if isinstance(x, Tensor):
            return x.map(self._tanh)
        return math.tanh(x)

    def _tanh_derivative(self, x: Union[Tensor, float]) -> Union[Tensor, float]:
        if isinstance(x, Tensor):
            return x.map(self._tanh_derivative)
        return 1 - math.tanh(x) ** 2

    def _softmax(self, x: Union[Tensor, List[float]]) -> Union[Tensor, List[float]]:
        if isinstance(x, Tensor):
            return x.map(self._softmax)
        exps = [math.exp(i) for i in x]
        sum_of_exps = sum(exps)
        return [j / sum_of_exps for j in exps]

    def _softmax_derivative(self, x: Union[Tensor, List[float]]) -> Union[Tensor, List[float]]:
        if isinstance(x, Tensor):
            return x.map(self._softmax_derivative)
        s = self._softmax(x)
        return [s_i * (1 - s_i) for s_i in s]
