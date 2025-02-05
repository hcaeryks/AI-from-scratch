from implementations.basic_mlp import BasicMLP
from implementations.util import ActivationFunctionType, LossFunctionType, Tensor

if __name__ == "__main__":
    basic_mlp = BasicMLP(1, 2, 2, 1, ActivationFunctionType.RELU, ActivationFunctionType.SIGMOID, LossFunctionType.MSE)
    print(basic_mlp.forward(Tensor([2.0, 1.0])))