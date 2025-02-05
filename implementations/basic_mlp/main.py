from implementations.util import ActivationFunctionType, LossFunctionType, ActivationFunction, LossFunction, Tensor
from typing import List, Union
import random

class BasicMLP():
    class Neuron():
        def __init__(self, input_dim: int, activation_function: ActivationFunction) -> None:
            self.weights = [random.uniform(-1, 1) for _ in range(input_dim)]
            self.bias = random.uniform(-1, 1)
            self.activation_function = activation_function

        def forward(self, inputs: List[Tensor]) -> Tensor:
            return self.activation_function.activate(sum([inputs[i] * self.weights[i] for i in range(len(inputs))], Tensor(0)) + self.bias)

    class Layer():
        def __init__(self, input_dim: int, output_dim: int, activation_function: ActivationFunction) -> None:
            self.neurons = [BasicMLP.Neuron(input_dim, activation_function) for _ in range(output_dim)]
        
        def forward(self, inputs: Union[List[Tensor], Tensor]) -> Union[List[Tensor], Tensor]:
            return [neuron.forward(inputs) for neuron in self.neurons]

    def __init__(self, input_dim: int, hidden_dim: int, hidden_layer_amount: int, output_dim: int, activation_function: ActivationFunctionType, output_activation_function: ActivationFunctionType, loss_function: LossFunctionType) -> None:
        self.activation_function = ActivationFunction(activation_function)
        self.output_activation_function = ActivationFunction(output_activation_function)
        self.loss_function = LossFunction(loss_function)

        self.layers = []
        self.layers.append(self.Layer(input_dim, hidden_dim, self.activation_function))
        for _ in range(hidden_layer_amount - 1):
            self.layers.append(self.Layer(hidden_dim, hidden_dim, self.activation_function))
        self.layers.append(self.Layer(hidden_dim, output_dim, self.output_activation_function))

    def forward(self, inputs: Tensor) -> Union[List[Tensor], Tensor]:
        inputs = [inputs]
        output = None
        for layer in self.layers:
            output = layer.forward(inputs)
        return output

    def train(self, inputs: List[Tensor], targets: List[Tensor], epochs: int, learning_rate: float) -> None:
        for _ in range(epochs):
            for i in range(len(inputs)):
                outputs = self.forward(inputs[i])
                loss = self.loss_function.loss(outputs, targets[i])
                print(f"Loss: {loss}")