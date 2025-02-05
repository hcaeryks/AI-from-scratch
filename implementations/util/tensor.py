from typing import List, Union
import math

class Tensor:
    def __init__(self, data: Union[List[float], float], requires_grad: bool = False):
        if isinstance(data, Tensor):
            data = data.data
            # in case i mess up lol
        self.data = data
        self.requires_grad = requires_grad

    def shape(self) -> List[int]:
        return [len(self.data)]
    
    def __repr__(self) -> str:
        return f"Tensor({self.data})"
    
    def __iter__(self):
        return iter(self.data)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> float:
        return self.data[index]
    
    def __add__(self, other: 'Tensor') -> 'Tensor':
        if type(other) is not Tensor:
            other = Tensor(other)

        if isinstance(self.data, list) and isinstance(other.data, list):
            return Tensor([a + b for a, b in zip(self.data, other.data)])
        elif isinstance(self.data, list):
            return Tensor([a + other.data for a in self.data])
        elif isinstance(other.data, list):
            return Tensor([self.data + b for b in other.data])
        else:
            return Tensor(self.data + other.data)
        
    def __sub__(self, other: 'Tensor') -> 'Tensor':
        if type(other) is not Tensor:
            other = Tensor(other)

        if isinstance(self.data, list) and isinstance(other.data, list):
            return Tensor([a - b for a, b in zip(self.data, other.data)])
        elif isinstance(self.data, list):
            return Tensor([a - other.data for a in self.data])
        elif isinstance(other.data, list):
            return Tensor([self.data - b for b in other.data])
        else:
            return Tensor(self.data - other.data)
        
    def __mul__(self, other: 'Tensor') -> 'Tensor':
        if type(other) is not Tensor:
            other = Tensor(other)

        if isinstance(self.data, list) and isinstance(other.data, list):
            return Tensor([a * b for a, b in zip(self.data, other.data)])
        elif isinstance(self.data, list):
            return Tensor([a * other.data for a in self.data])
        elif isinstance(other.data, list):
            return Tensor([self.data * b for b in other.data])
        else:
            return Tensor(self.data * other.data)
    
    def __truediv__(self, other: 'Tensor') -> 'Tensor':
        if type(other) is not Tensor:
            other = Tensor(other)

        if isinstance(self.data, list) and isinstance(other.data, list):
            return Tensor([a / b for a, b in zip(self.data, other.data)])
        elif isinstance(self.data, list):
            return Tensor([a / other.data for a in self.data])
        elif isinstance(other.data, list):
            return Tensor([self.data / b for b in other.data])
        else:
            return Tensor(self.data / other.data)
        
    def __pow__(self, other: 'Tensor') -> 'Tensor':
        if type(other) is not Tensor:
            other = Tensor(other)

        if isinstance(self.data, list) and isinstance(other.data, list):
            return Tensor([a ** b for a, b in zip(self.data, other.data)])
        elif isinstance(self.data, list):
            return Tensor([a ** other.data for a in self.data])
        elif isinstance(other.data, list):
            return Tensor([self.data ** b for b in other.data])
        else:
            return Tensor(self.data ** other.data)
        
    def exp(self) -> 'Tensor':
        if isinstance(self.data, list):
            return Tensor([math.e ** a for a in self.data])
        else:
            return Tensor(math.e ** self.data)
        
    def log(self) -> 'Tensor':
        if isinstance(self.data, list):
            return Tensor([math.log(a) for a in self.data])
        else:
            return Tensor(math.log(self.data))
        
    def matmul(self, other: 'Tensor') -> 'Tensor':
        if isinstance(other, Tensor):
            if isinstance(self.data, list) and isinstance(other.data, list):
                if all(isinstance(i, list) for i in self.data) and all(isinstance(j, list) for j in other.data):
                    result = [[sum(a * b for a, b in zip(X_row, Y_col)) for Y_col in zip(*other.data)] for X_row in self.data]
                    return Tensor(result)
                else:
                    raise ValueError("Both tensors must be 2D lists for matrix multiplication")
            else:
                raise ValueError("Both tensors must be lists for matrix multiplication")
        else:
            raise TypeError("Operand must be of type Tensor")
        
    def dot(self, other: 'Tensor') -> 'Tensor':
        if isinstance(other, Tensor):
            if isinstance(self.data, list) and isinstance(other.data, list):
                return Tensor(sum(a * b for a, b in zip(self.data, other.data)))
            else:
                raise ValueError("Both tensors must be lists for dot product")
        else:
            raise TypeError("Operand must be of type Tensor")
        
    def map(self, func: callable) -> 'Tensor':
        if isinstance(self.data, list):
            return Tensor([func(a) for a in self.data])
        else:
            return Tensor(func(self.data))
