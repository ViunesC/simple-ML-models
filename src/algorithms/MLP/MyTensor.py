import numpy as np

class MyTensor:
    def __init__(self, data: np.number, _op="", _children=(), label=""):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"MyTensor<{self.data}>"

    # unit operations
    def __add__(self, other):
        # convert from number to MyTensor if other is not a MyTensor object
        other = other if isinstance(other, MyTensor) else MyTensor(other)

        output = MyTensor(self.data + other.data, "+", (self, other))

        def _backward():
            # f(a + b)' w.r.t a = 1
            self.grad += 1.0 * output.grad
            other.grad += 1.0 * output.grad
        
        output._backward = _backward
        return output

    def __mul__(self, other):
        other = other if isinstance(other, MyTensor) else MyTensor(other)

        output = MyTensor(self.data * other.data, "*", (self, other))

        def _backward():
            # f(a*b)' w.r.t a = b
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad
        
        output._backward = _backward
        return output

    # slightly complicated operations
    def __pow__(self, other):
        if not isinstance(other, (int, float, np.number)):
            raise ValueError("Other must be a numpy number-like object.")
        
        output = MyTensor(np.pow(self.data, other), "**", (self, ))

        def _backward():
            # f(x^a)' w.r.t x = a*x^(a-1)
            self.grad += other * np.pow(self.data, other-1) * output.grad
        
        output._backward = _backward
        return output

    def exp(self):
        output = MyTensor(np.exp(self.data), "exp", (self, ))

        def _backward():
            # f(e^x)' w.r.t x = e^x
            self.grad += output.data * output.grad
        
        output._backward = _backward
        return output

    def tanh(self):
        """Tanh activation function"""
        e_x, n_e_x = np.exp(self.data), np.exp(-self.data)
        numer, denom = e_x - n_e_x, e_x + n_e_x

        output = MyTensor(numer / denom, "tanh", (self, ))

        def _backward():
            # f(tanh(x))' w.r.t x = 1 - tanh^2(x)
            self.grad += (1 - np.pow(output.data, 2)) * output.grad
        
        output._backward = _backward
        return output

    # composite unit operations
    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1)
    
    # Neural network related
    def backward(self):
        self.grad = 1.0
        visited = set()
        nodes = []

        def topo(root: MyTensor):
            """Get topological order of tensors"""
            if root not in visited:
                visited.add(root)
                for child in root._prev:
                    topo(child)
                nodes.append(root)
        
        topo(self)
        for node in reversed(nodes):
            node._backward()