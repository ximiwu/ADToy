import numpy as np

class Add:
    def backward(self, grad, inputs):
        a, b = inputs
        a.grad += grad
        b.grad += grad

class Mul:
    def backward(self, grad, inputs):
        a, b = inputs
        if a.data.shape == ():
            grad_a = (grad * b.data).sum()
        else:
            grad_a = grad * b.data
        if b.data.shape == ():
            grad_b = grad * a.data.sum()
        else:
            grad_b = grad * a.data
        a.grad += grad_a
        b.grad += grad_b
        
        

class Sub:
    def backward(self, grad, inputs):
        a, b = inputs
        a.grad += grad
        b.grad -= grad

class MatMul:
    def backward(self, grad, inputs):
        A, B = inputs
        grad_a = grad @ np.swapaxes(B.data, -1, -2)
        grad_b = np.swapaxes(A.data, -1, -2) @ grad
        while grad_a.shape != A.data.shape:
            grad_a = np.sum(grad_a, axis=(0))
        while grad_b.shape != B.data.shape:
            grad_b = np.sum(grad_b, axis=(0))
        A.grad += grad_a
        B.grad += grad_b
    

class tensor:
    def __init__(self, data, requires_grad=False, is_leaf = True):
        self.data = np.array(data, dtype=float)
        self.requires_grad = requires_grad
        self.grad = np.zeros(self.data.shape, dtype=float)
        self.op = None
        self.is_leaf = is_leaf
        self.inputs = []
        self.backprop_done = False

                
    
    def backward(self, grad=None):
        if self.backprop_done:
            return
        self.backprop_done = True
        if not self.requires_grad:
            return
        if grad is not None:
            self.grad += grad
        if self.op is None:
            return
        self.op.backward(self.grad, self.inputs)
        for node in self.inputs:
            node.backward()
    
    def get_ones(self):
        return np.ones(self.data.shape, dtype=float)    
        
    def __mul__(self, other):
        requires_grad = self.requires_grad or other.requires_grad
        out = tensor(self.data * other.data, requires_grad, is_leaf=False)
        out.op = Mul()
        out.inputs = [self, other]
        return out
    
    def __matmul__(self, other):
        requires_grad = self.requires_grad or other.requires_grad
        out = tensor(self.data @ other.data, requires_grad, is_leaf=False)
        out.op = MatMul()
        out.inputs = [self, other]
        return out
    
    def __add__(self, other):
        requires_grad = self.requires_grad or other.requires_grad
        out = tensor(self.data + other.data, requires_grad, is_leaf=False)
        out.op = Add()
        out.inputs = [self, other]
        return out
    def __sub__(self, other):
        requires_grad = self.requires_grad or other.requires_grad
        out = tensor(self.data - other.data, requires_grad, is_leaf=False)
        out.op = Sub()
        out.inputs = [self, other]
        return out
