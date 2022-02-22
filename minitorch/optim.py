class Optimizer:
    def __init__(self, parameters):
        self.parameters = parameters


class SGD(Optimizer):
    def __init__(self, parameters, lr=1.0):
        super().__init__(parameters)
        self.lr = lr

    def zero_grad(self):
        for p in self.parameters:
            if p.value.derivative is not None:
                p.value._derivative = None

    def step(self):
        for p in self.parameters:
            if p.value.derivative is not None:
                p.update(p.value - self.lr * p.value.derivative)


class MomentumSGD(Optimizer):
    def __init__(self, parameters, lr=1.0, alpha=0.95):
        super().__init__(parameters)
        self.lr = lr
        self.alpha = alpha

    def zero_grad(self):
        for p in self.parameters:
            if p.value.derivative is not None:
                p.value._derivative = None

    def step(self):
        theta = 0.
        for p in self.parameters:
            if p.value.derivative is not None:
                theta = theta * self.alpha - self.lr * p.value.derivative
                p.update(p.value + theta)
