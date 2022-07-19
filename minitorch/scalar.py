from .autodiff import FunctionBase, Variable, History, Context
from . import operators
from typing import Callable, Union

import numpy as np


# ## Task 1.1
# Central Difference calculation


def central_difference(
    f: Callable[..., float], *vals: float, arg: int = 0, epsilon: float = 1e-6
) -> float:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals (list of floats): n-float values :math:`x_0 \ldots x_{n-1}`
        arg (int): the number :math:`i` of the arg to compute the derivative
        epsilon (float): a small constant

    Returns:
        float : An approximation of :math:`f'_i(x_0, \ldots, x_{n-1})`
    """
    lvals = list(vals)
    rvals = list(vals)
    lvals[arg] = lvals[arg] + epsilon
    rvals[arg] = rvals[arg] - epsilon

    return (f(*lvals) - f(*rvals)) / (2.0 * epsilon)


# ## Task 1.2 and 1.4
# Scalar Forward and Backward


class Scalar:
    ...


class Scalar(Variable):
    """
    A reimplementation of scalar values for autodifferentiation
    tracking.  Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    :class:`ScalarFunction`.

    Attributes:
        data (float): The wrapped scalar value.
    """

    def __init__(
        self, value: float, back: History = History(), name: Union[str, None] = None
    ):
        super().__init__(back, name=name)
        self.data: float = float(value)

    def __repr__(self) -> str:
        return f"Scalar({self.data})"

    def __mul__(self, b: Union[int, float, Scalar]) -> Scalar:
        return Mul.apply(self, b)

    def __truediv__(self, b: Union[int, float, Scalar]) -> Scalar:
        if not isinstance(b, Variable):
            return Mul.apply(self, 1 / b)
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: Union[int, float, Scalar]) -> Scalar:
        return Mul.apply(b, Inv.apply(self))

    def __add__(self, b: Union[int, float, Scalar]) -> Scalar:
        return Add.apply(self, b)

    def __sub__(self, b: Union[int, float, Scalar]) -> Scalar:
        if not isinstance(b, Variable):
            return Add.apply(self, -b)
        return Add.apply(self, Neg.apply(b))

    def __bool__(self) -> bool:
        return bool(self.data)

    def __lt__(self, b: Union[int, float, Scalar]) -> bool:
        return LT.apply(self, b)

    def __gt__(self, b: Union[int, float, Scalar]) -> bool:
        return LT.apply(b, self)

    def __eq__(self, b: Union[int, float, Scalar]) -> bool:
        return EQ.apply(self, b)

    def __neg__(self) -> Scalar:
        return Neg.apply(self)

    def log(self) -> Scalar:
        return Log.apply(self)

    def exp(self) -> Scalar:
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        return ReLU.apply(self)

    def get_data(self) -> float:
        "Returns the raw float value"
        return self.data


class ScalarFunction(FunctionBase):
    """
    A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @staticmethod
    def forward(ctx: Context, *inputs: float):
        r"""
        Forward call, compute :math:`f(x_0 \ldots x_{n-1})`.

        Args:
            ctx (:class:`Context`): A container object to save
                                    any information that may be needed
                                    for the call to backward.
            *inputs (list of floats): n-float values :math:`x_0 \ldots x_{n-1}`.

        Should return float the computation of the function :math:`f`.
        """
        pass  # pragma: no cover

    @staticmethod
    def backward(ctx: Context, d_out: float):
        r"""
        Backward call, computes :math:`f'_{x_i}(x_0 \ldots x_{n-1}) \times d_{out}`.

        Args:
            ctx (Context): A container object holding any information saved during in the corresponding `forward` call.
            d_out (float): :math:`d_out` term in the chain rule.

        Should return the computation of the derivative function
        :math:`f'_{x_i}` for each input :math:`x_i` times `d_out`.

        """
        pass  # pragma: no cover

    # Checks.
    variable = Scalar
    data_type = float

    @staticmethod
    def data(a):
        return a


class Add(ScalarFunction):
    "Addition function :math:`f(x, y) = x + y`"

    @staticmethod
    def forward(ctx: Context, a: float, b: float):
        return operators.add(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float):
        return d_output, d_output


class Log(ScalarFunction):
    "Log function :math:`f(x) = log(x)`"

    @staticmethod
    def forward(ctx: Context, a: float):
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float):
        a = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    "Multiplication function"

    @staticmethod
    def forward(ctx: Context, a: float, b: float):
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float):
        return ctx.saved_values[1] * d_output, ctx.saved_values[0] * d_output


class Inv(ScalarFunction):
    "Inverse function"

    @staticmethod
    def forward(ctx: Context, a: float):
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float):
        a = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    "Negation function"

    @staticmethod
    def forward(ctx: Context, a: float):
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float):
        return -1.0 * d_output


class Sigmoid(ScalarFunction):
    "Sigmoid function"

    @staticmethod
    def forward(ctx: Context, a: float):
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float):
        a = ctx.saved_values
        return operators.sigmoid(a) * (1.0 - operators.sigmoid(a)) * d_output


class ReLU(ScalarFunction):
    "ReLU function"

    @staticmethod
    def forward(ctx: Context, a: float):
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float):
        a = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    "Exp function"

    @staticmethod
    def forward(ctx: Context, a: float):
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float):
        a = ctx.saved_values
        return operators.exp(a) * d_output


class LT(ScalarFunction):
    "Less-than function :math:`f(x) =` 1.0 if x is less than y else 0.0"

    @staticmethod
    def forward(ctx: Context, a: float, b: float):
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float):
        return 0.0, 0.0


class EQ(ScalarFunction):
    "Equal function :math:`f(x) =` 1.0 if x is equal to y else 0.0"

    @staticmethod
    def forward(ctx: Context, a: float, b: float):
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float):
        return 0.0, 0.0


def derivative_check(f: Callable[..., Scalar], *scalars: Scalar):
    """
    Checks that autodiff works on a python function.
    Asserts False if derivative is incorrect.

    Parameters:
        f (function) : function from n-scalars to 1-scalar.
        *scalars (list of :class:`Scalar`) : n input scalar values.
    """
    for x in scalars:
        x.requires_grad_(True)
    out = f(*scalars)
    out.backward()

    vals = [v for v in scalars]
    err_msg = """
        Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
        but was expecting derivative f'=%f from central difference.
    """

    for i, x in enumerate(scalars):
        check = central_difference(f, *vals, arg=i)
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
