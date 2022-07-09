"""
Collection of the core mathematical operators used throughout the code base.
"""

import math

from typing import Callable, Sequence

EPS = 1e-6


def mul(x: float, y: float) -> float:
    ":math:`f(x, y) = x * y`"
    return x * y


def id(x: float) -> float:
    ":math:`f(x) = x`"
    return x


def add(x: float, y: float) -> float:
    ":math:`f(x, y) = x + y`"
    return x + y


def neg(x: float) -> float:
    ":math:`f(x) = -x`"
    return -1. * x


def lt(x: float, y: float) -> bool:
    ":math:`f(x) =` 1.0 if x is less than y else 0.0"
    return x < y


def eq(x: float, y: float) -> bool:
    ":math:`f(x) =` 1.0 if x is equal to y else 0.0"
    return x == y


def max(x: float, y: float) -> float:
    ":math:`f(x) =` x if x is greater than y else y"
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    ":math:`f(x) = |x - y| < 1e-2`"
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    r"""
    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}`

    (See `<https://en.wikipedia.org/wiki/Sigmoid_function>`_ .)

    Calculate as

    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}` if x >=0 else :math:`\frac{e^x}{(1.0 + e^{x})}`

    for stability.

    Args:
        x (float): input

    Returns:
        float : sigmoid value
    """
    return 1. / (1. + math.exp(-x) + EPS)


def relu(x: float) -> float:
    """
    :math:`f(x) =` x if x is greater than 0, else 0

    (See `<https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`_ .)

    Args:
        x (float): input

    Returns:
        float : relu value
    """
    return x if x > 0. else 0.


def log(x: float) -> float:
    ":math:`f(x) = log(x)`"
    return math.log(x + EPS)


def exp(x: float) -> float:
    ":math:`f(x) = e^{x}`"
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    r"If :math:`f = log` as above, compute :math:`d \times f'(x)`"
    return d / (x + EPS)


def inv(x: float) -> float:
    ":math:`f(x) = 1/x`"
    return 1. / x


def inv_back(x: float, d: float) -> float:
    r"If :math:`f(x) = 1/x` compute :math:`d \times f'(x)`"
    return - d / x ** 2


def relu_back(x: float, d: float) -> float:
    r"If :math:`f = relu` compute :math:`d \times f'(x)`"
    return 1. * d * (x >= 0.)


# Small library of elementary higher-order functions for practice.


def map(fn: Callable[[float], float]) -> Callable[[Sequence[float]], Sequence[float]]:
    """
    Higher-order map.

    .. image:: figs/Ops/maplist.png


    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (one-arg function): Function from one value to one value.

    Returns:
        function : A function that takes a list, applies `fn` to each element, and returns a
        new list
    """
    def mapper(ls: Sequence[float]) -> Sequence[float]:
        return [fn(item) for item in ls]

    return mapper


def negList(ls: Sequence[float]) -> Sequence[float]:
    "Use :func:`map` and :func:`neg` to negate each element in `ls`"
    return map(neg)(ls)


def zipWith(fn: Callable[[float, float], float]) -> Callable[[Sequence[float], Sequence[float]], Sequence[float]]:
    """
    Higher-order zipwith (or map2).

    .. image:: figs/Ops/ziplist.png

    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (two-arg function): combine two values

    Returns:
        function : takes two equally sized lists `ls1` and `ls2`, produce a new list by
        applying fn(x, y) on each pair of elements.

    """
    def mapper(ls1: Sequence[float], ls2: Sequence[float]) -> Sequence[float]:
        return [fn(ls1[i], ls2[i]) for i in range(min(len(ls1), len(ls2)))]

    return mapper


def addLists(ls1: Sequence[float], ls2: Sequence[float]) -> Sequence[float]:
    "Add the elements of `ls1` and `ls2` using :func:`zipWith` and :func:`add`"
    return zipWith(add)(ls1, ls2)


def reduce(fn: Callable[[float, float], float], start: float) -> Callable[[Sequence[float]], float]:
    r"""
    Higher-order reduce.

    .. image:: figs/Ops/reducelist.png


    Args:
        fn (two-arg function): combine two values
        start (float): start value :math:`x_0`

    Returns:
        function : function that takes a list `ls` of elements
        :math:`x_1 \ldots x_n` and computes the reduction :math:`fn(x_3, fn(x_2,
        fn(x_1, x_0)))`
    """
    def mapper(ls: Sequence[float]) -> float:
        res = start

        for item in ls:
            res = fn(item, res)

        return res

    return mapper


def sum(ls: Sequence[float]) -> float:
    "Sum up a list using :func:`reduce` and :func:`add`."
    return reduce(add, 0)(ls)


def prod(ls: Sequence[float]) -> float:
    "Product of a list using :func:`reduce` and :func:`mul`."
    return reduce(mul, 1)(ls)
