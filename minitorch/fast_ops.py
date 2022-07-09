from re import A
import numpy as np
from .tensor_data import (
    to_index,
    index_to_position,
    broadcast_index,
    shape_broadcast,
    MAX_DIMS,
)
from numba import njit, prange
from typing import Callable, Any, Optional
from .tensor import Tensor


# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


def tensor_map(fn: Callable[[float], float]) -> Any:
    """
    NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

        * Main loop in parallel
        * All indices use numpy buffers
        * When `out` and `in` are stride-aligned, avoid indexing

    Args:
        fn: function mappings floats-to-floats to apply.
        out (array): storage for out tensor.
        out_shape (array): shape for out tensor.
        out_strides (array): strides for out tensor.
        in_storage (array): storage for in tensor.
        in_shape (array): shape for in tensor.
        in_strides (array): strides for in tensor.

    Returns:
        None : Fills in `out`
    """

    def _map(out: np.ndarray[Any, np.float64],
             out_shape: np.ndarray[Any, np.int64],
             out_strides: np.ndarray[Any, np.int64],
             in_storage: np.ndarray[Any, np.float64],
             in_shape: np.ndarray[Any, np.int64],
             in_strides: np.ndarray[Any, np.int64]):
        for index in prange(len(out)):
            out_index, in_index = np.zeros(len(out_shape)), np.zeros(len(in_shape))
            to_index(index, out_shape, out_index)

            broadcast_index(out_index, out_shape, in_shape, in_index)

            out[index] = fn(in_storage[index_to_position(in_index, in_strides)])

    return njit(parallel=True)(_map)


def map(fn: Callable[[float], float]) -> Any:
    """
    Higher-order tensor map function ::

      fn_map = map(fn)
      fn_map(a, out)
      out

    Args:
        fn: function from float-to-float to apply.
        a (:class:`Tensor`): tensor to map over
        out (:class:`Tensor`): optional, tensor data to fill in,
               should broadcast with `a`

    Returns:
        :class:`Tensor` : new tensor
    """

    # This line JIT compiles your tensor_map
    f = tensor_map(njit()(fn))

    def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
        if out is None:
            out = a.zeros(a.shape)
        f(*out.tuple(), *a.tuple())
        return out

    return ret


def tensor_zip(fn: Callable[[float, float], float]) -> Any:
    """
    NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.


    Optimizations:

        * Main loop in parallel
        * All indices use numpy buffers
        * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
        fn: function maps two floats to float to apply.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        b_storage (array): storage for `b` tensor.
        b_shape (array): shape for `b` tensor.
        b_strides (array): strides for `b` tensor.

    Returns:
        None : Fills in `out`
    """

    def _zip(
            out: np.ndarray[Any, np.float64],
            out_shape: np.ndarray[Any, np.int64],
            out_strides: np.ndarray[Any, np.int64],
            a_storage: np.ndarray[Any, np.float64],
            a_shape: np.ndarray[Any, np.int64],
            a_strides: np.ndarray[Any, np.int64],
            b_storage: np.ndarray[Any, np.float64],
            b_shape: np.ndarray[Any, np.int64],
            b_strides: np.ndarray[Any, np.int64]) -> Any:
        for i in prange(len(out)):
            a_index, b_index, out_index = np.zeros(len(a_shape)), np.zeros(len(b_shape)), np.zeros(len(out_shape))

            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            out[i] = fn(a_storage[index_to_position(a_index, a_strides)],
                        b_storage[index_to_position(b_index, b_strides)])

    return njit(parallel=True)(_zip)


def zip(fn: Callable[[float, float], float]) -> Any:
    """
    Higher-order tensor zip function.

      fn_zip = zip(fn)
      c = fn_zip(a, b)

    Args:
        fn: function from two floats-to-float to apply
        a (:class:`Tensor`): tensor to zip over
        b (:class:`Tensor`): tensor to zip over

    Returns:
        :class:`Tensor` : new tensor data
    """
    f = tensor_zip(njit()(fn))

    def ret(a: Tensor, b: Tensor) -> Tensor:
        c_shape = shape_broadcast(a.shape, b.shape)
        out = a.zeros(c_shape)
        f(*out.tuple(), *a.tuple(), *b.tuple())
        return out

    return ret


def tensor_reduce(fn: Callable[[float, float], float]) -> Any:
    """
    NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

        * Main loop in parallel
        * All indices use numpy buffers
        * Inner-loop should not call any functions or write non-local variables

    Args:
        fn: reduction function mapping two floats to float.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        reduce_dim (int): dimension to reduce out

    Returns:
        None : Fills in `out`

    """

    def _reduce(out: np.ndarray[Any, np.float64],
                out_shape: np.ndarray[Any, np.int64],
                out_strides: np.ndarray[Any, np.int64],
                a_storage: np.ndarray[Any, np.float64],
                a_shape: np.ndarray[Any, np.int64],
                a_strides: np.ndarray[Any, np.int64],
                reduce_dim: int) -> None:
        for i in prange(len(out)):
            out_index = np.zeros(len(out_shape))

            to_index(i, out_shape, out_index)
            a_pos = index_to_position(out_index, a_strides)

            args = np.array([a_storage[a_pos + j * a_strides[reduce_dim]] for j in range(a_shape[reduce_dim])])

            for arg in args:
                out[i] = fn(out[i], arg)

    return njit(parallel=True)(_reduce)


def reduce(fn: Callable[[float, float], float], start: float = 0.0) -> Any:
    """
    Higher-order tensor reduce function. ::

      fn_reduce = reduce(fn)
      out = fn_reduce(a, dim)


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`Tensor`): tensor to reduce over
        dim (int): int of dim to reduce

    Returns:
        :class:`Tensor` : new tensor
    """

    f = tensor_reduce(njit()(fn))

    def ret(a: Tensor, dim: int) -> Tensor:
        out_shape = list(a.shape)
        out_shape[dim] = 1

        # Other values when not sum.
        out = a.zeros(tuple(out_shape))
        out._tensor._storage[:] = start

        f(*out.tuple(), *a.tuple(), dim)
        return out

    return ret


@njit(parallel=True, fastmath=True)
def tensor_matrix_multiply(
        out: np.ndarray[Any, np.float64],
        out_shape: np.ndarray[Any, np.int64],
        out_strides: np.ndarray[Any, np.int64],
        a_storage: np.ndarray[Any, np.float64],
        a_shape: np.ndarray[Any, np.int64],
        a_strides: np.ndarray[Any, np.int64],
        b_storage: np.ndarray[Any, np.float64],
        b_shape: np.ndarray[Any, np.int64],
        b_strides: np.ndarray[Any, np.int64]) -> None:
    """
    NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as:

        assert a_shape[-1] == b_shape[-2]

    Optimizations:

        * Outer loop in parallel
        * No index buffers or function calls
        * Inner loop should have no global writes, 1 multiply.


    Args:
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    for i in prange(len(out)):
        for j in prange(a_shape[-1]):
            a_pos = (a_batch_stride * (i // out_strides[0]) + a_strides[-2] * ((i % out_strides[0]) // out_strides[1]) + a_strides[-1] * j)
            b_pos = (b_batch_stride * (i // out_strides[0]) + b_strides[-2] * j + b_strides[-1] * ((i % out_strides[0]) % out_strides[1]))

            out[i] += a_storage[a_pos] * b_storage[b_pos]


def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
    """
    Batched tensor matrix multiply ::

        for n:
          for i:
            for j:
              for k:
                out[n, i, j] += a[n, i, k] * b[n, k, j]

    Where n indicates an optional broadcasted batched dimension.

    Should work for tensor shapes of 3 dims ::

        assert a.shape[-1] == b.shape[-2]

    Args:
        a (:class:`Tensor`): tensor data a
        b (:class:`Tensor`): tensor data b

    Returns:
        :class:`Tensor` : new tensor data
    """

    # Make these always be a 3 dimensional multiply
    both_2d = 0
    if len(a.shape) == 2:
        a = a.contiguous().view(1, a.shape[0], a.shape[1])
        both_2d += 1
    if len(b.shape) == 2:
        b = b.contiguous().view(1, b.shape[0], b.shape[1])
        both_2d += 1
    both_2d = both_2d == 2

    ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
    ls.append(a.shape[-2])
    ls.append(b.shape[-1])
    assert a.shape[-1] == b.shape[-2]
    out = a.zeros(tuple(ls))

    tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

    # Undo 3d if we added it.
    if both_2d:
        out = out.view(out.shape[1], out.shape[2])
    return out


class FastOps:
    map = map
    zip = zip
    reduce = reduce
    matrix_multiply = matrix_multiply
